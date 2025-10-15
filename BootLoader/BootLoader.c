// BootLoader.c (Secure Boot + GOP + FrameBuffer + Jump to Kernel)
#include <Uefi.h>
#include <Library/BaseMemoryLib.h>             // ✅ CopyMem, ZeroMem 사용
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseLib.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/LoadedImage.h>
#include <Protocol/GraphicsOutput.h>
#include <Guid/FileInfo.h>
#include <Guid/Acpi.h>

#include "pubkey_der.h"
#include "mbedtls/pk.h"
#include "mbedtls/sha256.h"

typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
    uint8_t verified;
    uint8_t kernel_hash[32];
} BootInfo;

typedef void (*KernelEntry)(BootInfo*);

// ---------------- Secure Boot verification ----------------
STATIC
EFI_STATUS VerifyKernelSignature(
    VOID* kernelBuf, UINTN kernelSize,
    UINT8* sigBuf, UINTN sigSize,
    BootInfo* bi
) {
    int ret;
    mbedtls_pk_context pk;
    unsigned char hash[32];

    mbedtls_pk_init(&pk);

    // parse embedded DER public key
    ret = mbedtls_pk_parse_public_key(&pk,
        (const unsigned char*)pubkey_der,
        (size_t)pubkey_der_len);
    if (ret != 0) {
        Print(L"[SECURE BOOT] Failed to parse public key (%d)\n", ret);
        mbedtls_pk_free(&pk);
        return EFI_SECURITY_VIOLATION;
    }

    // SHA-256 hash
#if defined(MBEDTLS_SHA256_ALT) || defined(MBEDTLS_SHA256_C)
    ret = mbedtls_sha256_ret((const unsigned char*)kernelBuf, kernelSize, hash, 0);
#else
    ret = mbedtls_sha256((const unsigned char*)kernelBuf, kernelSize, hash, 0);
#endif
    if (ret != 0) {
        Print(L"[SECURE BOOT] sha256 failed (%d)\n", ret);
        mbedtls_pk_free(&pk);
        return EFI_SECURITY_VIOLATION;
    }

    // Verify signature
    ret = mbedtls_pk_verify(&pk, MBEDTLS_MD_SHA256, hash, 0,
                            (const unsigned char*)sigBuf, sigSize);
    mbedtls_pk_free(&pk);

    if (ret != 0) {
        Print(L"[SECURE BOOT] signature invalid (%d)\n", ret);
        return EFI_SECURITY_VIOLATION;
    }

    // ✅ OK — store hash into BootInfo
    bi->verified = 1;
    CopyMem(bi->kernel_hash, hash, 32);   // ✅ memcpy 대체
    Print(L"[SECURE BOOT] signature OK\n");

    return EFI_SUCCESS;
}

// ---------------- Main ----------------
EFI_STATUS EFIAPI UefiMain(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable) {
    EFI_STATUS Status;
    EFI_LOADED_IMAGE_PROTOCOL *LoadedImage;
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *FileSystem;
    EFI_FILE_PROTOCOL *RootDir = NULL, *KernelFile = NULL, *SigFile = NULL;
    EFI_FILE_INFO *FileInfo = NULL, *SigInfo = NULL;
    UINTN FileInfoSize = 0, SigInfoSize = 0;
    VOID *KernelBuffer = NULL;
    UINT8 *SigBuffer = NULL;
    UINTN KernelSize, SigSize;
    EFI_GRAPHICS_OUTPUT_PROTOCOL* Gop = NULL;
    
    BootInfo bi; ZeroMem(&bi, sizeof(BootInfo));   // ✅ memset 대체
    KernelEntry EntryPoint;

    Print(L"[UEFI] Skylar BootLoader start\n");

    // Filesystem
    Status = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
    if (EFI_ERROR(Status)) return Status;
    Status = gBS->HandleProtocol(LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&FileSystem);
    if (EFI_ERROR(Status)) return Status;
    Status = FileSystem->OpenVolume(FileSystem, &RootDir);
    if (EFI_ERROR(Status)) return Status;

    // kernel.elf
    Status = RootDir->Open(RootDir, &KernelFile, L"kernel.elf", EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) {
        Print(L"[ERROR] Cannot open kernel.elf\n");
        return Status;
    }

    Status = KernelFile->GetInfo(KernelFile, &gEfiFileInfoGuid, &FileInfoSize, NULL);
    if (Status == EFI_BUFFER_TOO_SMALL) {
        gBS->AllocatePool(EfiLoaderData, FileInfoSize, (VOID**)&FileInfo);
        KernelFile->GetInfo(KernelFile, &gEfiFileInfoGuid, &FileInfoSize, FileInfo);
    }
    KernelSize = FileInfo->FileSize;
    gBS->AllocatePages(AllocateAnyPages, EfiLoaderData,
        EFI_SIZE_TO_PAGES(KernelSize), (EFI_PHYSICAL_ADDRESS*)&KernelBuffer);

    UINTN toRead = KernelSize;
    KernelFile->Read(KernelFile, &toRead, KernelBuffer);
    Print(L"[INFO] Kernel loaded at %p (%u bytes)\n", KernelBuffer, (UINT32)KernelSize);

    // kernel.sig
    Status = RootDir->Open(RootDir, &SigFile, L"kernel.sig", EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) {
        Print(L"[SECURE BOOT] kernel.sig not found — abort\n");
        for (;;) { __asm__ __volatile__("hlt"); }
    }
    SigFile->GetInfo(SigFile, &gEfiFileInfoGuid, &SigInfoSize, NULL);
    gBS->AllocatePool(EfiLoaderData, SigInfoSize, (VOID**)&SigInfo);
    SigFile->GetInfo(SigFile, &gEfiFileInfoGuid, &SigInfoSize, SigInfo);
    SigSize = SigInfo->FileSize;
    gBS->AllocatePool(EfiLoaderData, SigSize, (VOID**)&SigBuffer);
    toRead = SigSize;
    SigFile->Read(SigFile, &toRead, SigBuffer);
    Print(L"[SECURE BOOT] kernel.sig loaded (%u bytes)\n", (UINT32)SigSize);

    Status = VerifyKernelSignature(KernelBuffer, KernelSize, SigBuffer, SigSize, &bi);
    if (EFI_ERROR(Status)) {
        Print(L"[SECURE BOOT] INVALID - Halting.\n");
        for (;;) { __asm__ __volatile__("hlt"); }
    }

    // Framebuffer (dummy)
    /*
    bi.FrameBufferBase = (VOID*)0x00000000;
    bi.HorizontalResolution = 800;
    bi.VerticalResolution = 600;
    bi.PixelsPerScanLine = 800;
    */

    Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID**)&Gop);

    if (EFI_ERROR(Status)) {
        Print(L"Failed to get GOP\n");
        return Status;
    }

    bi.FrameBufferBase = (VOID*)Gop->Mode->FrameBufferBase;
    bi.HorizontalResolution = Gop->Mode->Info->HorizontalResolution;
    bi.VerticalResolution = Gop->Mode->Info->VerticalResolution;
    bi.PixelsPerScanLine = Gop->Mode->Info->PixelsPerScanLine;

    EntryPoint = (KernelEntry)KernelBuffer;

    // Exit Boot Services
    UINTN MapSize = 0, MapKey, DescriptorSize;
    UINT32 DescriptorVersion;
    EFI_MEMORY_DESCRIPTOR *MemMap = NULL;

    gBS->GetMemoryMap(&MapSize, MemMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    MapSize += DescriptorSize * 10;
    gBS->AllocatePool(EfiLoaderData, MapSize, (VOID**)&MemMap);
    gBS->GetMemoryMap(&MapSize, MemMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    
    
    
    gBS->Stall(5000000);   // 5,000,000 microseconds = 5 seconds

    gBS->ExitBootServices(ImageHandle, MapKey);

    EntryPoint(&bi);
    return EFI_SUCCESS;
}
