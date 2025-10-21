#include <Uefi.h>
#include <Library/BaseMemoryLib.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseLib.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/LoadedImage.h>
#include <Protocol/GraphicsOutput.h>
#include <Guid/FileInfo.h>
#include <Guid/Acpi.h>
#include <elf.h>

#include "pubkey_der.h"
#include "mbedtls/pk.h"
#include "mbedtls/sha256.h"

// ================= BootInfo (ABI 고정) =================
typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
    UINT8  verified;
    UINT8  kernel_hash[32];

    // NEW: ExitBootServices 이후에도 커널이 읽을 수 있게 사본 전달
    VOID*  MemoryMap;        // 커널이 읽을 사본 주소
    UINTN  MemoryMapSize;    // 바이트 단위
    UINTN  DescriptorSize;
    UINT32 ABI_Version;      // 호환성용 (예: 1)
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
    unsigned char hash[32];
    mbedtls_pk_context pk;

    mbedtls_pk_init(&pk);
    ret = mbedtls_pk_parse_public_key(&pk,
        (const unsigned char*)pubkey_der,
        (size_t)pubkey_der_len);
    if (ret != 0) {
        Print(L"[SECURE BOOT] Failed to parse public key (%d)\n", ret);
        mbedtls_pk_free(&pk);
        return EFI_SECURITY_VIOLATION;
    }

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

    ret = mbedtls_pk_verify(&pk, MBEDTLS_MD_SHA256, hash, 32,
                            (const unsigned char*)sigBuf, sigSize);
    mbedtls_pk_free(&pk);

    if (ret != 0) {
        Print(L"[SECURE BOOT] signature invalid (%d)\n", ret);
        return EFI_SECURITY_VIOLATION;
    }

    bi->verified = 1;
    CopyMem(bi->kernel_hash, hash, 32);
    Print(L"[SECURE BOOT] signature OK\n");
    return EFI_SUCCESS;
}

// ---------------- ELF Loader ----------------
// NOTE: 커널이 ID-맵(물리=가상)로 링크되었다면 p_vaddr==p_paddr.
// 일반적으로는 p_vaddr 기준 적재가 자연스럽다.
STATIC
EFI_STATUS LoadELFSegments(VOID* elfBase) {
    Elf64_Ehdr* ehdr = (Elf64_Ehdr*)elfBase;

    if (ehdr->e_ident[EI_MAG0]!=ELFMAG0 || ehdr->e_ident[EI_MAG1]!=ELFMAG1 ||
        ehdr->e_ident[EI_MAG2]!=ELFMAG2 || ehdr->e_ident[EI_MAG3]!=ELFMAG3) {
        Print(L"[ELF] Invalid ELF\n"); return EFI_LOAD_ERROR;
    }
    if (ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
        Print(L"[ELF] Not 64-bit\n"); return EFI_LOAD_ERROR;
    }

    Elf64_Phdr* phdr = (Elf64_Phdr*)((UINT8*)elfBase + ehdr->e_phoff);

    for (int i = 0; i < ehdr->e_phnum; i++) {
        if (phdr[i].p_type != PT_LOAD) continue;

        // 목적지 예약: p_vaddr 기준으로 '그 주소'를 내가 쓰겠다고 UEFI에 알림
        EFI_PHYSICAL_ADDRESS dst_page = (EFI_PHYSICAL_ADDRESS)(phdr[i].p_vaddr & ~0xFFFULL);
        UINTN page_off = (UINTN)(phdr[i].p_vaddr & 0xFFFULL);
        UINTN need_bytes = (UINTN)phdr[i].p_memsz + page_off;
        UINTN pages = EFI_SIZE_TO_PAGES(need_bytes);

        EFI_STATUS s = gBS->AllocatePages(AllocateAddress, EfiLoaderData, pages, &dst_page);
        if (EFI_ERROR(s)) {
            Print(L"[ELF] AllocatePages seg%d vaddr=0x%lx pages=%u fail %r\n",
                  i, phdr[i].p_vaddr, (UINT32)pages, s);
            return s;
        }

        VOID* dst = (VOID*)(phdr[i].p_vaddr);
        VOID* src = (VOID*)((UINT8*)elfBase + phdr[i].p_offset);

        CopyMem(dst, src, (UINTN)phdr[i].p_filesz);
        if (phdr[i].p_memsz > phdr[i].p_filesz) {
            SetMem((UINT8*)dst + phdr[i].p_filesz,
                   (UINTN)(phdr[i].p_memsz - phdr[i].p_filesz), 0);
        }

        Print(L"[ELF] seg%d vaddr=0x%lx filesz=%lu memsz=%lu pages=%u\n",
              i, phdr[i].p_vaddr, phdr[i].p_filesz, phdr[i].p_memsz, (UINT32)pages);
    }
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
    UINT8 *SigBuffer = NULL;
    UINTN KernelSize, SigSize;
    EFI_GRAPHICS_OUTPUT_PROTOCOL* Gop = NULL;
    KernelEntry EntryPoint;

    Print(L"[UEFI] Skylar BootLoader start\n");

    // Filesystem
    Status = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
    if (EFI_ERROR(Status)) return Status;

    Status = gBS->HandleProtocol(LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&FileSystem);
    if (EFI_ERROR(Status)) return Status;

    Status = FileSystem->OpenVolume(FileSystem, &RootDir);
    if (EFI_ERROR(Status)) return Status;

    // Load kernel.elf -> Pool buffer
    Status = RootDir->Open(RootDir, &KernelFile, L"kernel.elf", EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) { Print(L"[ERROR] Cannot open kernel.elf\n"); return Status; }

    Status = KernelFile->GetInfo(KernelFile, &gEfiFileInfoGuid, &FileInfoSize, NULL);
    if (Status != EFI_BUFFER_TOO_SMALL) return Status;

    Status = gBS->AllocatePool(EfiLoaderData, FileInfoSize, (VOID**)&FileInfo);
    if (EFI_ERROR(Status)) return Status;

    Status = KernelFile->GetInfo(KernelFile, &gEfiFileInfoGuid, &FileInfoSize, FileInfo);
    if (EFI_ERROR(Status)) return Status;

    KernelSize = FileInfo->FileSize;

    VOID* ElfBuf = NULL;
    Status = gBS->AllocatePool(EfiLoaderData, KernelSize, &ElfBuf);
    if (EFI_ERROR(Status)) { Print(L"[ERROR] AllocatePool ElfBuf\n"); return Status; }

    UINTN toRead = KernelSize;
    Status = KernelFile->Read(KernelFile, &toRead, ElfBuf);
    if (EFI_ERROR(Status) || toRead != KernelSize) {
        Print(L"[ERROR] Read kernel.elf failed (%r) read=%u\n", Status, (UINT32)toRead);
        return EFI_LOAD_ERROR;
    }
    Print(L"[INFO] Kernel ELF loaded into pool buffer (%u bytes)\n", (UINT32)KernelSize);

    // Load kernel.sig -> Pool buffer
    Status = RootDir->Open(RootDir, &SigFile, L"kernel.sig", EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) {
        Print(L"[SECURE BOOT] kernel.sig not found — abort\n");
        for(;;){ __asm__ __volatile__("hlt"); }
    }

    Status = SigFile->GetInfo(SigFile, &gEfiFileInfoGuid, &SigInfoSize, NULL);
    if (Status != EFI_BUFFER_TOO_SMALL) return Status;

    Status = gBS->AllocatePool(EfiLoaderData, SigInfoSize, (VOID**)&SigInfo);
    if (EFI_ERROR(Status)) return Status;

    Status = SigFile->GetInfo(SigFile, &gEfiFileInfoGuid, &SigInfoSize, SigInfo);
    if (EFI_ERROR(Status)) return Status;

    SigSize = SigInfo->FileSize;

    Status = gBS->AllocatePool(EfiLoaderData, SigSize, (VOID**)&SigBuffer);
    if (EFI_ERROR(Status)) return Status;

    toRead = SigSize;
    Status = SigFile->Read(SigFile, &toRead, SigBuffer);
    if (EFI_ERROR(Status) || toRead != SigSize) {
        Print(L"[SECURE BOOT] read sig failed (%r)\n", Status);
        for(;;){ __asm__ __volatile__("hlt"); }
    }
    Print(L"[SECURE BOOT] kernel.sig loaded (%u bytes)\n", (UINT32)SigSize);

    // Secure Boot Check (ElfBuf로 검증)
    BootInfo* bi;
    Status = gBS->AllocatePool(EfiLoaderData, sizeof(BootInfo), (VOID**)&bi);
    if (EFI_ERROR(Status)) return Status;
    ZeroMem(bi, sizeof(BootInfo));
    bi->ABI_Version = 1;

    Status = VerifyKernelSignature((VOID*)ElfBuf, KernelSize, SigBuffer, SigSize, bi);
    if (EFI_ERROR(Status)) {
        Print(L"[SECURE BOOT] INVALID - Halting.\n");
        for(;;){ __asm__ __volatile__("hlt"); }
    }

    // Framebuffer → BootInfo
    Status = gBS->LocateProtocol(&gEfiGraphicsOutputProtocolGuid, NULL, (VOID**)&Gop);
    if (EFI_ERROR(Status)) { Print(L"Failed to get GOP\n"); return Status; }

    bi->FrameBufferBase      = (VOID*)Gop->Mode->FrameBufferBase;
    bi->HorizontalResolution = Gop->Mode->Info->HorizontalResolution;
    bi->VerticalResolution   = Gop->Mode->Info->VerticalResolution;
    bi->PixelsPerScanLine    = Gop->Mode->Info->PixelsPerScanLine;

    Print(L"[GOP] FB=0x%lx, Res=%dx%d, Stride=%d\n",
          bi->FrameBufferBase,
          bi->HorizontalResolution,
          bi->VerticalResolution,
          bi->PixelsPerScanLine);

    // ELF Segments → p_vaddr로 예약+복사 (ElfBuf 사용)
    Status = LoadELFSegments(ElfBuf);
    if (EFI_ERROR(Status)) { Print(L"[ELF] load failed\n"); return Status; }

    // Entry (ElfBuf의 e_entry)
    Elf64_Ehdr* ehdr = (Elf64_Ehdr*)ElfBuf;
    EntryPoint = (KernelEntry)(ehdr->e_entry);
    Print(L"[BOOT] Jumping to kernel entry: 0x%lx\n", ehdr->e_entry);

    // 여기까지 파일/풀 정리: 최종 GetMemoryMap 전에만 OK
    if (KernelFile) KernelFile->Close(KernelFile);
    if (SigFile)    SigFile->Close(SigFile);
    if (FileInfo)   gBS->FreePool(FileInfo);
    if (SigInfo)    gBS->FreePool(SigInfo);
    // SigBuffer/ElfBuf/bi 는 커널 점프 직전까지 유지 (해제 X)

    // ======== Memory Map (사본 만들기) ========
    UINTN MapSize = 0, MapKey = 0, DescriptorSize = 0;
    UINT32 DescriptorVersion = 0;
    EFI_MEMORY_DESCRIPTOR* TempMap = NULL; // 임시 버퍼(풀)
    EFI_STATUS s;

    // 0) 크기 파악
    s = gBS->GetMemoryMap(&MapSize, NULL, &MapKey, &DescriptorSize, &DescriptorVersion);
    if (s != EFI_BUFFER_TOO_SMALL) { Print(L"[MMAP] probe failed %r\n", s); return s; }

    // 여유분
    UINTN Slack = DescriptorSize * 16;
    MapSize += Slack;

    // 1) 사본 저장용 페이지 미리 할당 (EBS 이후에도 유효)
    EFI_PHYSICAL_ADDRESS MMapCopy = 0;
    UINTN Pages = EFI_SIZE_TO_PAGES(MapSize);
    s = gBS->AllocatePages(AllocateAnyPages, EfiLoaderData, Pages, &MMapCopy);
    if (EFI_ERROR(s)) { Print(L"[MMAP] copy alloc failed %r\n", s); return s; }

    // 2) 최종 GetMemoryMap을 담을 임시 버퍼(풀)도 미리 할당
    s = gBS->AllocatePool(EfiLoaderData, MapSize, (VOID**)&TempMap);
    if (EFI_ERROR(s)) { Print(L"[MMAP] temp alloc failed %r\n", s); return s; }

    // 3) 최종 GetMemoryMap
    s = gBS->GetMemoryMap(&MapSize, TempMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    if (EFI_ERROR(s)) { Print(L"[MMAP] final get failed %r\n", s); return s; }

    // 4) 사본 페이지에 복사 (복사는 OK, 할당/해제 금지)
    CopyMem((VOID*)MMapCopy, TempMap, MapSize);

    bi->MemoryMap      = (VOID*)MMapCopy;
    bi->MemoryMapSize  = MapSize;
    bi->DescriptorSize = DescriptorSize;

    // 5) 즉시 ExitBootServices
    s = gBS->ExitBootServices(ImageHandle, MapKey);
    if (EFI_ERROR(s)) {
        // 재시도 루틴
        UINTN NewSize = 0, NewKey = 0, NewDescSize = 0; UINT32 NewDescVer = 0;
        s = gBS->GetMemoryMap(&NewSize, NULL, &NewKey, &NewDescSize, &NewDescVer);
        if (s != EFI_BUFFER_TOO_SMALL) {Print(L"[EBS] probe failed %r\n", s);for(;;){__asm__ __volatile__("hlt");} }

        if (NewSize > MapSize) {
            // TempMap 재할당
            gBS->FreePool(TempMap);
            NewSize += NewDescSize * 16;
            s = gBS->AllocatePool(EfiLoaderData, NewSize, (VOID**)&TempMap);
            if (EFI_ERROR(s)) { Print(L"[EBS] temp realloc failed %r\n", s); for(;;){__asm__ __volatile__("hlt");} }

            // MMapCopy 페이지도 부족하면 재할당
            if (NewSize > EFI_PAGES_TO_SIZE(Pages)) {
                EFI_PHYSICAL_ADDRESS NewCopy = 0;
                s = gBS->AllocatePages(AllocateAnyPages, EfiLoaderData, EFI_SIZE_TO_PAGES(NewSize), &NewCopy);
                if (EFI_ERROR(s)) { Print(L"[EBS] copy realloc failed %r\n", s); for(;;){__asm__ __volatile__("hlt");} }
                MMapCopy = NewCopy;
                Pages = EFI_SIZE_TO_PAGES(NewSize);
            }
            MapSize = NewSize;
            DescriptorSize = NewDescSize;
        }

        // 최종 GetMemoryMap (다시)
        s = gBS->GetMemoryMap(&MapSize, TempMap, &MapKey, &DescriptorSize, &DescriptorVersion);
        if (EFI_ERROR(s)) { Print(L"[EBS] final get failed %r\n", s); for(;;){__asm__ __volatile__("hlt");} }

        // 복사 → 즉시 EBS
        CopyMem((VOID*)MMapCopy, TempMap, MapSize);

        s = gBS->ExitBootServices(ImageHandle, MapKey);
        if (EFI_ERROR(s)) { Print(L"[EBS] failed %r\n", s); for(;;){__asm__ __volatile__("hlt");} }
    }

    // Jump!
    EntryPoint(bi);
    return EFI_SUCCESS;
}
