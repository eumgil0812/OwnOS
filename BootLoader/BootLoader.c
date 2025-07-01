// BootLoader.c
#include <Uefi.h>
#include <Library/UefiLib.h>
#include <Library/UefiBootServicesTableLib.h>
#include <Library/MemoryAllocationLib.h>
#include <Library/BaseMemoryLib.h>
#include <Library/DevicePathLib.h>
#include <Protocol/SimpleFileSystem.h>
#include <Protocol/LoadedImage.h>
#include <Guid/FileInfo.h>

// Define FrameBufferInfo struct to pass to kernel
typedef struct {
    void* FrameBufferBase;
    unsigned int HorizontalResolution;
    unsigned int VerticalResolution;
    unsigned int PixelsPerScanLine;
} FrameBufferInfo;

typedef void (*KernelEntry)(FrameBufferInfo*);

EFI_STATUS EFIAPI UefiMain(IN EFI_HANDLE ImageHandle, IN EFI_SYSTEM_TABLE *SystemTable) {
    EFI_STATUS Status;
    EFI_GRAPHICS_OUTPUT_PROTOCOL *Gop;
    EFI_GUID gopGuid = EFI_GRAPHICS_OUTPUT_PROTOCOL_GUID;
    EFI_LOADED_IMAGE_PROTOCOL *LoadedImage;
    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *FileSystem;
    EFI_FILE_PROTOCOL *RootDir, *KernelFile;
    EFI_FILE_INFO *FileInfo;
    UINTN FileInfoSize = 0;
    VOID *KernelBuffer = NULL;
    KernelEntry EntryPoint;
    FrameBufferInfo fbInfo;

    Print(L"[UEFI] Skylar's BootLoader Starting...\n");

    // Get loaded image protocol
    Status = gBS->HandleProtocol(ImageHandle, &gEfiLoadedImageProtocolGuid, (VOID**)&LoadedImage);
    if (EFI_ERROR(Status)) return Status;

    // Get file system protocol
    Status = gBS->HandleProtocol(LoadedImage->DeviceHandle, &gEfiSimpleFileSystemProtocolGuid, (VOID**)&FileSystem);
    if (EFI_ERROR(Status)) return Status;

    // Open root directory
    Status = FileSystem->OpenVolume(FileSystem, &RootDir);
    if (EFI_ERROR(Status)) return Status;

    // Open kernel.elf
    Status = RootDir->Open(RootDir, &KernelFile, L"kernel.elf", EFI_FILE_MODE_READ, 0);
    if (EFI_ERROR(Status)) {
        Print(L"[ERROR] Cannot open kernel.elf\n");
        return Status;
    }

    // Get file size
    Status = KernelFile->GetInfo(KernelFile, &gEfiFileInfoGuid, &FileInfoSize, NULL);
    if (Status == EFI_BUFFER_TOO_SMALL) {
        Status = gBS->AllocatePool(EfiLoaderData, FileInfoSize, (VOID**)&FileInfo);
        if (EFI_ERROR(Status)) return Status;

        Status = KernelFile->GetInfo(KernelFile, &gEfiFileInfoGuid, &FileInfoSize, FileInfo);
        if (EFI_ERROR(Status)) return Status;
    }

    // Allocate buffer for kernel
    Status = gBS->AllocatePages(AllocateAnyPages, EfiLoaderData,
        EFI_SIZE_TO_PAGES(FileInfo->FileSize), (EFI_PHYSICAL_ADDRESS*)&KernelBuffer);
    if (EFI_ERROR(Status)) return Status;

    // Read kernel into buffer
    UINTN KernelSize = FileInfo->FileSize;
    Status = KernelFile->Read(KernelFile, &KernelSize, KernelBuffer);
    if (EFI_ERROR(Status)) return Status;

    Print(L"[INFO] Kernel loaded at address: %p\n", KernelBuffer);

 
    Status = gBS->LocateProtocol(&gopGuid, NULL, (VOID**)&Gop);
    if (EFI_ERROR(Status)) {
        Print(L"Failed to get GOP\n");
        return Status;
    }

    // Fill fbInfo with real data
    fbInfo.FrameBufferBase = (VOID*)Gop->Mode->FrameBufferBase;
    fbInfo.HorizontalResolution = Gop->Mode->Info->HorizontalResolution;
    fbInfo.VerticalResolution = Gop->Mode->Info->VerticalResolution;
    fbInfo.PixelsPerScanLine = Gop->Mode->Info->PixelsPerScanLine;

    // Entry point is at beginning for this simple binary
    EntryPoint = (KernelEntry)KernelBuffer;

    // Exit boot services
    UINTN MapSize = 0, MapKey, DescriptorSize;
    UINT32 DescriptorVersion;
    EFI_MEMORY_DESCRIPTOR *MemMap = NULL;

    gBS->GetMemoryMap(&MapSize, MemMap, &MapKey, &DescriptorSize, &DescriptorVersion);
    MapSize += DescriptorSize * 10;
    gBS->AllocatePool(EfiLoaderData, MapSize, (VOID**)&MemMap);
    gBS->GetMemoryMap(&MapSize, MemMap, &MapKey, &DescriptorSize, &DescriptorVersion);

    gBS->ExitBootServices(ImageHandle, MapKey);

    // Jump to kernel
    EntryPoint(&fbInfo);
    return EFI_SUCCESS;
}
