#include "memory.h"
#include "kprintf.h"  // kprintf(bi, ...) 사용 가능하도록
#include <stdint.h>

 const char* efi_type(uint32_t t) {
    switch (t) {
        case 6:  return "BootServicesCode";
        case 7:  return "Conventional";       // EfiConventionalMemory
        case 8:  return "BootServicesData";
        case 9:  return "LoaderCode";
        case 10: return "LoaderData";
        default: return "Other";
    }
}

void memmap_report(BootInfo* bi) {
    uint8_t* p = (uint8_t*)bi->MemoryMap;
    uint64_t total_pages = 0, conv_pages = 0;

    for (uint64_t off = 0; off < bi->MemoryMapSize; off += bi->DescriptorSize) {
        EFI_MEMORY_DESCRIPTOR* d = (EFI_MEMORY_DESCRIPTOR*)(p + off);
        total_pages += d->NumberOfPages;
        if (d->Type == 7)  // EfiConventionalMemory
            conv_pages += d->NumberOfPages;
    }

    uint64_t total_mb = (total_pages * 4096ULL) >> 20;
    uint64_t usable_mb = (conv_pages * 4096ULL) >> 20;

    kprintf(bi, "[RAM] total=%llu MB, usable=%llu MB\n",
            (unsigned long long)total_mb,
            (unsigned long long)usable_mb);
}
