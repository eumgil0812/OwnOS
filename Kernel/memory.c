#include "memory.h"
#include "kprintf.h"  // Allows use of kprintf(bi, ...)
#include <stdint.h>

#define KiB(x) ((uint64_t)(x) << 10)
#define MiB(x) ((uint64_t)(x) << 20)
#define GiB(x) ((uint64_t)(x) << 30)


const char* efi_type(uint32_t t) {
    switch (t) {
        case 0:  return "Reserved";
        case 1:  return "LoaderCode";
        case 2:  return "LoaderData";
        case 3:  return "BootServicesCode";
        case 4:  return "BootServicesData";
        case 5:  return "RuntimeServicesCode";
        case 6:  return "RuntimeServicesData";
        case 7:  return "Conventional";
        case 8:  return "Unusable";
        case 9:  return "ACPIReclaim";
        case 10: return "ACPINVS";
        default: return "Other";
    }
}

typedef struct {
    uint64_t base;
    uint64_t pages;
    uint32_t type;
    uint64_t attr;
} region_t;


void print_mem_attrs(BootInfo* bi, uint64_t a) {
    kputs_fb(bi, "[");
    int first = 1;
    #define P(flag, name) if (a & (flag)) { if (!first) kputs_fb(bi, "|"); kputs_fb(bi, name); first = 0; }
    P(EFI_MEMORY_UC, "UC");
    P(EFI_MEMORY_WC, "WC");
    P(EFI_MEMORY_WT, "WT");
    P(EFI_MEMORY_WB, "WB");
    P(EFI_MEMORY_UCE, "UCE");
    P(EFI_MEMORY_WP, "WP");
    P(EFI_MEMORY_RP, "RP");
    P(EFI_MEMORY_XP, "XP");
    P(EFI_MEMORY_NV, "NV");
    P(EFI_MEMORY_MORE_RELIABLE, "MR");
    P(EFI_MEMORY_RO, "RO");
    P(EFI_MEMORY_SP, "SP");
    P(EFI_MEMORY_RUNTIME, "RUNTIME");
    #undef P
    if (first) kputs_fb(bi, "-");
    kputs_fb(bi, "]");
}

uint64_t pages_to_bytes(uint64_t pages) { return pages * 4096ULL; }

void print_size_auto(BootInfo* bi, uint64_t bytes) {
    if (bytes >= GiB(1)) kprintf(bi, "%llu GiB", (unsigned long long)(bytes >> 30));
    else if (bytes >= MiB(1)) kprintf(bi, "%llu MiB", (unsigned long long)(bytes >> 20));
    else if (bytes >= KiB(1)) kprintf(bi, "%llu KiB", (unsigned long long)(bytes >> 10));
    else kprintf(bi, "%llu B", (unsigned long long)bytes);
}


static region_t find_largest_conventional(BootInfo* bi) {
    const uint8_t* p = (const uint8_t*)bi->MemoryMap;
    region_t best = {0,0,7,0};
    for (uint64_t off=0; off < bi->MemoryMapSize; off += bi->DescriptorSize) {
        const EFI_MEMORY_DESCRIPTOR* d = (const EFI_MEMORY_DESCRIPTOR*)(p + off);
        if (d->Type == 7 && d->NumberOfPages > best.pages) {
            best.base = d->PhysicalStart;
            best.pages = d->NumberOfPages;
            best.attr  = d->Attribute;
        }
    }
    return best;
}


void memmap_dump(BootInfo* bi, uint32_t limit) {
    const uint8_t* p = (const uint8_t*)bi->MemoryMap;
    uint32_t count = 0;

    kputs_fb(bi, "┌─[Memory Map Dump]───────────────────────────────────────────\n");
    kputs_fb(bi, "│  Type              Base               Pages      Size       Attr\n");
    kputs_fb(bi, "│  --------------------------------------------------------------\n");

    for (uint64_t off=0; off < bi->MemoryMapSize; off += bi->DescriptorSize) {
        const EFI_MEMORY_DESCRIPTOR* d = (const EFI_MEMORY_DESCRIPTOR*)(p + off);
        uint64_t sz = d->NumberOfPages * 4096ULL;

        kputs_fb(bi, "│  ");
        print_padded(bi, efi_type(d->Type), 16);
        kprintf(bi, "  0x%016llx %10llu ",
                (unsigned long long)d->PhysicalStart,
                (unsigned long long)d->NumberOfPages);

        print_size_auto(bi, sz);
        kprintf(bi, "  0x%llx ", (unsigned long long)d->Attribute);
        print_mem_attrs(bi, d->Attribute);
        kputs_fb(bi, "\n");

        if (++count >= limit) break;
    }
    kputs_fb(bi, "└───────────────────────────────────────────────────────────────\n");
}

void memmap_summary(BootInfo* bi) {
    const uint8_t* p = (const uint8_t*)bi->MemoryMap;
    uint64_t total_pages=0, usable_pages=0, regions=0;
    uint64_t type_count[16] = {0}; // 0~13만 대충 집계

    for (uint64_t off=0; off < bi->MemoryMapSize; off += bi->DescriptorSize) {
        const EFI_MEMORY_DESCRIPTOR* d = (const EFI_MEMORY_DESCRIPTOR*)(p + off);
        total_pages += d->NumberOfPages;
        if (d->Type == 7) usable_pages += d->NumberOfPages;
        if (d->Type < 16) type_count[d->Type]++;
        regions++;
    }

    uint64_t total_b  = pages_to_bytes(total_pages);
    uint64_t usable_b = pages_to_bytes(usable_pages);

    kputs_fb(bi, "┌─[Memory Summary]─────────────────────────────────────────────\n");
    kprintf(bi, "│  Regions: %llu\n", (unsigned long long)regions);
    kprintf(bi, "│  Total : "); print_size_auto(bi, total_b);  kputs_fb(bi, "\n");
    kprintf(bi, "│  Usable: "); print_size_auto(bi, usable_b); kputs_fb(bi, "  (Conventional)\n");

    // 타입별 집계 (필요한 것만)
    kputs_fb(bi, "│  By Type:\n");
    for (uint32_t t=0; t<=13; ++t) {
        if (!type_count[t]) continue;
        kputs_fb(bi, "│   - ");
        print_padded(bi, efi_type(t), 16);
        kprintf(bi, " : %llu regions\n", (unsigned long long)type_count[t]);

    }

    region_t best = find_largest_conventional(bi);
    if (best.pages) {
        kputs_fb(bi, "│  Largest Usable Region:\n");
        kprintf(bi, "│    base=0x%llx pages=%llu size=",
            (unsigned long long)best.base, (unsigned long long)best.pages);
        print_size_auto(bi, pages_to_bytes(best.pages));
        kprintf(bi, " attr=0x%llx\n", (unsigned long long)best.attr);
    }
    kputs_fb(bi, "└───────────────────────────────────────────────────────────────\n");
}

void memmap_report(BootInfo* bi) {
    // 1) Start address of the memory map table in bytes
    uint8_t* p = (uint8_t*)bi->MemoryMap;

    // 2) Accumulators for total pages and usable (Conventional) pages
    uint64_t total_pages = 0, conv_pages = 0;

    // 3) Iterate through the memory map table in steps of DescriptorSize
    for (uint64_t off = 0; off < bi->MemoryMapSize; off += bi->DescriptorSize) {
        EFI_MEMORY_DESCRIPTOR* d = (EFI_MEMORY_DESCRIPTOR*)(p + off);

        total_pages += d->NumberOfPages;        // Add to total page count
        if (d->Type == 7)                       // 7 = EfiConventionalMemory
            conv_pages += d->NumberOfPages;     // Add only usable memory pages
    }

    // 4) Convert from 4KB (4096-byte) pages to megabytes: (pages * 4096) >> 20
    uint64_t total_mb  = (total_pages * 4096ULL) >> 20;
    uint64_t usable_mb = (conv_pages  * 4096ULL) >> 20;

    // 5) Print the result
    kprintf(bi, "[RAM] total=%llu MB, usable=%llu MB\n",
            (unsigned long long)total_mb,
            (unsigned long long)usable_mb);
}
