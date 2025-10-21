#include "pmm.h"
#include "memory.h"   // EFI_MEMORY_DESCRIPTOR 정의 (Type==7)
#include "kprintf.h"  // 출력용
#include <string.h>   // memset

static pmm_state_t PMM;

static inline void bitmap_set(uint64_t idx)   { PMM.bitmap[idx >> 3] |=  (1u << (idx & 7)); }
static inline void bitmap_clear(uint64_t idx) { PMM.bitmap[idx >> 3] &= ~(1u << (idx & 7)); }
static inline int  bitmap_test(uint64_t idx)  { return (PMM.bitmap[idx >> 3] >> (idx & 7)) & 1u; }

static void mark_range_used(uint64_t phys, uint64_t size) {
    if (size == 0) return;
    uint64_t start = phys;
    uint64_t end   = phys + size;
    if (end <= PMM.base || start >= PMM.base + PMM.length) return; // 관리범위 밖
    if (start < PMM.base) start = PMM.base;
    if (end   > PMM.base + PMM.length) end = PMM.base + PMM.length;

    uint64_t first = (PAGE_ALIGN_DOWN(start) - PMM.base) / PAGE_SIZE;
    uint64_t last  = (PAGE_ALIGN_UP(end)    - PMM.base) / PAGE_SIZE;
    if (last > PMM.total_pages) last = PMM.total_pages;

    for (uint64_t i = first; i < last; ++i) {
        if (!bitmap_test(i)) { bitmap_set(i); PMM.used_pages++; }
    }
}

void pmm_init(BootInfo* bi, uint64_t min_phys) {
    // 1) 가장 큰 Conventional(7) 블록 선택 (min_phys 이상 고려)
    uint8_t* p = (uint8_t*)bi->MemoryMap;
    uint64_t best_start = 0, best_len = 0;

    for (uint64_t off = 0; off < bi->MemoryMapSize; off += bi->DescriptorSize) {
        EFI_MEMORY_DESCRIPTOR* d = (EFI_MEMORY_DESCRIPTOR*)(p + off);
        if (d->Type != 7) continue; // EfiConventionalMemory
        uint64_t start = (uint64_t)d->PhysicalStart;
        uint64_t end   = start + ((uint64_t)d->NumberOfPages << 12);
        if (end <= min_phys) continue;
        if (start < min_phys) start = min_phys;  // min_phys 이하 제외
        uint64_t len = end - start;
        if (len > best_len) { best_len = len; best_start = start; }
    }

    // 2) 상태 설정
    PMM.base   = PAGE_ALIGN_UP(best_start);
    PMM.length = PAGE_ALIGN_DOWN(best_start + best_len) - PMM.base;
    PMM.total_pages = PMM.length / PAGE_SIZE;
    PMM.used_pages  = 0;

    // 3) 비트맵 메모리 확보: 비트맵 자체는 커널의 단순 선형 힙(or 임시 전역 버퍼)로부터
    //    여기서는 커널 BSS에 여유 static 버퍼를 사용(데모 목적). 필요시 kmalloc로 바꿔.
    static uint8_t bitmap_storage[1 << 20]; // 1MB = 8,388,608 bits = 8,388,608 pages 커버(32GB) 데모용
    PMM.bitmap = bitmap_storage;
    PMM.bitmap_bytes = (PMM.total_pages + 7) / 8;
    if (PMM.bitmap_bytes > sizeof(bitmap_storage)) {
        // 너무 크면 범위를 강제로 줄인다(데모 방어).
        uint64_t max_pages = (sizeof(bitmap_storage) * 8ULL);
        PMM.total_pages = (max_pages < PMM.total_pages) ? max_pages : PMM.total_pages;
        PMM.length = PMM.total_pages * PAGE_SIZE;
        PMM.bitmap_bytes = (PMM.total_pages + 7) / 8;
    }
    memset(PMM.bitmap, 0, PMM.bitmap_bytes);

    // 4) 초기 예약: (a) 베이스 이전/이후는 밖이므로 무시됨. (b) 우리가 알고 있는 자원 예약 처리.
    //    - 커널 코드/데이터(대략 1MiB~현재 링크 범위) -> 필요시 링크 스크립트 심볼로 정확 표시 가능
    //    - 프레임버퍼 영역
    //    - 부트로더가 넘긴 MemoryMap 사본(물리주소는 모르면 생략 가능)
    // 프레임버퍼는 물리 주소로 알려져야 정확하지만, GOP Base는 보통 물리로 넘어온다(네 구조체 그대로라 가정).
    // 만약 가상주소면 아래 예약은 스킵.
    mark_range_used((uint64_t)(uintptr_t)bi->FrameBufferBase,
                    (uint64_t)bi->PixelsPerScanLine * bi->VerticalResolution * 4ULL);

    // (선택) 커널이 사용하는 저주소 대역(예: 0x100000~0x400000) 예약
    mark_range_used(0x00100000ULL, 0x00300000ULL);

    kprintf(bi, "[PMM] base=0x%llx, len=%llu MB, pages=%llu, bitmap=%llu bytes\n",
            (unsigned long long)PMM.base,
            (unsigned long long)(PMM.length >> 20),
            (unsigned long long)PMM.total_pages,
            (unsigned long long)PMM.bitmap_bytes);
}

void* pmm_alloc_page(void) {
    if (!PMM.bitmap || PMM.total_pages == 0) return NULL;

    // 단순 선형 스캔
    for (uint64_t i = 0; i < PMM.total_pages; ++i) {
        if (!bitmap_test(i)) {
            bitmap_set(i);
            PMM.used_pages++;
            return (void*)(uintptr_t)(PMM.base + i * PAGE_SIZE);
        }
    }
    return NULL; // 더 없음
}

void pmm_free_page(void* pa) {
    if (!pa) return;
    uint64_t phys = (uint64_t)(uintptr_t)pa;

    if (phys < PMM.base || phys >= PMM.base + PMM.length) return; // 관리 범위 밖
    if ((phys - PMM.base) & (PAGE_SIZE - 1)) return;              // 페이지 정렬 안됨

    uint64_t idx = (phys - PMM.base) / PAGE_SIZE;
    if (bitmap_test(idx)) {
        bitmap_clear(idx);
        if (PMM.used_pages) PMM.used_pages--;
    }
}

uint64_t pmm_total_pages(void)  { return PMM.total_pages; }
uint64_t pmm_used_pages(void)   { return PMM.used_pages;  }
uint64_t pmm_base_phys(void)    { return PMM.base;        }
uint64_t pmm_length_bytes(void) { return PMM.length;      }

void pmm_reserve_range(uint64_t phys, uint64_t size) {
    mark_range_used(phys, size);
}

void pmm_debug_dump(void (*print)(const char*)) {
    (void)print; // 필요 시 구현
}

