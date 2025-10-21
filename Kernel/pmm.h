#pragma once
#include <stdint.h>
#include <stddef.h>
#include "fb.h"   // BootInfo (MemoryMap, DescriptorSize 등)

#define PAGE_SIZE 4096ULL
#define PAGE_ALIGN_DOWN(x) ((uint64_t)(x) & ~(PAGE_SIZE-1))
#define PAGE_ALIGN_UP(x)   (((uint64_t)(x) + PAGE_SIZE-1) & ~(PAGE_SIZE-1))

typedef struct {
    uint8_t*   bitmap;        // 비트맵 시작 (커널 가상주소)
    uint64_t   bitmap_bytes;  // 비트맵 바이트 길이
    uint64_t   base;          // 관리 시작 물리주소
    uint64_t   length;        // 총 관리 길이(바이트)
    uint64_t   total_pages;   // length / PAGE_SIZE
    uint64_t   used_pages;    // 할당된 페이지 수
} pmm_state_t;

// 초기화: MemoryMap에서 가장 큰 Conventional 영역 선택 (min_phys 이상)
void pmm_init(BootInfo* bi, uint64_t min_phys);

// 페이지 단위 할당/해제
void* pmm_alloc_page(void);   // 4KB 한 페이지 반환 (물리주소)
void  pmm_free_page(void* pa);

// 통계/도움
uint64_t pmm_total_pages(void);
uint64_t pmm_used_pages(void);
uint64_t pmm_base_phys(void);
uint64_t pmm_length_bytes(void);

// 특정 물리 범위를 “예약(사용중)” 처리 (예: 커널, FB 등)
void pmm_reserve_range(uint64_t phys, uint64_t size);

// 디버그 출력(선택)
void pmm_debug_dump(void (*print)(const char*));
