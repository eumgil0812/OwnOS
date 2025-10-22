/*
 * ==========================================================
 *  Virtual Memory Manager (VMM)
 *  - Sets up 4-level paging for x86_64
 *  - Uses 2MiB large pages for kernel identity mapping
 *  - Builds page tables dynamically via PMM allocation
 *
 *  Author : Skylar
 *  Purpose: Early-stage kernel virtual memory initialization
 * ==========================================================
 */
#include "vmm.h"
#include "kprintf.h"
#include <string.h>
#include <stdbool.h>

#define PAGE_SIZE_4K   0x1000ULL
#define PAGE_SIZE_2M   0x200000ULL

// PTE/PMD/PDPT/PML4 엔트리 비트
#define PTE_P   (1ULL<<0)   // Present
#define PTE_W   (1ULL<<1)   // Write
#define PTE_U   (1ULL<<2)   // User (지금은 0으로)
#define PTE_PWT (1ULL<<3)
#define PTE_PCD (1ULL<<4)
#define PTE_A   (1ULL<<5)
#define PTE_D   (1ULL<<6)
#define PTE_PS  (1ULL<<7)   // Page Size (PD 엔트리에 2MiB)
#define PTE_G   (1ULL<<8)   // Global
#define PTE_NX  (1ULL<<63)  // NX (사용 시)

static uint64_t* g_pml4 = NULL;

// 간단 인라인: CR3 로드
static inline void write_cr3(uint64_t phys) {
    __asm__ __volatile__("mov %0, %%cr3" :: "r"(phys) : "memory");
}

// 4KiB 페이지 하나를 PMM에서 받아 0으로 클리어하고 물리주소 리턴
static uint64_t new_zero_page_phys(void) {
    void* pa = pmm_alloc_page();
    if (!pa) return 0;
    // 아이덴티티 매핑 전이므로 '물리주소 == 현재 접근 가능한 주소' 전제
    // 초기에는 FW의 기존 테이블 덕분에 메모리 접근 가능.
    memset((void*)(uintptr_t)pa, 0, PAGE_SIZE_4K);
    return (uint64_t)(uintptr_t)pa;
}

// PML4[pl4i]가 가리키는 PDPT가 없으면 만들고 포인터 리턴
static uint64_t* get_or_make_pdpt(uint64_t* pml4, uint64_t pl4i) {
    if (!(pml4[pl4i] & PTE_P)) {
        uint64_t pdpt_pa = new_zero_page_phys();
        pml4[pl4i] = (pdpt_pa) | PTE_P | PTE_W;
    }
    uint64_t pdpt_pa = pml4[pl4i] & 0x000FFFFFFFFFF000ULL;
    return (uint64_t*)(uintptr_t)pdpt_pa;
}

// PDPT[pdpti]가 가리키는 PD가 없으면 만들고 포인터 리턴
static uint64_t* get_or_make_pd(uint64_t* pdpt, uint64_t pdpti) {
    if (!(pdpt[pdpti] & PTE_P)) {
        uint64_t pd_pa = new_zero_page_phys();
        pdpt[pdpti] = (pd_pa) | PTE_P | PTE_W;
    }
    uint64_t pd_pa = pdpt[pdpti] & 0x000FFFFFFFFFF000ULL;
    return (uint64_t*)(uintptr_t)pd_pa;
}

// [phys_start, phys_start+size) 범위를 2MiB 대페이지로 아이덴티티 매핑
void vmm_map_range_2m(uint64_t phys_start, uint64_t size_bytes) {
    uint64_t start = phys_start & ~(PAGE_SIZE_2M - 1);
    uint64_t end   = (phys_start + size_bytes + PAGE_SIZE_2M - 1) & ~(PAGE_SIZE_2M - 1);

    for (uint64_t addr = start; addr < end; addr += PAGE_SIZE_2M) {
        // 가상=물리 아이덴티티
        uint64_t v = addr;

        // 인덱스 계산
        uint64_t pl4i  = (v >> 39) & 0x1FF;
        uint64_t pdpti = (v >> 30) & 0x1FF;
        uint64_t pdi   = (v >> 21) & 0x1FF;

        uint64_t* pdpt = get_or_make_pdpt(g_pml4, pl4i);
        uint64_t* pd   = get_or_make_pd(pdpt, pdpti);

        if (!(pd[pdi] & PTE_P)) {
            // 2MiB 대페이지: PD 엔트리에 PS=1, 베이스=물리주소
            pd[pdi] = (addr & 0x000FFFFFFFE00000ULL) | PTE_P | PTE_W | PTE_PS;
        }
    }
}

void vmm_init(BootInfo* bi, uint64_t map_gb) {
    if (map_gb == 0) map_gb = 1; // 최소 1GiB

    // 1) 새 PML4 1페이지 확보
    uint64_t pml4_pa = new_zero_page_phys();
    g_pml4 = (uint64_t*)(uintptr_t)pml4_pa;

    // 2) 0 ~ map_gb GiB 아이덴티티 2MiB 매핑
    uint64_t map_bytes = map_gb * (1ULL << 30);
    vmm_map_range_2m(0, map_bytes);

    // 3) 프레임버퍼도 포함되도록 별도 매핑(안전하게)
    uint64_t fb_phys = (uint64_t)(uintptr_t)bi->FrameBufferBase;
    uint64_t fb_size = (uint64_t)bi->PixelsPerScanLine * bi->VerticalResolution * 4ULL;
    vmm_map_range_2m(fb_phys, fb_size);

    // 4) CR3 교체
    write_cr3(pml4_pa);

    kprintf(bi, "[VMM] PML4=0x%llx, identity %llu GiB mapped, FB @0x%llx (%llu KB)\n",
            (unsigned long long)pml4_pa,
            (unsigned long long)map_gb,
            (unsigned long long)fb_phys,
            (unsigned long long)(fb_size >> 10));
}


void vmm_map_page(uint64_t vaddr, uint64_t paddr, uint64_t flags)
{
    uint64_t pl4i  = (vaddr >> 39) & 0x1FF;
    uint64_t pdpti = (vaddr >> 30) & 0x1FF;
    uint64_t pdi   = (vaddr >> 21) & 0x1FF;
    uint64_t pti   = (vaddr >> 12) & 0x1FF;

    uint64_t* pdpt = get_or_make_pdpt(g_pml4, pl4i);
    uint64_t* pd   = get_or_make_pd(pdpt, pdpti);

    // PD[pdi]가 2MiB 대페이지면 쪼개야 함
    if (pd[pdi] & PTE_PS) {
        // 기존 2MiB 매핑 제거 → 새로운 PT 생성
        uint64_t old_base = pd[pdi] & 0x000FFFFFFFE00000ULL;
        uint64_t* pt = (uint64_t*)(uintptr_t)new_zero_page_phys();
        for (uint64_t i = 0; i < 512; i++) {
            pt[i] = (old_base + i * PAGE_SIZE_4K) | PTE_P | PTE_W;
        }
        pd[pdi] = ((uint64_t)(uintptr_t)pt) | PTE_P | PTE_W;
    }

    // PT 주소 얻기
    uint64_t* pt = (uint64_t*)(uintptr_t)(pd[pdi] & 0x000FFFFFFFFFF000ULL);

    // 실제 4KiB 매핑
    pt[pti] = (paddr & 0x000FFFFFFFFFF000ULL) | flags | PTE_P;
}


void vmm_unmap_page(uint64_t vaddr)
{
    uint64_t pl4i  = (vaddr >> 39) & 0x1FF;
    uint64_t pdpti = (vaddr >> 30) & 0x1FF;
    uint64_t pdi   = (vaddr >> 21) & 0x1FF;
    uint64_t pti   = (vaddr >> 12) & 0x1FF;

    uint64_t* pdpt = (uint64_t*)(uintptr_t)(g_pml4[pl4i] & 0x000FFFFFFFFFF000ULL);
    if (!pdpt) return;

    uint64_t* pd = (uint64_t*)(uintptr_t)(pdpt[pdpti] & 0x000FFFFFFFFFF000ULL);
    if (!pd) return;

    if (pd[pdi] & PTE_PS) return; // 2MiB 대페이지는 생략

    uint64_t* pt = (uint64_t*)(uintptr_t)(pd[pdi] & 0x000FFFFFFFFFF000ULL);
    if (!pt) return;

    uint64_t entry = pt[pti];
    if (entry & PTE_P) {
        uint64_t phys = entry & 0x000FFFFFFFFFF000ULL;
        pmm_free_page((void*)(uintptr_t)phys);
        pt[pti] = 0;
        __asm__ volatile("invlpg (%0)" :: "r"(vaddr) : "memory");
    }
}

// 페이지 폴트 핸들러 (ISR에서 호출)
void vmm_handle_pagefault(uint64_t fault_addr, uint64_t error_code)
{
    // CR2: fault_addr, error_code 비트 의미
    // bit0: P=0 (not-present)
    // bit1: W=1 (write fault)
    // bit2: U=1 (user mode)
    // bit3: RSVD=1
    // bit4: I/D=1 (instruction fetch)

    kprintf(NULL, "[#PF] addr=0x%llx err=0x%llx\n",
            (unsigned long long)fault_addr,
            (unsigned long long)error_code);

    // 없는 페이지면 새로 할당
    if ((error_code & 0x1) == 0) {
        void* new_page = pmm_alloc_page();
        if (new_page) {
            vmm_map_page(PAGE_ALIGN_DOWN(fault_addr), (uint64_t)new_page, PTE_W);
            kprintf(NULL, "[#PF] Auto-allocated page @ 0x%llx\n",
                    (unsigned long long)PAGE_ALIGN_DOWN(fault_addr));
            return;
        }
    }

    // 실패 시 커널 패닉
    kprintf(NULL, "[#PF] Fatal: cannot handle page fault.\n");
    while (1) __asm__ volatile("hlt");
}

