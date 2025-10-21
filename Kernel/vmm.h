#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "pmm.h"
#include "fb.h"   // 또는 fb.h에 BootInfo가 있으면 그걸

// 페이징 초기화: 0 ~ map_gb GiB 아이덴티티 매핑 + 프레임버퍼 구간 매핑
void vmm_init(BootInfo* bi, uint64_t map_gb);

// 임의 물리 구간을 2MiB 단위로 아이덴티티 매핑 (RW)
void vmm_map_range_2m(uint64_t phys_start, uint64_t size_bytes);

// (선택) 1 페이지(4KiB) 매핑 버전이 필요하면 이후 추가
