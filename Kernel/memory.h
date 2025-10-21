#pragma once
#include <stdint.h>
#include "fb.h"   // BootInfo 구조체 정의 포함되어 있어야 함

// UEFI 메모리 맵 디스크립터
typedef struct {
    uint32_t Type;
    uint32_t Pad;
    uint64_t PhysicalStart;
    uint64_t VirtualStart;
    uint64_t NumberOfPages;
    uint64_t Attribute;
} __attribute__((packed)) EFI_MEMORY_DESCRIPTOR;

// 메모리 타입 문자열 반환
const char* efi_type(uint32_t type);

// 메모리 맵 전체를 분석하고 RAM 총량/가용량 출력
void memmap_report(BootInfo* bi);
