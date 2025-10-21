#pragma once
#include <stdint.h>

// BootInfo는 다른 헤더에서 정의됨(예: fb.h/bootinfo.h)
// 여기서는 전방선언만 필요하면 아래 주석 해제
// typedef struct BootInfo BootInfo;

/**
 * 인터럽트 서브시스템 초기화:
 * - IDT 로드 및 ISR/IRQ 게이트 설치
 * - PIC 리맵(0x20/0x28)
 * - PIT 100Hz 설정
 * - STI (인터럽트 허용)
 */
void interrupts_init(void);

/**
 * 공통 ISR C 핸들러 (ASM 스텁에서 호출)
 * vec: 벡터 번호(예외 0~31, IRQ 32~47)
 * err: 에러 코드(없는 경우 0이 push됨)
 */
void isr_handler_c(uint64_t vec, uint64_t err);

// (선택) 타이머 틱을 외부에서 보고 싶다면 getter 제공 가능
// uint64_t isr_get_ticks(void);
