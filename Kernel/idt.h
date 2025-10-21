#pragma once
#include <stdint.h>

typedef struct {
    uint16_t offset_low;
    uint16_t selector;
    uint8_t  ist;       // 0
    uint8_t  type_attr; // 0x8E = present,int gate
    uint16_t offset_mid;
    uint32_t offset_high;
    uint32_t zero;
} __attribute__((packed)) idt_entry_t;

typedef struct {
    uint16_t limit;
    uint64_t base;
} __attribute__((packed)) idtr_t;

void idt_init(void);
void isr_install(void);   // 예외 핸들러
void irq_install(void);   // IRQ(0~15) 핸들러
void enable_interrupts(void);
