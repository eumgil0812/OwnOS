#include "idt.h"
#include <string.h>

#define IDT_MAX 256
extern void (*isr_stub_table[])(void); // isr.asm에서 제공 (0~31 + IRQ 32~47)

static idt_entry_t idt[IDT_MAX];
static idtr_t idtr;

static void set_gate(int vec, void (*handler)(void), uint8_t type_attr) {
    uint64_t addr = (uint64_t)handler;
    idt[vec].offset_low  = addr & 0xFFFF;
    idt[vec].selector    = 0x08;     // 커널 코드 세그먼트
    idt[vec].ist         = 0;
    idt[vec].type_attr   = type_attr;
    idt[vec].offset_mid  = (addr >> 16) & 0xFFFF;
    idt[vec].offset_high = (addr >> 32) & 0xFFFFFFFF;
    idt[vec].zero        = 0;
}

void idt_init(void) {
    memset(idt, 0, sizeof(idt));
    idtr.limit = sizeof(idt) - 1;
    idtr.base  = (uint64_t)idt;
    __asm__ __volatile__("lidt %0" : : "m"(idtr));
}

void isr_install(void) {
    for (int i=0;i<32;i++)
        set_gate(i, isr_stub_table[i], 0x8E);
}

void irq_install(void) {
    for (int i=32;i<48;i++)
        set_gate(i, isr_stub_table[i], 0x8E);
}

void enable_interrupts(void) { __asm__ __volatile__("sti"); }
