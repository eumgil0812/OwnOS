#include <stdint.h>
#include "kprintf.h"
#include "idt.h"
#include "fb.h"    // BootInfo for printing
extern BootInfo* g_bootinfo;  // kernel_main에서 전역 저장해두자

/* PIC ports */
#define PIC1_CMD 0x20
#define PIC1_DAT 0x21
#define PIC2_CMD 0xA0
#define PIC2_DAT 0xA1
#define PIC_EOI  0x20

static inline void outb(uint16_t p, uint8_t v){ __asm__ __volatile__("outb %0,%1"::"a"(v),"Nd"(p)); }
static inline uint8_t inb(uint16_t p){ uint8_t r; __asm__ __volatile__("inb %1,%0":"=a"(r):"Nd"(p)); return r; }

static void pic_remap(void){
    uint8_t a1 = inb(PIC1_DAT), a2 = inb(PIC2_DAT);

    outb(PIC1_CMD, 0x11);
    outb(PIC2_CMD, 0x11);
    outb(PIC1_DAT, 0x20); // Master offset 0x20
    outb(PIC2_DAT, 0x28); // Slave  offset 0x28
    outb(PIC1_DAT, 0x04);
    outb(PIC2_DAT, 0x02);
    outb(PIC1_DAT, 0x01);
    outb(PIC2_DAT, 0x01);

    outb(PIC1_DAT, a1);
    outb(PIC2_DAT, a2);
}

static volatile uint64_t ticks = 0;

static void pit_init_100hz(void){
    uint16_t div = 1193182 / 100; // PIT 100Hz
    outb(0x43, 0x36);             // ch0, lo/hi, mode3
    outb(0x40, div & 0xFF);
    outb(0x40, div >> 8);
}

/* C-level ISR 공용 핸들러 */
void isr_handler_c(uint64_t vec, uint64_t err){
    if (vec < 32) {
        kprintf(g_bootinfo, "[EXC] vec=%llu err=%llu\n",
                (unsigned long long)vec, (unsigned long long)err);
        for(;;){ __asm__ __volatile__("hlt"); }
    } else {
        /* IRQ */
        if (vec == 32){   // IRQ0 PIT
            ticks++;
            if ((ticks % 100) == 0) {
                kprintf(g_bootinfo, "[TICK] %llu\n", (unsigned long long)ticks);
            }
            outb(PIC1_CMD, PIC_EOI);
        } else if (vec == 33){ // IRQ1 Keyboard(선택)
            outb(PIC1_CMD, PIC_EOI);
        } else if (vec >= 40 && vec <= 47){
            outb(PIC2_CMD, PIC_EOI);
            outb(PIC1_CMD, PIC_EOI);
        } else {
            outb(PIC1_CMD, PIC_EOI);
        }
    }
}

/* 초기가동 */
void interrupts_init(void){
    idt_init();
    isr_install();
    irq_install();
    pic_remap();
    pit_init_100hz();
    __asm__ __volatile__("sti");
}
