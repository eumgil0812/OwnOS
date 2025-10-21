.global isr_stub_table
.global isr_common_stub
.extern isr_handler_c

/* 0..31 + 32..47(IRQ) 총 48개 */
.section .text
.isr_make_noerr:
    /* vec in %rdi, err=0 */
    pushq $0               /* fake error code */
    jmp  isr_common_stub

.isr_make_err:
    /* vec in %rdi, err already on stack by CPU */
    jmp  isr_common_stub

/* 테이블 */
.align 16
isr_stub_table:
    /* 0: #DE */  .quad isr0
    .quad isr1, isr2, isr3, isr4, isr5, isr6, isr7
    .quad isr8, isr9, isr10, isr11, isr12, isr13, isr14, isr15
    .quad isr16, isr17, isr18, isr19, isr20, isr21, isr22, isr23
    .quad isr24, isr25, isr26, isr27, isr28, isr29, isr30, isr31
    /* 32..47: IRQ0..IRQ15 */
    .quad irq0, irq1, irq2, irq3, irq4, irq5, irq6, irq7
    .quad irq8, irq9, irq10, irq11, irq12, irq13, irq14, irq15

/* 매크로 */
.macro ISR_NOERR n
.global isr\n
isr\n:
    pushq $0              /* align stack for common stub */
    pushq $\n             /* vec */
    jmp isr_common_stub
.endm

.macro ISR_ERR n
.global isr\n
isr\n:
    pushq $\n             /* vec */
    jmp isr_common_stub   /* error code already pushed by CPU */
.endm

/* 예외: 에러코드 없는 것들 */
ISR_NOERR 0
ISR_NOERR 1
ISR_NOERR 2
ISR_NOERR 3
ISR_NOERR 4
ISR_NOERR 5
ISR_NOERR 6
ISR_NOERR 7
/* 8 (#DF) 에러코드 O */
ISR_ERR   8
ISR_NOERR 9
ISR_ERR   10
ISR_ERR   11
ISR_ERR   12
ISR_ERR   13
ISR_ERR   14
ISR_NOERR 15
ISR_NOERR 16
ISR_ERR   17
ISR_NOERR 18
ISR_NOERR 19
ISR_NOERR 20
ISR_NOERR 21
ISR_NOERR 22
ISR_NOERR 23
ISR_NOERR 24
ISR_NOERR 25
ISR_NOERR 26
ISR_NOERR 27
ISR_NOERR 28
ISR_NOERR 29
ISR_ERR   30
ISR_NOERR 31

/* IRQ 0..15 */
.macro IRQ n
.global irq\n
irq\n:
    pushq $0
    pushq $(32+\n)
    jmp isr_common_stub
.endm

IRQ 0
IRQ 1
IRQ 2
IRQ 3
IRQ 4
IRQ 5
IRQ 6
IRQ 7
IRQ 8
IRQ 9
IRQ 10
IRQ 11
IRQ 12
IRQ 13
IRQ 14
IRQ 15

/* 공통 스텁: 저장 → C 핸들러 호출 → 복원 → iretq */
isr_common_stub:
    /* error code, vector 가 스택에: [rsp]=vec, [rsp+8]=err */
    pushq %rax
    pushq %rbx
    pushq %rcx
    pushq %rdx
    pushq %rsi
    pushq %rdi
    pushq %rbp
    pushq %r8
    pushq %r9
    pushq %r10
    pushq %r11
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15

    mov  16*8(%rsp), %rdi   /* vec */
    mov  17*8(%rsp), %rsi   /* err */
    call isr_handler_c

    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %r11
    popq %r10
    popq %r9
    popq %r8
    popq %rbp
    popq %rdi
    popq %rsi
    popq %rdx
    popq %rcx
    popq %rbx
    popq %rax

    add  $16, %rsp       /* pop vec + err */
    iretq
