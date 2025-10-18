#include <stdarg.h>
#include <stdint.h>
#include "fb.h"

// 32-bit integer to string
static void itoa(unsigned int value, char* buffer, int base) {
    char* ptr = buffer;
    char* ptr1 = buffer;
    char tmp_char;
    unsigned int tmp_value;

    do {
        tmp_value = value;
        value /= base;
        *ptr++ = "0123456789ABCDEF"[tmp_value % base];
    } while (value);

    *ptr-- = '\0';
    while (ptr1 < ptr) {
        tmp_char = *ptr;
        *ptr-- = *ptr1;
        *ptr1++ = tmp_char;
    }
}

// 64-bit integer to string (for %llx, %llu)
static void itoa64(uint64_t value, char* buffer, int base) {
    char* ptr = buffer;
    char* ptr1 = buffer;
    char tmp_char;
    uint64_t tmp_value;

    do {
        tmp_value = value;
        value /= base;
        *ptr++ = "0123456789ABCDEF"[tmp_value % base];
    } while (value);

    *ptr-- = '\0';
    while (ptr1 < ptr) {
        tmp_char = *ptr;
        *ptr-- = *ptr1;
        *ptr1++ = tmp_char;
    }
}

// Minimal kernel printf
// Supports: %d, %x, %s, %c, %% + %llx, %llu (64-bit)
void kprintf(BootInfo* bi, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    char buf[64];

    for (const char* p = fmt; *p; p++) {
        if (*p == '%') {
            p++;

            // 🧭 64-bit format (%llx or %llu)
            if (*p == 'l' && *(p + 1) == 'l') {
                p += 2;
                switch (*p) {
                    case 'x': {  // 64-bit hex
                        uint64_t val = va_arg(args, uint64_t);
                        itoa64(val, buf, 16);
                        kputs_fb(bi, buf);
                        break;
                    }
                    case 'u': {  // 64-bit unsigned decimal
                        uint64_t val = va_arg(args, uint64_t);
                        itoa64(val, buf, 10);
                        kputs_fb(bi, buf);
                        break;
                    }
                    default:
                        kputs_fb(bi, "%ll?");
                        break;
                }
                continue;
            }

            switch (*p) {
                case 'd': {  // decimal
                    int val = va_arg(args, int);
                    if (val < 0) {
                        kputs_fb(bi, "-");
                        val = -val;
                    }
                    itoa((unsigned int)val, buf, 10);
                    kputs_fb(bi, buf);
                    break;
                }
                case 'x': {  // hex
                    unsigned int val = va_arg(args, unsigned int);
                    itoa(val, buf, 16);
                    kputs_fb(bi, buf);
                    break;
                }
                case 's': {  // string
                    char* str = va_arg(args, char*);
                    if (str) kputs_fb(bi, str);
                    else kputs_fb(bi, "(null)");
                    break;
                }
                case 'c': {  // char
                    char ch = (char)va_arg(args, int);
                    char tmp[2] = { ch, 0 };
                    kputs_fb(bi, tmp);
                    break;
                }
                case '%': {  // literal '%'
                    kputs_fb(bi, "%");
                    break;
                }
                default: {   // unknown format
                    kputs_fb(bi, "%");
                    char tmp[2] = { *p, 0 };
                    kputs_fb(bi, tmp);
                    break;
                }
            }
        } else {
            // Normal character
            char tmp[2] = { *p, 0 };
            kputs_fb(bi, tmp);
        }
    }

    va_end(args);
}
