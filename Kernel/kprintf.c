#include <stdarg.h>
#include "fb.h"

// Convert an unsigned integer to a string in the specified base
// Supports decimal (10) and hexadecimal (16)
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

    // Reverse the string
    while (ptr1 < ptr) {
        tmp_char = *ptr;
        *ptr-- = *ptr1;
        *ptr1++ = tmp_char;
    }
}

// Simple kernel printf-like function
// Supports %d, %x, %s, %c, and %%
void kprintf(BootInfo* bi, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    char buf[64];

    for (const char* p = fmt; *p; p++) {
        if (*p == '%') {
            p++;
            switch (*p) {
                case 'd': {  // Decimal integer
                    int val = va_arg(args, int);
                    if (val < 0) {
                        kputs_fb(bi, "-");
                        val = -val;
                    }
                    itoa((unsigned int)val, buf, 10);
                    kputs_fb(bi, buf);
                    break;
                }
                case 'x': {  // Hexadecimal integer
                    int val = va_arg(args, int);
                    itoa((unsigned int)val, buf, 16);
                    kputs_fb(bi, buf);
                    break;
                }
                case 's': {  // String
                    char* str = va_arg(args, char*);
                    if (str) kputs_fb(bi, str);
                    else kputs_fb(bi, "(null)");
                    break;
                }
                case 'c': {  // Single character
                    char ch = (char)va_arg(args, int);
                    char tmp[2] = { ch, 0 };
                    kputs_fb(bi, tmp);
                    break;
                }
                case '%': {  // Literal %
                    kputs_fb(bi, "%");
                    break;
                }
                default: {   // Unknown format specifier
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
