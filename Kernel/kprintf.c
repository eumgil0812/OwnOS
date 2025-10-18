#include <stdarg.h>
#include "fb.h"

// 간단한 정수 → 문자열 변환
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

void kprintf(BootInfo* bi, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    char buf[64];

    for (const char* p = fmt; *p; p++) {
        if (*p == '%') {
            p++;
            switch (*p) {
                case 'd': {
                    int val = va_arg(args, int);
                    if (val < 0) {
                        kputs_fb(bi, "-");
                        val = -val;
                    }
                    itoa((unsigned int)val, buf, 10);
                    kputs_fb(bi, buf);
                    break;
                }
                case 'x': {
                    int val = va_arg(args, int);
                    itoa((unsigned int)val, buf, 16);
                    kputs_fb(bi, buf);
                    break;
                }
                case 's': {
                    char* str = va_arg(args, char*);
                    if (str) kputs_fb(bi, str);
                    else kputs_fb(bi, "(null)");
                    break;
                }
                case 'c': {
                    char ch = (char)va_arg(args, int);
                    char tmp[2] = { ch, 0 };
                    kputs_fb(bi, tmp);
                    break;
                }
                case '%': {
                    kputs_fb(bi, "%");
                    break;
                }
                default:
                    // 알 수 없는 포맷은 그냥 출력
                    kputs_fb(bi, "%");
                    char tmp[2] = { *p, 0 };
                    kputs_fb(bi, tmp);
                    break;
            }
        } else {
            char tmp[2] = { *p, 0 };
            kputs_fb(bi, tmp);
        }
    }

    va_end(args);
}
