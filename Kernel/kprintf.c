#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#include "fb.h"      // kputs_fb(BootInfo*, const char*)


static inline void kputc(BootInfo* bi, char c) {
    char tmp[2] = { c, 0 };
    kputs_fb(bi, tmp);
}

static void kputs_raw(BootInfo* bi, const char* s) {
    if (!s) return;
    while (*s) { kputc(bi, *s++); }
}

static int parse_width(const char** p, char* pad_char) {
    // 지원: '0' 패딩 또는 공백 패딩 + 숫자 폭 (최대 32)
    const char* s = *p;
    int width = 0;
    if (*s == '0') { *pad_char = '0'; s++; }
    while (*s >= '0' && *s <= '9') {
        width = width * 10 + (*s - '0');
        s++;
    }
    *p = s;
    if (width > 32) width = 32;
    return width;
}

static void utoa_base_u64(uint64_t v, unsigned base, bool upper,
                          char* buf_rev, int* len_out) {
    static const char* L = "0123456789abcdef";
    static const char* U = "0123456789ABCDEF";
    const char* DIG = upper ? U : L;

    int i = 0;
    if (v == 0) {
        buf_rev[i++] = '0';
    } else {
        while (v) {
            buf_rev[i++] = DIG[v % base];
            v /= base;
        }
    }
    *len_out = i; // 역순 길이
}

// width/pad 적용해서 출력
static void print_uint(BootInfo* bi, uint64_t v, unsigned base,
                       int width, char pad, bool upper) {
    char rev[64];
    int n = 0;
    utoa_base_u64(v, base, upper, rev, &n);

    int pad_needed = width - n;
    while (pad_needed-- > 0) kputc(bi, pad);

    while (n--) kputc(bi, rev[n]);
}

static void print_int(BootInfo* bi, int64_t val, int width, char pad) {
    if (val < 0) {
        kputc(bi, '-');
        // 음수일 때 패딩은 부호 뒤에 적용
        if (width > 0) width--;
        print_uint(bi, (uint64_t)(-val), 10, width, pad, false);
    } else {
        print_uint(bi, (uint64_t)val, 10, width, pad, false);
    }
}

void kprintf(BootInfo* bi, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);

    for (const char* p = fmt; *p; ++p) {
        if (*p != '%') { kputc(bi, *p); continue; }

        // '%' 처리
        ++p;

        // 길이 수식자: h(무시), l, ll 지원
        enum { LEN_DEF, LEN_L, LEN_LL } len = LEN_DEF;
        char pad = ' ';
        int width = 0;

        // 패딩/폭 먼저 파싱
        width = parse_width(&p, &pad);

        // 길이 수식자 파싱
        if (*p == 'l') {
            ++p;
            if (*p == 'l') { len = LEN_LL; ++p; }
            else { len = LEN_L; }
        } else if (*p == 'h') {
            // h/ hh는 간단히 무시
            ++p;
            if (*p == 'h') ++p;
        }

        switch (*p) {
            case 'd': { // signed
                if (len == LEN_LL)      print_int(bi, va_arg(ap, long long), width, pad);
                else if (len == LEN_L)  print_int(bi, va_arg(ap, long),      width, pad);
                else                    print_int(bi, va_arg(ap, int),        width, pad);
                break;
            }
            case 'u': { // unsigned
                if (len == LEN_LL)      print_uint(bi, va_arg(ap, unsigned long long), 10, width, pad, false);
                else if (len == LEN_L)  print_uint(bi, va_arg(ap, unsigned long),      10, width, pad, false);
                else                    print_uint(bi, va_arg(ap, unsigned int),       10, width, pad, false);
                break;
            }
            case 'x': { // hex lower
                if (len == LEN_LL)      print_uint(bi, va_arg(ap, unsigned long long), 16, width, pad, false);
                else if (len == LEN_L)  print_uint(bi, va_arg(ap, unsigned long),      16, width, pad, false);
                else                    print_uint(bi, va_arg(ap, unsigned int),       16, width, pad, false);
                break;
            }
            case 'X': { // hex upper
                if (len == LEN_LL)      print_uint(bi, va_arg(ap, unsigned long long), 16, width, pad, true);
                else if (len == LEN_L)  print_uint(bi, va_arg(ap, unsigned long),      16, width, pad, true);
                else                    print_uint(bi, va_arg(ap, unsigned int),       16, width, pad, true);
                break;
            }
            case 'p': { // pointer as 0x...
                uintptr_t ptr = (uintptr_t)va_arg(ap, void*);
                kputs_raw(bi, "0x");
                // 64비트 환경이면 16자리, 32비트면 8자리 정도로 맞춰도 됨
                print_uint(bi, (uint64_t)ptr, 16,
                           (int)(sizeof(void*)*2), '0', false);
                break;
            }
            case 'c': {
                char ch = (char)va_arg(ap, int);
                kputc(bi, ch);
                break;
            }
            case 's': {
                const char* s = va_arg(ap, const char*);
                kputs_raw(bi, s ? s : "(null)");
                break;
            }
            case '%': {
                kputc(bi, '%');
                break;
            }
            default: {
                // 알 수 없는 포맷: 그대로 노출
                kputc(bi, '%'); kputc(bi, *p);
                break;
            }
        }
    }

    va_end(ap);
}
