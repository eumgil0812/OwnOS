#ifndef KPRINTF_H
#define KPRINTF_H

#include "fb.h"  // BootInfo 구조체 사용을 위해 필요

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 커널 printf 함수 (Framebuffer 콘솔 출력)
 *
 * 지원 포맷:
 *   - %d : 10진수 (음수 지원)
 *   - %x : 16진수
 *   - %s : 문자열
 *   - %c : 문자
 *   - %% : '%' 출력
 *
 * @param bi   BootInfo 포인터 (Framebuffer 정보)
 * @param fmt  포맷 문자열
 * @param ...  가변 인자
 */
void kprintf(BootInfo* bi, const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* KPRINTF_H */
