#ifndef KPRINTF_H
#define KPRINTF_H

#include "fb.h"  // Required for BootInfo structure

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Minimal kernel printf function (Framebuffer console output)
 *
 * Supported format specifiers:
 *   - %d : Decimal integer (supports negative values)
 *   - %x : Hexadecimal integer
 *   - %s : String
 *   - %c : Character
 *   - %% : Print literal '%'
 *
 * @param bi   Pointer to BootInfo (framebuffer information)
 * @param fmt  Format string
 * @param ...  Variadic arguments
 */
void kprintf(BootInfo* bi, const char* fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* KPRINTF_H */
