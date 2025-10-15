#include <stddef.h>                     // ✅ def size_t 
#include <Base.h>
#include <Library/BaseMemoryLib.h>
#include <Library/BaseLib.h>
#include <Library/MemoryAllocationLib.h>

// substitute basic libc func
void *memset(void *s, int c, size_t n) { SetMem(s, n, (UINT8)c); return s; }
void *memcpy(void *dst, const void *src, size_t n) { CopyMem(dst, src, n); return dst; }
int   memcmp(const void *a, const void *b, size_t n) { return (int)CompareMem(a, b, n); }
size_t strlen(const char *s) { return (size_t)AsciiStrLen(s); }
char *strstr(const char *h, const char *n) { return (char*)AsciiStrStr((CONST CHAR8*)h, (CONST CHAR8*)n); }

// glibc fortified 함수 우회
void *__memset_chk(void *s, int c, size_t n, size_t dstlen) { (void)dstlen; return memset(s, c, n); }

// malloc 계열 대체
void *calloc(size_t m, size_t n) { return AllocateZeroPool(m * n); }
void free(void *p) { if (p) FreePool(p); }
