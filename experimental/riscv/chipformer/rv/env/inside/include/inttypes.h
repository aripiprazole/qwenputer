// The format macros wasm3 uses (ilp32: long long is 64-bit), for qlibc's printf family.
#pragma once
#include <stdint.h>
#define PRId32 "d"
#define PRIi32 "d"
#define PRIu32 "u"
#define PRIx32 "x"
#define PRId64 "lld"
#define PRIi64 "lld"
#define PRIu64 "llu"
#define PRIx64 "llx"
