// env/doom's qlibc, plus the two parsers wasm3's m3_CallArgv names (unused by firmware.c).
#pragma once
#include "qlibc.h"
#define strtoul(s, end, base) ((unsigned long)strtol((s), (end), (base)))
#define strtoull(s, end, base) ((unsigned long long)strtol((s), (end), (base)))
