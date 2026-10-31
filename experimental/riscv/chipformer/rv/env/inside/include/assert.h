// wasm3's asserts compile out in the firmware (NDEBUG-style).
#pragma once
#define assert(x) ((void)0)
