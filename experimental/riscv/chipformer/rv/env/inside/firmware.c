/* RV32 guest firmware for the "inside" map: wasm3 interpreting the chat harness module
 * (harness.c, compiled to wasm32 and linked into the DROM by module.S). Every instruction here,
 * in wasm3 and in the module runs in Qwen's circuit layers. The only I/O is the machine's
 * ecalls: putchar into Qwen's context (a7 = 1), display to the terminal (2), yield to Qwen (3)
 * and key wait (4). The module's imports are those ecalls; its run() loop never returns, so
 * wasm3 is entered once and a Qwen token costs one import call. */
#include "qlibc.h"
#include "wasm3.h"

#define ROW1 (TEXT_ADDR + 16)  /* text row 1: the token (or key) that woke the guest */
#define WASM_STACK (16 * 1024)

extern const unsigned char q_module[], q_module_end[];

static uint32_t memory_size;  /* the module's linear memory: one page, fixed (no growth) */

/* A module pointer (offset at, n bytes) checked against its memory; _mem is the memory's base. */
#define checked(at, n) \
    ((at) <= memory_size && (uint32_t)(n) <= memory_size - (at) ? (char *)_mem + (at) : 0)

m3ApiRawFunction(model_put) {
    m3ApiGetArg(int32_t, c);
    q_ecall(1, c);
    m3ApiSuccess();
}

m3ApiRawFunction(model_write) {
    m3ApiGetArg(uint32_t, at);
    m3ApiGetArg(int32_t, n);
    const unsigned char *p = (const unsigned char *)checked(at, n);
    if (!p) m3ApiTrap(m3Err_trapOutOfBoundsMemoryAccess);
    while (n-- > 0) q_ecall(1, *p++);
    m3ApiSuccess();
}

m3ApiRawFunction(screen_write) {
    m3ApiGetArg(uint32_t, at);
    m3ApiGetArg(int32_t, n);
    const unsigned char *p = (const unsigned char *)checked(at, n);
    if (!p) m3ApiTrap(m3Err_trapOutOfBoundsMemoryAccess);
    while (n-- > 0) q_ecall(2, *p++);
    m3ApiSuccess();
}

/* A word's bytes below 32 (a control byte; NUL included), as 0x80 flags: exact for the lowest
 * one, the only one used. */
#define LOW(w) (((w) - 0x20202020u) & ~(w) & 0x80808080u)
#define NUL(w) (((w) - 0x01010101u) & ~(w) & 0x80808080u)

/* model_yield(row, echo) -> length | 32 if a byte is < 32 (a control: newline...) | 64 if row[0]
 * is '<' (as a special token's text is). The token Qwen wrote is stored at row (16 bytes, 4-byte
 * aligned). With echo set, a plain token (neither flag) is also shown on the terminal: the
 * module streams plain text with one import call per token. */
m3ApiRawFunction(model_yield) {
    m3ApiReturnType(int32_t);
    m3ApiGetArg(uint32_t, at);
    m3ApiGetArg(int32_t, echo);
    uint32_t *w = (uint32_t *)checked(at, 16);
    if (!w || (at & 3)) m3ApiTrap(m3Err_trapOutOfBoundsMemoryAccess);
    q_ecall(3, 0);  /* returns once Qwen has written one token, its bytes in row 1 */
    const volatile uint32_t *row = (const volatile uint32_t *)ROW1;
    uint32_t w0 = row[0], w1 = row[1], w2 = row[2], w3 = row[3];
    w[0] = w0; w[1] = w1; w[2] = w2; w[3] = w3;
    int n = 0, flags = (w0 & 255) == '<' ? 64 : 0;
    for (int i = 0; i < 4; i++) {
        uint32_t v = w[i], z = NUL(v);
        if (!z) {  /* four bytes of text */
            flags |= LOW(v) ? 32 : 0;
            n += 4;
            continue;
        }
        int k = z & 0x80 ? 0 : z & 0x8000 ? 1 : z & 0x800000 ? 2 : 3;  /* the first NUL */
        if (k) flags |= LOW(v | 0xFFFFFFFFu << 8 * k) ? 32 : 0;
        n += k;
        break;
    }
    if (echo && !flags)
        for (const unsigned char *b = (const unsigned char *)w; n && b < (const unsigned char *)w + n; b++)
            q_ecall(2, *b);
    m3ApiReturn(n | flags);
}

m3ApiRawFunction(key_wait) {
    m3ApiReturnType(int32_t);
    q_ecall(4, 0);  /* returns once the user pressed a key: its byte at row 1 */
    m3ApiReturn(*(const volatile unsigned char *)ROW1);
}

static void fail(const char *what, M3Result r) {
    printf("\r\nwasm3: %s: %s\r\n", what, r ? r : "?");
    for (;;) q_ecall(4, 0);  /* idle; the firmware never exits */
}

int main(void) {
    IM3Environment env = m3_NewEnvironment();
    if (!env) fail("environment", "out of memory");
    IM3Runtime runtime = m3_NewRuntime(env, WASM_STACK, NULL);
    if (!runtime) fail("runtime", "out of memory");
    IM3Module module;
    M3Result r = m3_ParseModule(env, &module, q_module, (uint32_t)(q_module_end - q_module));
    if (r) fail("parse", r);
    if ((r = m3_LoadModule(runtime, module))) fail("load", r);
    memory_size = m3_GetMemorySize(runtime);
    static const struct { const char *name, *sig; M3RawCall call; } imports[] = {
        {"model_put", "v(i)", model_put}, {"model_write", "v(ii)", model_write},
        {"screen_write", "v(ii)", screen_write}, {"model_yield", "i(ii)", model_yield},
        {"key_wait", "i()", key_wait}};
    for (unsigned i = 0; i < sizeof imports / sizeof *imports; i++) {
        r = m3_LinkRawFunction(module, "env", imports[i].name, imports[i].sig, imports[i].call);
        if (r && r != m3Err_functionLookupFailed) fail(imports[i].name, r);
    }
    /* compile every function now, at boot, so no turn pays for wasm3's lazy compilation */
    if ((r = m3_CompileModule(module))) fail("compile", r);
    IM3Function run;
    if ((r = m3_FindFunction(&run, runtime, "run"))) fail("run", r);
    r = m3_CallV(run);
    fail("trap", r);
    return 0;
}
