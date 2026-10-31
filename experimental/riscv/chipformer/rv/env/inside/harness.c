/* The chat harness as a WebAssembly module (wasm32, no libc). wasm3 interprets it inside the
 * RV32 guest firmware (firmware.c), which runs in Qwen's circuit layers. This module owns the
 * whole chat: the chat template it writes into Qwen, the prompt and line editor on the user's
 * terminal, the streamed reply, the tool-call capture, the Forth-like guest stack language
 * (ported from env/harness/harness.c) and the turn control. It contains no language model;
 * every reply token comes from Qwen through model_yield.
 *
 * Imports (module "env", implemented by firmware.c as the machine's ecalls):
 *   model_put(c)        one token into Qwen's context: a byte, or 256 + k a special token
 *   model_write(p, n)   n bytes into Qwen's context
 *   screen_write(p, n)  n bytes to the user's terminal only
 *   model_yield(p, e)   Qwen writes one token; its bytes (16, NUL padded) are stored at p;
 *                       returns its length | 32 if a byte is < 32 | 64 if it starts with '<';
 *                       with e set, a token with neither flag is also shown on the terminal
 *   key_wait()          the next key the user pressed
 * Export: run() (never returns).
 */
typedef unsigned int u32;
typedef int i32;
typedef unsigned char u8;

#define IMPORT(name) __attribute__((import_module("env"), import_name(name)))
IMPORT("model_put") void model_put(int c);
IMPORT("model_write") void model_write(const char *p, int n);
IMPORT("screen_write") void screen_write(const char *p, int n);
IMPORT("model_yield") int model_yield(u8 *row, int echo);
IMPORT("key_wait") int key_wait(void);

/* PREFIX (the chat template's system turn) and the turn pieces as Qwen's own tokens, generated
 * by build.py from the Qwen3.5 template and tokenizer: putchar values (see tsend). */
#include "prefix.h"

void *memcpy(void *d, const void *s, unsigned long n) {
    u8 *a = d; const u8 *b = s;
    while (n--) *a++ = *b++;
    return d;
}
void *memset(void *d, int c, unsigned long n) {
    u8 *a = d;
    while (n--) *a++ = (u8)c;
    return d;
}

enum { TR_OPEN, TR_CLOSE, THINK, END_THINK, IM_START, IM_END, TC_OPEN, TC_CLOSE, EOT };
static const char *const SPECIAL[] = {"<tool_response>", "</tool_response>", "<think>",
    "</think>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<|endoftext|>"};
#define TURN_TOKENS 768

static int length(const char *s) { int n = 0; while (s[n]) n++; return n; }
static void show(const char *s) { screen_write(s, length(s)); }
static void showc(int c) { char b = (char)c; screen_write(&b, 1); }

/* Pre-tokenized text into Qwen, one token each: 0..255 a byte's token, 256 + k a special
 * token, 264 + j a token of the table the bridge appends to the printables (tokens.json). Read
 * one byte per token, a long system turn degrades Qwen's tool calls. */
static void tsend(const unsigned short *t, int n) {
    for (int i = 0; i < n; i++) model_put(t[i]);
}
#define TSEND(name) tsend(name, name##_N)

/* ------------------------------------------------------------ the guest stack language */
/* A port of env/harness/harness.c; its output goes to `out` instead of putchar. */

#define WORDS 16
#define NAME 24
#define BODY 256
#define STACK 64
#define CELLS 128
#define INPUT 768
#define OUT 512

static struct { char name[NAME]; char body[BODY]; } words[WORDS];
static u32 stack[STACK], cells[CELLS];
static int sp, fuel, failed, output, outn;
static char input[INPUT], out[OUT];

static void put(int c) { if (outn < OUT - 1) out[outn++] = (char)c; }
static void text(const char *s) { while (*s) put((u8)*s++); }
static int equal(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return *a == *b;
}
static void copy(char *to, const char *from) { while ((*to++ = *from++)) {} }
static void error(const char *s) {
    if (!failed) { text("error: "); text(s); put('\n'); }
    failed = 1;
}
static void emit(int c) {
    if (output++ >= 160) { error("output limit"); return; }
    put(c & 255);
}
static void number(u32 value) {
    char digits[10];
    int n = 0;
    if ((i32)value < 0) { emit('-'); value = 0u - value; }
    do { digits[n++] = (char)('0' + value % 10); value /= 10; } while (value);
    while (n) emit(digits[--n]);
    emit(' ');
}
static void push(u32 value) {
    if (sp == STACK) error("stack full"); else stack[sp++] = value;
}
static u32 pop(void) {
    if (!sp) { error("stack empty"); return 0; }
    return stack[--sp];
}
static int next(const char **position, char *word) {
    const char *p = *position;
    int n = 0;
    while (*p && (u8)*p <= 32) p++;
    while (*p && (u8)*p > 32) {
        if (n == NAME - 1) { error("token too long"); return 0; }
        word[n++] = *p++;
    }
    word[n] = 0;
    *position = p;
    return n;
}
static int find(const char *name) {
    for (int i = 0; i < WORDS; i++) if (equal(name, words[i].name)) return i;
    return -1;
}
static int integer(const char *s, u32 *value) {
    int minus = *s == '-';
    if (minus) s++;
    if (*s < '0' || *s > '9') return 0;
    u32 v = 0;
    do { v = v * 10 + (u32)(*s++ - '0'); } while (*s >= '0' && *s <= '9');
    if (*s) return 0;
    *value = minus ? 0u - v : v;
    return 1;
}
static void skip(const char **p, int allow_else) {
    char word[NAME];
    int depth = 0;
    while (next(p, word)) {
        if (equal(word, "if")) depth++;
        else if (equal(word, "then")) { if (!depth) return; depth--; }
        else if (allow_else && !depth && equal(word, "else")) return;
    }
    error("missing then");
}
static void define(const char **p) {
    char name[NAME], body[BODY];
    if (!next(p, name)) { error("missing word name"); return; }
    int at = find(name), n = 0;
    if (at < 0) for (int i = 0; i < WORDS; i++) if (!words[i].name[0]) { at = i; break; }
    if (at < 0) { error("dictionary full"); return; }
    while (**p && (u8)**p <= 32) (*p)++;
    while (**p && **p != ';') {
        if (n == BODY - 1) { error("definition too long"); return; }
        body[n++] = *(*p)++;
    }
    if (**p != ';') { error("missing semicolon"); return; }
    (*p)++;
    body[n] = 0;
    copy(words[at].name, name);
    copy(words[at].body, body);
    text("defined "); text(name); put('\n');
}
static u32 divide(u32 a, u32 b, int remainder) {
    if (!b) { error("division by zero"); return 0; }
    u32 aa = (i32)a < 0 ? 0u - a : a, bb = (i32)b < 0 ? 0u - b : b;
    if (remainder) { u32 r = aa % bb; return (i32)a < 0 ? 0u - r : r; }
    u32 q = aa / bb;
    return ((a ^ b) >> 31) ? 0u - q : q;
}
static void eval(const char *source, int depth) {
    const char *p = source;
    char word[NAME];
    u32 a, b, value;
    if (depth > 12) { error("call depth"); return; }
    while (!failed && next(&p, word)) {
        if (!fuel--) { error("instruction budget"); return; }
        if (integer(word, &value)) push(value);
        else if (equal(word, ":")) define(&p);
        else if (equal(word, "dup")) { a = pop(); push(a); push(a); }
        else if (equal(word, "drop")) (void)pop();
        else if (equal(word, "swap")) { b = pop(); a = pop(); push(b); push(a); }
        else if (equal(word, "over")) { b = pop(); a = pop(); push(a); push(b); push(a); }
        else if (equal(word, ".")) { a = pop(); if (!failed) number(a); }
        else if (equal(word, "emit")) { a = pop(); if (!failed) emit((int)a); }
        else if (equal(word, "cr")) emit('\n');
        else if (equal(word, "clear")) sp = 0;
        else if (equal(word, ".s")) { for (int i = 0; i < sp; i++) number(stack[i]); }
        else if (equal(word, "words")) {
            for (int i = 0; i < WORDS; i++) if (words[i].name[0]) { text(words[i].name); put(' '); }
            put('\n');
        } else if (equal(word, "forget")) {
            if (!next(&p, word)) error("missing word name");
            else { int at = find(word); if (at < 0) error("unknown word"); else words[at].name[0] = 0; }
        } else if (equal(word, "if")) { a = pop(); if (!a && !failed) skip(&p, 1); }
        else if (equal(word, "else")) skip(&p, 0);
        else if (equal(word, "then")) {}
        else if (equal(word, "@")) {
            a = pop(); if (a >= CELLS) error("cell out of range"); else push(cells[a]);
        } else if (equal(word, "!")) {
            a = pop(); b = pop(); if (a >= CELLS) error("cell out of range"); else cells[a] = b;
        } else if (equal(word, "help")) {
            text(": name body ; | words forget | + - * / % lt gt eq | dup drop swap over | . emit cr .s clear | if else then | @ !\n");
        } else {
            int at = find(word);
            if (at >= 0) eval(words[at].body, depth + 1);
            else {
                b = pop(); a = pop();
                if (failed) return;
                if (equal(word, "+")) push(a + b);
                else if (equal(word, "-")) push(a - b);
                else if (equal(word, "*")) push(a * b);
                else if (equal(word, "/")) push(divide(a, b, 0));
                else if (equal(word, "%")) push(divide(a, b, 1));
                else if (equal(word, "lt")) push((i32)a < (i32)b);
                else if (equal(word, "gt")) push((i32)a > (i32)b);
                else if (equal(word, "eq")) push(a == b);
                else if (equal(word, "and")) push(a & b);
                else if (equal(word, "or")) push(a | b);
                else if (equal(word, "xor")) push(a ^ b);
                else if (equal(word, "shl")) push(a << (b & 31));
                else if (equal(word, "shr")) push(a >> (b & 31));
                else error("unknown word");
            }
        }
    }
}

/* ------------------------------------------------------------------- the tool call */

#define CALL 1024
static char call[CALL + 1];
static int calln;

static char *search(char *h, const char *n) {
    for (; *h; h++) {
        int i = 0;
        while (n[i] && h[i] == n[i]) i++;
        if (!n[i]) return h;
    }
    return 0;
}
static int space(int c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

/* A JSON-style call ({"name": "harness", "arguments": {"command": "CMD"}}): the string value
 * of "command", unescaped in place. */
static char *json_command(void) {
    char *p = search(call, "\"command\"");
    if (!p) return 0;
    p += 9;
    while (*p && (space(*p) || *p == ':')) p++;
    if (*p != '"') return 0;
    char *out = ++p, *start = p;
    for (; *p && *p != '"'; p++) {
        if (*p == '\\' && p[1]) {
            p++;
            *out++ = *p == 'n' ? '\n' : *p == 't' ? '\t' : *p;
        } else {
            *out++ = *p;
        }
    }
    *out = 0;
    return start;
}

/* The command parameter of <function=harness><parameter=command>\nCMD\n</parameter>. */
static char *command(void) {
    call[calln] = 0;
    char *p = search(call, "<parameter=");
    if (!p) return json_command();
    while (*p && *p != '>') p++;
    if (!*p) return 0;
    p++;
    char *end = search(p, "</parameter>");
    if (!end) end = p + length(p);
    while (p < end && space(*p)) p++;
    while (end > p && space(end[-1])) end--;
    *end = 0;
    return p;
}

/* The compact tool record: "┌ harness · CMD", "│ line"..., "└ neural guest". */
static void record(const char *cmd, const char *result, int after_text) {
    char line[136];
    int n = 0;
    for (const char *c = cmd; *c && n < 120; c++) line[n++] = *c == '\n' || *c == '\r' ? ' ' : *c;
    show(after_text ? "\r\n\xe2\x94\x8c harness \xc2\xb7 " : "\xe2\x94\x8c harness \xc2\xb7 ");
    screen_write(line, n);
    show("\r\n");
    while (*result) {
        const char *e = result;
        while (*e && *e != '\n') e++;
        show("\xe2\x94\x82 ");
        screen_write(result, (int)(e - result));
        show("\r\n");
        result = *e ? e + 1 : e;
    }
    show("\xe2\x94\x94 neural guest\r\n");
}

static void tool(int after_text) {
    char *cmd = command();
    outn = 0;
    failed = output = 0;
    fuel = 256;
    if (!cmd) error("no command parameter");
    else if (length(cmd) >= INPUT) error("command too long; use a shorter call");
    else { copy(input, cmd); eval(input, 0); }
    if (failed) sp = 0;
    while (outn && space(out[outn - 1])) outn--;  /* the template trims tool content */
    if (!outn) text("ok");
    out[outn] = 0;
    record(cmd ? cmd : "requested", out, after_text);
    TSEND(TOOL_OPEN);
    model_write(out, outn);
    TSEND(TOOL_CLOSE);
}

/* ------------------------------------------------------------------------- the turn */

static u8 row[16] __attribute__((aligned(16)));  /* the token Qwen wrote (model_yield) */
static u32 rows[CALL / 16][4];                    /* the tool call's tokens, a row each */
static int nrows;
enum { TEXT, IN_CALL, IN_THINK };
static int mode, shown, newlines;

static int special(void) {
    for (int k = 0; k <= EOT; k++) {
        const char *s = SPECIAL[k];
        int i = 0;
        while (i < 16 && s[i] && row[i] == (u8)s[i]) i++;
        if (!s[i] && (i == 16 || !row[i])) return k;
    }
    return -1;
}

static void capture(void) {
    if (nrows < CALL / 16) {
        const u32 *w = (const u32 *)row;
        u32 *r = rows[nrows++];
        r[0] = w[0]; r[1] = w[1]; r[2] = w[2]; r[3] = w[3];
    }
}

static void gather(void) {
    calln = 0;
    for (int i = 0; i < nrows; i++) {
        const u8 *b = (const u8 *)rows[i];
        for (int j = 0; j < 16 && b[j]; j++) call[calln++] = (char)b[j];
    }
}

/* The tokens off the fast path (a special token, a control byte, a tool call's text): 1 when
 * the turn is over. */
static __attribute__((noinline)) int token(int info) {
    int k = info & 64 ? special() : -1;
    if (k == IM_END || k == EOT) {
        show(shown ? "\r\n\r\n" : "\r\n");
        if (k == IM_END) model_put('\n');
        return 1;
    }
    if (mode == IN_CALL) {
        if (k == TC_CLOSE) { gather(); tool(shown); mode = TEXT; shown = newlines = 0; }
        else capture();
        return 0;
    }
    if (mode == IN_THINK) { if (k == END_THINK) mode = TEXT; return 0; }
    if (k == TC_OPEN) { mode = IN_CALL; nrows = 0; return 0; }
    if (k == THINK) { mode = IN_THINK; return 0; }
    if (k >= 0) return 0;  /* other special tokens are not shown */
    /* stream the token; newlines wait until more text follows (none before a tool record or
     * at the end of the reply, and none at its start) */
    char buf[48];
    int m = 0;
    for (int i = 0; i < 16 && row[i]; i++) {
        if (row[i] == '\n') { newlines += shown; continue; }
        if (row[i] == '\r') continue;
        for (; newlines; newlines--) { buf[m++] = '\r'; buf[m++] = '\n'; }
        buf[m++] = (char)row[i];
        shown = 1;
    }
    if (m) screen_write(buf, m);
    return 0;
}

static void turn(const char *line, int n) {
    mode = TEXT;
    shown = newlines = 0;
    TSEND(USER_OPEN);
    model_write(line, n);
    TSEND(TO_ASSISTANT);
    for (int left = TURN_TOKENS; left; left--) {
        /* info: the token's length | 32 if it has a control byte | 64 if it starts with '<' */
        int echo = mode == TEXT && !newlines;  /* plain text streams as it arrives */
        int info = model_yield(row, echo);
        if (!(info & 96) && !newlines) {  /* the fast path: plain text */
            if (mode == TEXT) {  /* already shown by model_yield */
                shown |= info != 0;
                continue;
            }
            if (mode == IN_CALL) { capture(); continue; }
        }
        if (token(info)) return;
    }
    show("\r\n\x1b[2m[turn stopped at 768 Qwen tokens]\x1b[0m\r\n");
    TSEND(TURN_END);
}

/* ------------------------------------------------------------------- the line editor */

#define LINE 512
static char line[LINE];

static void prompt(void) { show("\x1b[36m\xe2\x9d\xaf\x1b[0m "); }

/* A line from the keyboard: its length, or -1 for ctrl-d on an empty line. */
static int read_line(void) {
    int n = 0, escape = 0;
    prompt();
    for (;;) {
        int c = key_wait() & 255;
        if (escape) {  /* skip ESC [ ... final (arrows and the like) */
            if (escape == 1 && (c == '[' || c == 'O')) escape = 2;
            else if (escape == 1 || (c >= 0x40 && c <= 0x7e)) escape = 0;
            continue;
        }
        if (c == 27) escape = 1;
        else if (c == 13 || c == 10) { show("\r\n"); line[n] = 0; return n; }
        else if (c == 127 || c == 8) {
            if (n) {
                do n--; while (n && ((u8)line[n] & 0xC0) == 0x80);
                show("\b \b");
            }
        } else if (c == 3) { show("^C\r\n"); n = 0; prompt(); }
        else if (c == 21) { while (n) { n--; if (((u8)line[n] & 0xC0) != 0x80) show("\b \b"); } }
        else if (c == 4) { if (!n) { show("\r\n"); return -1; } }
        else if ((c >= 32 || c == 9) && n < LINE - 1) { line[n++] = (char)c; showc(c); }
    }
}

__attribute__((export_name("run"))) void run(void) {
    TSEND(PREFIX);
    show("\x1b[1mQwenputer\x1b[0m \x1b[2m\xc2\xb7 chat harness in wasm3 \xc2\xb7 rv32 guest inside "
         "Qwen's layers \xc2\xb7 ^D quits\x1b[0m\r\n");
    for (;;) {
        int n = read_line();
        if (n < 0) {
            show("\x1b[2mbye\x1b[0m\r\n");
            for (;;) key_wait();  /* idle: the firmware never exits */
        }
        int i = 0;
        while (i < n && space(line[i])) i++;
        if (i == n) continue;
        if (equal(line + i, "/new")) {
            show("\x1b[2m/new is not supported yet: the conversation and dictionary persist\x1b[0m\r\n");
            continue;
        }
        turn(line, n);
    }
}
