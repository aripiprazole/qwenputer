/* Agent-programmable harness. ALL parsing, word definitions, arithmetic, control
 * flow and memory below execute as RV32 guest instructions in the neural circuit.
 * No libc, host calls, native tool dispatcher, or pretrained language weights.
 */
typedef unsigned int u32;
typedef int i32;

#define TEXT ((const volatile unsigned char *)0x8000)
#define WORDS 16
#define NAME 24
#define BODY 256
#define STACK 64
#define CELLS 128
#define INPUT 768

static struct { char name[NAME]; char body[BODY]; } words[WORDS];
static u32 stack[STACK], cells[CELLS];
static int sp, fuel, failed, output;
static char input[INPUT];

static void syscall(int n, int value) {
    register int a0 __asm__("a0") = value;
    register int a7 __asm__("a7") = n;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}

static void put(int c) { syscall(1, c); }
static void text(const char *s) { while (*s) put((unsigned char)*s++); }
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
    do { digits[n++] = '0' + value % 10; value /= 10; } while (value);
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
    while (*p && (unsigned char)*p <= 32) p++;
    while (*p && (unsigned char)*p > 32) {
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
    do { v = v * 10 + *s++ - '0'; } while (*s >= '0' && *s <= '9');
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
    while (**p && (unsigned char)**p <= 32) (*p)++;
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
        else if (equal(word, "emit")) { a = pop(); if (!failed) emit(a); }
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
static int gather(void) {
    int n = 0, gt = 0;
    for (int row = 1; row <= 80; row++) {
        const volatile unsigned char *p = TEXT + 16 * row;
        if (!p[0]) break;
        for (int k = 0; k < 16 && p[k]; k++) {
            int c = p[k];
            if (gt < 2) gt += c == '>';
            else if (c == '<') { input[n] = 0; return n; }
            else if (n < INPUT - 1) input[n++] = c;
            else return -1;
        }
    }
    input[n] = 0;
    return -1;
}
int main(void) {
    for (;;) {
        syscall(93, 0); /* park; GO from </tool_call> resumes at the next instruction */
        int count = gather();
        put('\n'); put(256); put('\n');
        failed = output = 0; fuel = 256;
        if (count < 0) error("incomplete command; use a shorter call");
        else eval(input, 0);
        if (failed) sp = 0;
        put('\n'); put(257); put('\n'); put(258); put('\n'); put(259); put('\n');
    }
}
