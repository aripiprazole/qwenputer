// A freestanding C library for the Doom firmware (see include/qlibc.h). Files are read-only
// views of the WAD in the DROM; there is no writable file system, so writes report failure.
#include "qposix.h"

int errno;

// ---------------------------------------------------------------- memory

extern char __heap_start[], __heap_end[];
static char *brk = __heap_start;

// A bump allocator: Doom takes nearly all of its memory once, for the zone (Z_Init), so a
// block is only reclaimed when it is the newest one.
void *malloc(size_t n) {
    n = (n + 15) & ~(size_t)15;
    if (n + 16 > (size_t)(__heap_end - brk)) {
        errno = ENOMEM;
        return NULL;
    }
    size_t *h = (size_t *)brk;
    h[0] = n;
    brk += 16 + n;
    return h + 4;
}

void free(void *p) {
    if (!p) return;
    size_t *h = (size_t *)p - 4;
    if ((char *)p + h[0] == brk) brk = (char *)h;
}

void *calloc(size_t n, size_t size) {
    void *p = malloc(n * size);
    if (p) memset(p, 0, n * size);
    return p;
}

void *realloc(void *p, size_t n) {
    if (!p) return malloc(n);
    size_t old = ((size_t *)p - 4)[0];
    if (n <= old) return p;
    if ((char *)p + old == brk && (size_t)(__heap_end - (char *)p) >= n) {  // grow in place
        n = (n + 15) & ~(size_t)15;
        ((size_t *)p - 4)[0] = n;
        brk = (char *)p + n;
        return p;
    }
    void *q = malloc(n);
    if (q) memcpy(q, p, old);
    return q;
}

// Bytes up to a word boundary, then words (four per loop), then the tail: a word store is one
// circuit record where a byte store is a read-modify-write of four.
void *memcpy(void *d, const void *s, size_t n) {
    unsigned char *a = d;
    const unsigned char *b = s;
    if (!(((uintptr_t)a ^ (uintptr_t)b) & 3)) {
        for (; n && ((uintptr_t)a & 3); n--) *a++ = *b++;
        uint32_t *wa = (uint32_t *)a;
        const uint32_t *wb = (const uint32_t *)b;
        for (; n >= 16; n -= 16, wa += 4, wb += 4) {
            wa[0] = wb[0]; wa[1] = wb[1]; wa[2] = wb[2]; wa[3] = wb[3];
        }
        for (; n >= 4; n -= 4) *wa++ = *wb++;
        a = (unsigned char *)wa;
        b = (const unsigned char *)wb;
    }
    while (n--) *a++ = *b++;
    return d;
}

void *memmove(void *d, const void *s, size_t n) {
    unsigned char *a = d;
    const unsigned char *b = s;
    if (a <= b || a >= b + n) return memcpy(d, s, n);
    while (n--) a[n] = b[n];
    return d;
}

void *memset(void *d, int c, size_t n) {
    unsigned char *a = d;
    for (; n && ((uintptr_t)a & 3); n--) *a++ = (unsigned char)c;
    uint32_t w = (unsigned char)c * 0x01010101u, *wa = (uint32_t *)a;
    for (; n >= 16; n -= 16, wa += 4) wa[0] = wa[1] = wa[2] = wa[3] = w;
    for (; n >= 4; n -= 4) *wa++ = w;
    a = (unsigned char *)wa;
    while (n--) *a++ = (unsigned char)c;
    return d;
}

int memcmp(const void *x, const void *y, size_t n) {
    const unsigned char *a = x, *b = y;
    for (; n; n--, a++, b++)
        if (*a != *b) return *a - *b;
    return 0;
}

// ---------------------------------------------------------------- strings

size_t strlen(const char *s) {
    const char *p = s;
    while (*p) p++;
    return p - s;
}
char *strcpy(char *d, const char *s) {
    char *r = d;
    while ((*d++ = *s++)) {}
    return r;
}
char *strncpy(char *d, const char *s, size_t n) {
    size_t i = 0;
    for (; i < n && s[i]; i++) d[i] = s[i];
    for (; i < n; i++) d[i] = 0;
    return d;
}
char *strcat(char *d, const char *s) {
    strcpy(d + strlen(d), s);
    return d;
}
char *strncat(char *d, const char *s, size_t n) {
    char *e = d + strlen(d);
    while (n-- && *s) *e++ = *s++;
    *e = 0;
    return d;
}
int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) a++, b++;
    return (unsigned char)*a - (unsigned char)*b;
}
int strncmp(const char *a, const char *b, size_t n) {
    for (; n; n--, a++, b++) {
        if (*a != *b) return (unsigned char)*a - (unsigned char)*b;
        if (!*a) return 0;
    }
    return 0;
}
int strcasecmp(const char *a, const char *b) {
    while (*a && tolower(*a) == tolower(*b)) a++, b++;
    return tolower((unsigned char)*a) - tolower((unsigned char)*b);
}
int strncasecmp(const char *a, const char *b, size_t n) {
    for (; n; n--, a++, b++) {
        int x = tolower((unsigned char)*a), y = tolower((unsigned char)*b);
        if (x != y) return x - y;
        if (!x) return 0;
    }
    return 0;
}
char *strchr(const char *s, int c) {
    for (;; s++) {
        if (*s == (char)c) return (char *)s;
        if (!*s) return NULL;
    }
}
char *strrchr(const char *s, int c) {
    const char *r = NULL;
    for (;; s++) {
        if (*s == (char)c) r = s;
        if (!*s) return (char *)r;
    }
}
char *strstr(const char *h, const char *n) {
    size_t k = strlen(n);
    for (; *h; h++)
        if (!strncmp(h, n, k)) return (char *)h;
    return k ? NULL : (char *)h;
}
char *strdup(const char *s) {
    char *d = malloc(strlen(s) + 1);
    return d ? strcpy(d, s) : NULL;
}

int toupper(int c) { return c >= 'a' && c <= 'z' ? c - 32 : c; }
int tolower(int c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
int isspace(int c) { return c == ' ' || (c >= 9 && c <= 13); }
int isdigit(int c) { return c >= '0' && c <= '9'; }
int isupper(int c) { return c >= 'A' && c <= 'Z'; }
int islower(int c) { return c >= 'a' && c <= 'z'; }
int isalpha(int c) { return isupper(c) || islower(c); }
int isalnum(int c) { return isalpha(c) || isdigit(c); }
int isprint(int c) { return c >= 32 && c < 127; }
int isxdigit(int c) { return isdigit(c) || (tolower(c) >= 'a' && tolower(c) <= 'f'); }

long strtol(const char *s, char **end, int base) {
    while (isspace(*s)) s++;
    int neg = *s == '-';
    if (*s == '-' || *s == '+') s++;
    if ((base == 0 || base == 16) && s[0] == '0' && tolower(s[1]) == 'x') s += 2, base = 16;
    else if (base == 0) base = *s == '0' ? 8 : 10;
    long v = 0;
    for (;; s++) {
        int d = isdigit(*s) ? *s - '0' : isalpha(*s) ? tolower(*s) - 'a' + 10 : 99;
        if (d >= base) break;
        v = v * base + d;
    }
    if (end) *end = (char *)s;
    return neg ? -v : v;
}
int atoi(const char *s) { return (int)strtol(s, NULL, 10); }
long atol(const char *s) { return strtol(s, NULL, 10); }
int abs(int v) { return v < 0 ? -v : v; }
long labs(long v) { return v < 0 ? -v : v; }
char *getenv(const char *name) { (void)name; return NULL; }
int system(const char *cmd) { (void)cmd; return -1; }

// ---------------------------------------------------------------- 64-bit division

unsigned q_divlu(unsigned u1, unsigned u0, unsigned v) {
    const unsigned b = 1u << 16;
    int s = __builtin_clz(v);
    v <<= s;
    unsigned vn1 = v >> 16, vn0 = v & 0xFFFF;
    unsigned un32 = s ? u1 << s | u0 >> (32 - s) : u1, un10 = u0 << s;
    unsigned un1 = un10 >> 16, un0 = un10 & 0xFFFF;
    unsigned q1 = un32 / vn1, rhat = un32 - q1 * vn1;
    while (q1 >= b || q1 * vn0 > b * rhat + un1) {
        q1--;
        rhat += vn1;
        if (rhat >= b) break;
    }
    unsigned un21 = un32 * b + un1 - q1 * v;
    unsigned q0 = un21 / vn1;
    rhat = un21 - q0 * vn1;
    while (q0 >= b || q0 * vn0 > b * rhat + un0) {
        q0--;
        rhat += vn1;
        if (rhat >= b) break;
    }
    return q1 * b + q0;
}

// libgcc's 64-bit division, for any other long long quotient: shift and subtract.
unsigned long long __udivmoddi4(unsigned long long n, unsigned long long d, unsigned long long *r) {
    unsigned long long q = 0, rem = 0;
    for (int i = 63; i >= 0; i--) {
        rem = rem << 1 | (n >> i & 1);
        if (rem >= d) { rem -= d; q |= 1ull << i; }
    }
    if (r) *r = rem;
    return q;
}
unsigned long long __udivdi3(unsigned long long n, unsigned long long d) { return __udivmoddi4(n, d, 0); }
unsigned long long __umoddi3(unsigned long long n, unsigned long long d) {
    unsigned long long r;
    __udivmoddi4(n, d, &r);
    return r;
}
long long __divdi3(long long a, long long b) {
    unsigned long long q = __udivdi3(a < 0 ? -(unsigned long long)a : a, b < 0 ? -(unsigned long long)b : b);
    return (a < 0) != (b < 0) ? -(long long)q : (long long)q;
}
long long __moddi3(long long a, long long b) {
    unsigned long long r = __umoddi3(a < 0 ? -(unsigned long long)a : a, b < 0 ? -(unsigned long long)b : b);
    return a < 0 ? -(long long)r : (long long)r;
}

// ---------------------------------------------------------------- formatted output

struct sink { char *p; size_t room; int n; FILE *f; };
struct qfile { int fd; };
static struct qfile files[3] = {{0}, {1}, {2}};
FILE *const stdin = &files[0], *const stdout = &files[1], *const stderr = &files[2];

static void out(struct sink *s, int c) {
    if (s->f) q_show(c);
    else if (s->room > 1) { *s->p++ = (char)c; s->room--; }
    s->n++;
}

int vsink(struct sink *s, const char *fmt, va_list ap) {
    for (; *fmt; fmt++) {
        if (*fmt != '%') { out(s, *fmt); continue; }
        int left = 0, zero = 0, width = 0, prec = -1;
        for (;; fmt++) {
            if (fmt[1] == '-') left = 1;
            else if (fmt[1] == '0') zero = 1;
            else if (fmt[1] != '+' && fmt[1] != ' ' && fmt[1] != '#') break;
        }
        fmt++;
        if (*fmt == '*') { width = va_arg(ap, int); fmt++; }
        else while (isdigit(*fmt)) width = width * 10 + *fmt++ - '0';
        if (*fmt == '.') {
            fmt++;
            prec = 0;
            if (*fmt == '*') { prec = va_arg(ap, int); fmt++; }
            else while (isdigit(*fmt)) prec = prec * 10 + *fmt++ - '0';
        }
        while (*fmt == 'l' || *fmt == 'h' || *fmt == 'z') fmt++;
        char buf[16], *str = buf;
        int len = 0, neg = 0;
        switch (*fmt) {
        case 'd': case 'i': case 'u': case 'x': case 'X': case 'p': {
            unsigned v = va_arg(ap, unsigned);
            unsigned base = *fmt == 'x' || *fmt == 'X' || *fmt == 'p' ? 16 : 10;
            if ((*fmt == 'd' || *fmt == 'i') && (int)v < 0) { neg = 1; v = 0u - v; }
            const char *digits = *fmt == 'X' ? "0123456789ABCDEF" : "0123456789abcdef";
            char *e = buf + sizeof buf;
            do { *--e = digits[v % base]; v /= base; } while (v);
            str = e;
            len = buf + sizeof buf - e;
            if (prec > len) {  // an integer's precision is its minimum digit count
                zero = 1;
                width = prec + neg > width ? prec + neg : width;
            }
            prec = -1;
            break;
        }
        case 'c': buf[0] = (char)va_arg(ap, int); len = 1; break;
        case 's':
            str = va_arg(ap, char *);
            if (!str) str = "(null)";
            len = strlen(str);
            if (prec >= 0 && prec < len) len = prec;
            zero = 0;
            break;
        case '%': buf[0] = '%'; len = 1; break;
        default: buf[0] = '?'; len = 1; break;
        }
        int pad = width - len - neg;
        if (!left && !zero) while (pad-- > 0) out(s, ' ');
        if (neg) out(s, '-');
        if (!left && zero) while (pad-- > 0) out(s, '0');
        for (int i = 0; i < len; i++) out(s, str[i]);
        if (left) while (pad-- > 0) out(s, ' ');
    }
    if (!s->f && s->room) *s->p = 0;
    return s->n;
}

int vsnprintf(char *buf, size_t n, const char *fmt, va_list ap) {
    struct sink s = {buf, n, 0, NULL};
    return vsink(&s, fmt, ap);
}
int vsprintf(char *buf, const char *fmt, va_list ap) { return vsnprintf(buf, (size_t)-1 >> 1, fmt, ap); }
int vfprintf(FILE *f, const char *fmt, va_list ap) {
    struct sink s = {NULL, 0, 0, f};
    return vsink(&s, fmt, ap);
}
int vprintf(const char *fmt, va_list ap) { return vfprintf(stdout, fmt, ap); }
int printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vfprintf(stdout, fmt, ap);
    va_end(ap);
    return n;
}
int fprintf(FILE *f, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vfprintf(f, fmt, ap);
    va_end(ap);
    return n;
}
int sprintf(char *buf, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsprintf(buf, fmt, ap);
    va_end(ap);
    return n;
}
int snprintf(char *buf, size_t size, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, size, fmt, ap);
    va_end(ap);
    return n;
}
int putchar(int c) { q_show(c); return c; }
int fputs(const char *s, FILE *f) { (void)f; while (*s) q_show(*s++); return 0; }
int puts(const char *s) { fputs(s, stdout); q_show('\n'); return 0; }
int fflush(FILE *f) { (void)f; return 0; }
FILE *fopen(const char *path, const char *mode) { (void)path; (void)mode; errno = ENOENT; return NULL; }
int fclose(FILE *f) { (void)f; return 0; }
size_t fwrite(const void *p, size_t size, size_t n, FILE *f) {
    const char *c = p;
    for (size_t i = 0; i < size * n; i++) q_show(c[i]);
    (void)f;
    return n;
}
size_t fread(void *p, size_t size, size_t n, FILE *f) { (void)p; (void)size; (void)n; (void)f; return 0; }
int fscanf(FILE *f, const char *fmt, ...) { (void)f; (void)fmt; return EOF; }
int sscanf(const char *s, const char *fmt, ...) { (void)s; (void)fmt; return 0; }
int feof(FILE *f) { (void)f; return 1; }
int fseek(FILE *f, long off, int whence) { (void)f; (void)off; (void)whence; return -1; }
long ftell(FILE *f) { (void)f; return -1; }
void setbuf(FILE *f, char *buf) { (void)f; (void)buf; }
int getchar(void) { return EOF; }

// ---------------------------------------------------------------- files: the WAD only

#define WAD_NAME "doom1.wad"
static struct { int open; size_t pos; } fds[8];

static int is_wad(const char *path) {
    const char *base = strrchr(path, '/');
    return !strcasecmp(base ? base + 1 : path, WAD_NAME);
}
int open(const char *path, int flags, ...) {
    if (!is_wad(path) || (flags & 3) != O_RDONLY) { errno = ENOENT; return -1; }
    for (int fd = 3; fd < 8; fd++)
        if (!fds[fd].open) { fds[fd].open = 1; fds[fd].pos = 0; return fd; }
    errno = ENOMEM;
    return -1;
}
static int bad(int fd) { return fd < 3 || fd >= 8 || !fds[fd].open; }
int close(int fd) {
    if (bad(fd)) { errno = EBADF; return -1; }
    fds[fd].open = 0;
    return 0;
}
ssize_t read(int fd, void *buf, size_t n) {
    if (bad(fd)) { errno = EBADF; return -1; }
    size_t size = q_wad_end - q_wad;
    if (fds[fd].pos >= size) return 0;
    if (n > size - fds[fd].pos) n = size - fds[fd].pos;
    memcpy(buf, q_wad + fds[fd].pos, n);
    fds[fd].pos += n;
    return n;
}
ssize_t write(int fd, const void *buf, size_t n) {
    if (fd != 1 && fd != 2) { errno = EBADF; return -1; }
    fwrite(buf, 1, n, stdout);
    return n;
}
off_t lseek(int fd, off_t off, int whence) {
    if (bad(fd)) { errno = EBADF; return -1; }
    size_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? fds[fd].pos : (size_t)(q_wad_end - q_wad);
    fds[fd].pos = base + off;
    return fds[fd].pos;
}
int fstat(int fd, struct stat *st) {
    if (bad(fd)) { errno = EBADF; return -1; }
    st->st_size = q_wad_end - q_wad;
    return 0;
}
int stat(const char *path, struct stat *st) {
    if (!is_wad(path)) { errno = ENOENT; return -1; }
    st->st_size = q_wad_end - q_wad;
    return 0;
}
int access(const char *path, int mode) {
    if (!is_wad(path) || (mode & W_OK)) { errno = ENOENT; return -1; }
    return 0;
}
int mkdir(const char *path, mode_t mode) { (void)path; (void)mode; return -1; }
int unlink(const char *path) { (void)path; return -1; }
int usleep(unsigned usec) { (void)usec; return 0; }

// ---------------------------------------------------------------- exit

void exit(int code) {
    for (;;) q_ecall(93, code);
}
void abort(void) { exit(134); }
