// The small C library the Doom firmware links against (qlibc.c). Every routine runs as guest
// RV32 instructions; the only system calls are the machine's putchar and park/exit ecalls.
#pragma once
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>

typedef int ssize_t;
typedef int off_t;
typedef unsigned int mode_t;
typedef struct qfile FILE;

#define EOF (-1)
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#define O_RDONLY 0
#define O_WRONLY 1
#define O_RDWR 2
#define O_CREAT 0x100
#define O_TRUNC 0x200
#define O_BINARY 0
#define R_OK 4
#define W_OK 2
#define F_OK 0
#define ENOENT 2
#define EBADF 9
#define ENOMEM 12
#define EINVAL 22

extern int errno;
extern FILE *const stdin, *const stdout, *const stderr;

struct stat { off_t st_size; };

int printf(const char *fmt, ...);
int fprintf(FILE *f, const char *fmt, ...);
int sprintf(char *buf, const char *fmt, ...);
int snprintf(char *buf, size_t n, const char *fmt, ...);
int vprintf(const char *fmt, va_list ap);
int vfprintf(FILE *f, const char *fmt, va_list ap);
int vsprintf(char *buf, const char *fmt, va_list ap);
int vsnprintf(char *buf, size_t n, const char *fmt, va_list ap);
int putchar(int c);
int puts(const char *s);
int fputs(const char *s, FILE *f);
int fflush(FILE *f);
FILE *fopen(const char *path, const char *mode);
int fclose(FILE *f);
size_t fwrite(const void *p, size_t size, size_t n, FILE *f);
size_t fread(void *p, size_t size, size_t n, FILE *f);
int fscanf(FILE *f, const char *fmt, ...);
int sscanf(const char *s, const char *fmt, ...);
int feof(FILE *f);
int fseek(FILE *f, long off, int whence);
long ftell(FILE *f);
void setbuf(FILE *f, char *buf);
int getchar(void);


void *malloc(size_t n);
void *calloc(size_t n, size_t size);
void *realloc(void *p, size_t n);
void free(void *p);
void exit(int code) __attribute__((noreturn));
void abort(void) __attribute__((noreturn));
int atoi(const char *s);
long atol(const char *s);
long strtol(const char *s, char **end, int base);
int abs(int v);
long labs(long v);
char *getenv(const char *name);
int system(const char *cmd);
#define alloca __builtin_alloca

void *memcpy(void *d, const void *s, size_t n);
void *memmove(void *d, const void *s, size_t n);
void *memset(void *d, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);
char *strcpy(char *d, const char *s);
char *strncpy(char *d, const char *s, size_t n);
char *strcat(char *d, const char *s);
char *strncat(char *d, const char *s, size_t n);
int strcmp(const char *a, const char *b);
int strncmp(const char *a, const char *b, size_t n);
int strcasecmp(const char *a, const char *b);
int strncasecmp(const char *a, const char *b, size_t n);
char *strchr(const char *s, int c);
char *strrchr(const char *s, int c);
char *strstr(const char *h, const char *n);
char *strdup(const char *s);

int toupper(int c);
int tolower(int c);
int isspace(int c);
int isdigit(int c);
int isalpha(int c);
int isalnum(int c);
int isupper(int c);
int islower(int c);
int isprint(int c);
int isxdigit(int c);

// The 32-bit quotient of the 64-bit (hi:lo) by v; needs hi < v (Hacker's Delight, divlu).
unsigned q_divlu(unsigned hi, unsigned lo, unsigned v);

// The machine's system calls (spec.py): putchar (a7 = 1), display (a7 = 2) and park / exit
// (a7 = 93). stdout and stderr go to the display; Qwen reads only what q_putc writes.
static inline long q_ecall(long n, long a0) {
    register long r_a0 __asm__("a0") = a0;
    register long r_a7 __asm__("a7") = n;
    __asm__ volatile("ecall" : "+r"(r_a0) : "r"(r_a7) : "memory");
    return r_a0;
}
static inline void q_putc(int c) { q_ecall(1, c); }    // to Qwen: echoed as its text
static inline void q_show(int c) { q_ecall(2, c); }    // to the terminal only (DSP)
static inline void q_park(void) { q_ecall(93, 0); }  // GO resumes at the next instruction

// The WAD, linked into the DROM (wad.S), opened by name as a read-only file.
extern const unsigned char q_wad[], q_wad_end[];
