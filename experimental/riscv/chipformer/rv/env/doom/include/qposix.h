// POSIX file calls (qlibc.c), only for the headers that declare them: Doom's door types are
// called open and close.
#pragma once
#include "qlibc.h"

int open(const char *path, int flags, ...);
int close(int fd);
ssize_t read(int fd, void *buf, size_t n);
ssize_t write(int fd, const void *buf, size_t n);
off_t lseek(int fd, off_t off, int whence);
int fstat(int fd, struct stat *st);
int stat(const char *path, struct stat *st);
int access(const char *path, int mode);
int mkdir(const char *path, mode_t mode);
int unlink(const char *path);
int usleep(unsigned usec);
