# The WAD, read-only in the DROM; qlibc opens it as doom1.wad. Lumps are 4-byte aligned (wad.py)
# so Doom can read their fields in place.
    .section .rodata.wad, "a"
    .balign 16
    .globl q_wad, q_wad_end
q_wad:
    .incbin WAD_PATH
q_wad_end:
