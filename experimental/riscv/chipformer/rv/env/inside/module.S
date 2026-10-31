# The chat harness module (harness.wasm), read-only in the DROM; firmware.c hands it to wasm3.
    .section .rodata.q_module, "a"
    .balign 16
    .globl q_module, q_module_end, q_wad, q_wad_end
q_module:
    .incbin MODULE_PATH
q_module_end:
# qlibc.c's read-only files view a WAD; this firmware has none (an empty one).
q_wad:
q_wad_end:
