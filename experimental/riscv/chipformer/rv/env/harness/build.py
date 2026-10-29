"""Build freestanding RV32IM harness firmware with LLVM; no model is loaded."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile


def build():
    root = Path(__file__).resolve().parent
    clang = os.environ.get("RISCV_CLANG") or shutil.which("clang")
    # The unwrapped Nix LLVM supports the RISC-V target on this development Mac.
    if Path('/nix/store').exists() and 'RISCV_CLANG' not in os.environ:
        candidates = sorted(p for p in Path('/nix/store').glob('*-clang-*/bin/clang')
                            if 'wrapper' not in str(p))
        if candidates:
            clang = str(candidates[-1])
    linker = os.environ.get("RISCV_LLD") or shutil.which("ld.lld")
    if not linker and shutil.which("rustc"):
        sysroot = Path(subprocess.check_output(['rustc', '--print', 'sysroot'], text=True).strip())
        candidates = list(sysroot.glob('lib/rustlib/*/bin/rust-lld'))
        if candidates:
            linker = str(candidates[0])
    if not clang or not linker:
        raise SystemExit('Set RISCV_CLANG and RISCV_LLD to LLVM clang and an ELF-capable lld.')
    with tempfile.TemporaryDirectory(prefix='qwen-harness-build-') as temporary:
        objects = []
        for name in ['start.S', 'harness.c']:
            obj = Path(temporary) / (name + '.o')
            subprocess.run([clang, '--target=riscv32-unknown-none-elf', '-march=rv32im',
                            '-mabi=ilp32', '-msmall-data-limit=0', '-mno-relax', '-Os',
                            '-ffreestanding', '-fno-builtin', '-fno-stack-protector',
                            '-c', str(root / name), '-o', str(obj)], check=True)
            objects.append(str(obj))
        command = [linker]
        if Path(linker).name == 'rust-lld':
            command += ['-flavor', 'gnu']
        subprocess.run([*command, '-m', 'elf32lriscv', '--no-relax', '-T',
                        str(root.parent / 'rom.ld'), *objects, '-o', str(root / 'harness.elf')],
                       check=True)
    print(root / 'harness.elf')


if __name__ == '__main__':
    build()
