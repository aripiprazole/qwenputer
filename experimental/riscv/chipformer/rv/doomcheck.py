"""Measure the Doom firmware on the reference machine (a test oracle; no circuit, no model).

  QWENPUTER_RV_MAP=doom python -m chipformer.rv.doomcheck [keys ...]

Runs the firmware to its boot park, starts Doom with one call, then one call per argument
(the keys of a tool call), and reports instructions, tokens and the RAM high-water mark per
call. The reference model computes these numbers; they are not circuit measurements.
"""

import sys
import time
from pathlib import Path

from chipformer.rv import elf, ref
from chipformer.rv import spec as S

FIRMWARE = Path(__file__).parent / "env/doom/doom.elf"


def call_rows(keys: str) -> list[bytes]:
    """The text buffer of a ``doom`` tool call, one byte per row (the bridge stores one Qwen
    token per row; any split gives the firmware the same text)."""
    body = f"<function=doom><parameter=keys>{keys}</parameter></function>".encode()
    rows = [bytes(S.TEXT_ROW)] + [bytes([b]).ljust(S.TEXT_ROW, b"\0") for b in body]
    return (rows + [bytes(S.TEXT_ROW)] * S.TEXT_ROWS)[:S.TEXT_ROWS]


def run(m: ref.Machine, limit: int) -> dict:
    """Records until the next park (HALT); counts without keeping the trace."""
    m.halted, m.out, m.screen = False, bytearray(), bytearray()
    steps0, tokens, t0 = m.steps, 0, time.time()
    while not m.halted:
        tokens += len(m.record())
        if m.steps - steps0 > limit:
            raise RuntimeError(f"no park within {limit} instructions")
    if m.fault is not None:
        raise RuntimeError(f"fault {S.FAULTS[m.fault[0]]} at {m.fault[1]:#x}: {m.out[-400:]}")
    return {"instructions": m.steps - steps0, "tokens": tokens, "seconds": time.time() - t0,
            "out": m.out.decode("latin-1"), "screen": m.screen.decode("latin-1")}


def high_water(m: ref.Machine) -> int:
    """The highest nonzero RAM word below the stack (bytes)."""
    top = S.RAM_WORDS - 16 * 1024 // 4
    for i in range(top - 1, -1, -1):
        if m.ram[i]:
            return 4 * i + 4
    return 0


def main(argv: list[str]) -> None:
    m = ref.Machine.from_program(elf.program(FIRMWARE))
    m.text = call_rows("")
    print("boot", {k: v for k, v in run(m, 10**6).items() if k != "out"})
    for keys in [""] + argv:
        m.text = call_rows(keys)
        r = run(m, 2 * 10**9)
        print(r["screen"].split("\f")[-1])
        print("to Qwen:", repr(r["out"]))
        print(f"call {keys!r}: {r['instructions']:,} instructions, {r['tokens']:,} tokens, "
              f"{r['seconds']:.0f} s on the reference; RAM high-water {high_water(m):,} bytes",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
