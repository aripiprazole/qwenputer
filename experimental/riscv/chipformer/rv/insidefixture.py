"""Bounded offline check of the "inside" circuit: no pretrained Qwen, CPU sparse tensors.

  QWENPUTER_RV_MAP=inside python -m chipformer.rv.insidefixture [--rounds N]

A small guest parks with YLD (ecall a7 = 3); each text token that arrives while it is parked
wakes it (GO) with the token's bytes in text row 1. The guest shows the byte on the display,
writes it and <|im_end|> into the context, and waits for a key (KWT, a7 = 4); the key wakes it
the same way. The compiled circuit (core.build for the inside map, a byte-level text bridge) is
fed the reference machine's stream (the test oracle) and must predict every token the circuit
owns, and own nothing after a YLD / KWT. A correctness fixture, not a speed figure.
"""

import argparse
import time

from chipformer.rv import spec as S

PROGRAM = f"""
.text
    li a7, 3
    ecall               # boot parks here: the first text token wakes the guest
loop:
    li t0, {S.TEXT_ADDR + S.TEXT_ROW}
    lbu a0, 0(t0)       # row 1: the token that woke us
    li a7, 2
    ecall               # display
    li a7, 1
    ecall               # echo into the context
    li a0, 261
    ecall               # <|im_end|>
    li a7, 4
    ecall               # wait for a key
    lbu a0, 0(t0)
    li a7, 2
    ecall
    li a7, 3
    ecall               # back to Qwen
    j loop
"""


class InsideText:
    """Byte-level text for the fixture: ids 0..255 are bytes, then the bridge's specials."""

    def __init__(self):
        from chipformer.rv.bridge import SPECIALS
        self.bytes = [bytes([i]) for i in range(256)] + [s.encode() for s in SPECIALS]
        self.printable = list(range(len(self.bytes)))
        self.TCO = self.bytes.index(b"<tool_call>")
        self.TCC = self.bytes.index(b"</tool_call>")
        self.NL = 10

    def printable_index(self, token):
        return token if 0 <= token < len(self.bytes) else None

    def row_bytes(self, token):
        return self.bytes[token][:S.TEXT_ROW].ljust(S.TEXT_ROW, b"\0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    assert S.INSIDE, "run with QWENPUTER_RV_MAP=inside"

    import torch

    from chipformer.rv import asm, bridge, core, ref
    from chipformer.rv.decode import CodeDecoder
    from chipformer.rv.fixture_tensors import SparseTensorCircuit

    torch.set_num_threads(4)
    text = InsideText()
    t0 = time.time()
    c = core.build(text=text)
    prog = asm.assemble(PROGRAM)
    core.snapshot(c, ref.Machine.from_program(prog))
    ex = SparseTensorCircuit(c)
    offset = len(text.bytes)
    dec = CodeDecoder(offset, text)
    print(f"circuit: {c.stats()} built in {time.time() - t0:.0f} s", flush=True)

    oracle = ref.Machine.from_program(prog)
    bridge.park(oracle)
    sess = bridge.Session(text, oracle)

    def step(st):
        with torch.no_grad():
            h = ex.step_token(st)
            p = int(dec(h[c.keep[:core.NFAM + core.NPAY]]))
        if p < 0:
            return p
        return p - offset if p >= offset else S.TEXT_TOKEN | p

    # boot: GO, then free-run to the first park
    tok, n = S.tok("GO"), 0
    while tok not in (S.tok("YLD"), S.tok("KWT")):
        tok = step(tok)
        n += 1
        assert tok >= 0 and n < 200, ("boot lost", tok, n)
    assert step(tok) < 0, "the circuit owns the position after the boot park"
    print(f"boot: parked after {n} circuit steps", flush=True)

    inputs = [ord(ch) for ch in "hello world"][:2 * args.rounds]
    checked = owned = 0
    t1 = time.time()
    for q in inputs:
        k = len(sess.toks)
        sess.wake(q)
        stream = sess.toks[k:]
        for i, t in enumerate(stream):
            pred = step(t)
            checked += 1
            if i + 1 < len(stream):
                if pred != stream[i + 1]:
                    raise SystemExit(f"FAIL after {S.name(t) if t < S.VOCAB else t}: circuit "
                                     f"{pred}, reference {stream[i + 1]}")
                owned += 1
            elif pred >= 0:
                raise SystemExit(f"FAIL: the circuit owns the position after {S.name(t)}")
        assert oracle.released in ("YLD", "KWT")
    ctx = bytes(t - S.TEXT_TOKEN for t in sess.toks if S.is_text(t) and t - S.TEXT_TOKEN < 256)
    print(f"PASS: {owned} owned predictions equal the reference, {checked - owned} releases "
          f"unowned; display {bytes(oracle.screen)!r}, context bytes {ctx!r}; "
          f"{time.time() - t1:.0f} s on CPU", flush=True)


if __name__ == "__main__":
    main()
