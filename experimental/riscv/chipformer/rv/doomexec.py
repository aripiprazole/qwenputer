"""The Doom firmware on the GPU circuit executor, without Qwen: correctness and circuit speed.

  QWENPUTER_RV_MAP=doom python -m chipformer.rv.doomexec [--keys "w*4" ...] [--no-check]

The doom-map circuit (byte-level text bridge, as harness_fixture) runs on the Triton executor
exactly as RvChat runs its chip layers: fused independent sublayers, the device token loop
(ExecRow.run_chip) for runs of chip tokens. The firmware is booted, then each call (the start,
then one per --keys) is fed as its tool-call bytes and the circuit's own predictions are fed
back until it gives the position up. With the check on, every token is compared with the
reference machine's stream (the test oracle, run first on the host). Reports circuit tokens per
second; no language model is loaded, so there is no text decoding in these numbers.
"""

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from chipformer.rv import bridge, core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.chat import ExecRow, RvChat
from chipformer.rv.decode import CodeDecoder
from chipformer.rv.doomcheck import FIRMWARE
from chipformer.rv.harness_fixture import ByteText


class Circuit:
    """The parts of RvChat that ExecRow uses, for the circuit alone (ByteText model ids)."""

    chip_table = RvChat.chip_table
    code_token = RvChat.code_token
    _exec_emb = RvChat._exec_emb

    def __init__(self, c, dev: str, cache: Path) -> None:
        self.c, self.dev, self.fuse, self.path = c, dev, True, cache
        self.T = ByteText()
        self.v0 = len(self.T.bytes)
        self.code_dims = list(range(core.NFAM + core.NPAY))
        self.keep_ir = torch.tensor(c.keep, device=dev)
        self.keep_cols = list(range(len(c.keep)))
        self.decoder = CodeDecoder(self.v0, self.T, device=dev)
        self.emb_cache = {}
        self.erow = ExecRow(self)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", action="append", default=[])
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--cache", default=tempfile.gettempdir())
    ap.add_argument("--firmware", default=str(FIRMWARE))
    ap.add_argument("--tool", default="doom")
    ap.add_argument("--param", default="keys")
    ap.add_argument("--first", default="", help="the first call's argument")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False

    t0 = time.time()
    text = ByteText()
    c = core.build(text=text)
    prog = elf.program(args.firmware)
    core.snapshot(c, ref.Machine.from_program(prog))
    cir = Circuit(c, "cuda", Path(args.cache))
    row = cir.erow
    print(f"circuit {c.stats()}: {len(row.ex.steps)} executor steps (fused), "
          f"built in {time.time() - t0:.0f} s", flush=True)
    t0 = time.time()
    row._build_loop()
    print(f"device loop ready ({time.time() - t0:.0f} s; feature table cached in {args.cache})",
          flush=True)

    oracle = None
    if not args.no_check:
        oracle = ref.Machine.from_program(prog)
        bridge.park(oracle)
        sess = bridge.Session(text, oracle)

    def model(st):  # circuit stream token -> ByteText model id
        return st + cir.v0 if st < S.TEXT_TOKEN else st - S.TEXT_TOKEN

    # boot, as RvChat._exec_boot
    row.start([text.NL])
    tok, halt = cir.v0 + S.tok("GO"), cir.v0 + S.tok("HALT")
    while True:
        keep = row.step(tok)
        if tok == halt:
            break
        tok = cir.code_token(keep[: len(cir.code_dims)])
    print("booted (parked)", flush=True)

    for keys in [args.first] + args.keys:
        body = (f"<function={args.tool}><parameter={args.param}>{keys}</parameter>"
                "</function>").encode()
        call = [text.TCO, *body, text.TCC]
        if oracle is not None:
            t1 = time.time()
            k = len(sess.toks)
            sess.qwen(call, max_records=10**9)
            want = [model(t) for t in sess.toks[k + len(call):]]
            print(f"  reference stream: {len(want):,} tokens ({time.time() - t1:.0f} s, host)",
                  flush=True)
        for q in call:
            keep = row.step(q)
        nxt = cir.code_token(keep[: len(cir.code_dims)])
        got, screen = [], bytearray()
        dsp0 = S.FAMILIES["DSP"][0] if "DSP" in S.FAMILIES else -10**9
        torch.cuda.synchronize()
        t1 = time.time()
        last = [t1]

        def progress(block):
            n = len(got) + progress.n
            progress.n += len(block)
            if time.time() - last[0] > 30:
                last[0] = time.time()
                print(f"    {n:,} tokens, {n / (last[0] - t1):,.0f} tok/s", flush=True)

        progress.n = 0
        while nxt is not None and nxt >= 0:
            if nxt >= cir.v0:
                progress.n = 0
                fed, nxt = row.run_chip(nxt, 10**9, progress)
            else:  # an echoed text token: one host step
                fed = [nxt]
                keep = row.step(nxt)
                nxt = cir.code_token(keep[: len(cir.code_dims)])
            got += fed
        torch.cuda.synchronize()
        dt = time.time() - t1
        for t in got:
            if t >= cir.v0 and 0 <= t - cir.v0 - dsp0 < 256:
                screen.append(t - cir.v0 - dsp0)
        echo = bytes(t for t in got if t < 256)
        if screen:
            if 14 in screen:  # a color frame: half blocks (escape codes) for the log
                from chipformer.rv.doom_harness import COLS, ROWS, halfblocks
                at = screen.rindex(14) + 1
                print("\n".join(halfblocks(bytes(screen[at:at + 2 * ROWS * COLS]))))
            else:
                print(screen.decode("latin-1").split("\f")[-1])
        print(f"call {keys!r}: {len(got):,} circuit tokens in {dt:.1f} s = "
              f"{len(got) / dt:,.0f} tok/s (circuit only, GPU executor); Qwen would read "
              f"{echo.strip()!r}", flush=True)
        if oracle is not None:
            a, b = np.array(got, np.int64), np.array(want, np.int64)
            n = min(len(a), len(b))
            bad = np.nonzero(a[:n] != b[:n])[0]
            if len(bad) or len(a) != len(b):
                i = int(bad[0]) if len(bad) else n
                raise SystemExit(f"FAIL at token {i} of the call: circuit "
                                 f"{a[i] if i < len(a) else 'end'}, reference "
                                 f"{b[i] if i < len(b) else 'end'}")
            print(f"  PASS: all {len(a):,} tokens equal the reference stream", flush=True)


if __name__ == "__main__":
    main()
