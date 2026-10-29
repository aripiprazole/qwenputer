"""Executor launch configs on the same token stream: every residual value must be equal.

  QWENPUTER_RV_MAP=doom python -m chipformer.rv.execcompare [--tokens N] [--skip N]

The stream is the Doom start call's (the reference machine's tokens after the firmware's boot,
teacher-forced); --skip starts the comparison that many tokens in (the earlier ones still run,
on the reference variant only, and its state is copied to the others). Variant 0 is the default
config (a program per value head, 4 warps, MLP blocks of 64) with every weight in fp32; the
others keep exact weights in fp16 (Executor2's compact storage), use programmatic dependent
launch with the early trigger, add independent MLP steps into the residual directly, and
force one candidate of Executor2.tune on every step; the last is the tuned choice. Prints each
variant's time.
"""

import argparse
import time

import torch

from chipformer.rv import bridge, core, elf, ref
from chipformer.rv import spec as S
from chipformer.rv.doomcheck import FIRMWARE
from chipformer.rv.harness_fixture import ByteText
from chipformer.rv.tritonexec import Executor2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=20000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--firmware", default=str(FIRMWARE))
    args = ap.parse_args()
    text = ByteText()
    c = core.build(text=text)
    prog = elf.program(args.firmware)
    core.snapshot(c, ref.Machine.from_program(prog))
    m = ref.Machine.from_program(prog)
    bridge.park(m)
    sess = bridge.Session(text, m)
    body = b"<function=doom><parameter=keys></parameter></function>"
    try:
        sess.qwen([text.TCO, *body, text.TCC], max_records=args.skip + args.tokens)
    except RuntimeError:
        pass
    toks = sess.toks[: args.skip + args.tokens]

    head_cfgs = [(kh, nw) for kh in (False, True) for nw in (4, 8, 16)]
    ffn_cfgs = [(blk, nw) for blk in (16, 32, 64) for nw in (1, 2, 4, 8)]
    forced = [(head_cfgs[i % len(head_cfgs)], ffn_cfgs[i % len(ffn_cfgs)])
              for i in range(max(len(head_cfgs), len(ffn_cfgs)))]
    names = ["default, fp32 weights"] + [f"head {h} ffn {f}" for h, f in forced] + ["tuned"]
    variants = [Executor2(c, 1, "cuda", fuse=True, compact=i > 0, pdl=i > 0, early=i > 0,
                          direct=i > 0) for i in range(len(names))]
    for ex, cfg in zip(variants[1:-1], forced):
        for st in ex.steps:
            st["cfg"] = cfg[0] if st["kind"] == "head" else cfg[1]
    for ex in variants:
        ex.load_init()
    tuned = variants[-1].tune(variants[-1].embed([S.tok("GO")]))
    print("tuned:", tuned, flush=True)
    vocab = sorted(set(toks))
    emb = variants[0].embed(vocab)
    row = {t: i for i, t in enumerate(vocab)}

    for t in toks[: args.skip]:
        variants[0].step(emb[row[t]: row[t] + 1].clone())
    if args.skip:
        for ex in variants[1:]:
            for a, b in zip(variants[0].steps, ex.steps):
                if a["kind"] == "head":
                    b["S"].copy_(a["S"])
                    if a["hist"].shape != b["hist"].shape:
                        raise SystemExit("--skip needs variants with the same history layout")
                    b["hist"].copy_(a["hist"])
    times = [0.0] * len(variants)
    for i, t in enumerate(toks[args.skip:]):
        outs = []
        for j, ex in enumerate(variants):
            x = emb[row[t]: row[t] + 1].clone()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ex.step(x)
            torch.cuda.synchronize()
            times[j] += time.perf_counter() - t0
            outs.append(x)
        for j in range(1, len(variants)):
            if not torch.equal(outs[0], outs[j]):
                bad = torch.nonzero(outs[0] != outs[j])[:5, 1].tolist()
                raise SystemExit(f"FAIL: {names[j]} at token {args.skip + i} "
                                 f"({S.name(t) if t < S.VOCAB else 'text'}): "
                                 f"dims {[c.dims[d].name for d in bad]}")
    n = len(toks) - args.skip
    print(f"PASS: {n} tokens, {len(variants)} configs, every residual value equal")
    for name, tt in zip(names, times):
        print(f"  {name:32s} {tt / n * 1e3:.3f} ms/token (eager, synced)")


if __name__ == "__main__":
    main()
