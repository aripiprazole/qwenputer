"""Bounded offline check of the "doom" circuit: no pretrained Qwen, CPU sparse tensors.

  QWENPUTER_RV_MAP=doom python -m chipformer.rv.doomfixture [--steps N]

The compiled circuit (core.build for the doom address map, with the byte-level text bridge of
harness_fixture) runs the Doom firmware from its ELF snapshot: boot to the first park, then the
``doom`` tool call that starts the game. Every token the circuit predicts is fed back to it
(free-running) and compared with the reference machine's stream (the test oracle) for the first
N tokens of the call. This is a correctness fixture, not a game session and not a speed figure.
"""

import argparse
import time

from chipformer.rv import spec as S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--keys", default="")
    args = ap.parse_args()
    assert S.MAP == "doom", "run with QWENPUTER_RV_MAP=doom"

    import torch

    from chipformer.rv import bridge, core, elf, ref
    from chipformer.rv.decode import CodeDecoder
    from chipformer.rv.doomcheck import FIRMWARE
    from chipformer.rv.fixture_tensors import SparseTensorCircuit
    from chipformer.rv.harness_fixture import ByteText

    torch.set_num_threads(4)
    text = ByteText()
    t0 = time.time()
    c = core.build(text=text)
    prog = elf.program(FIRMWARE)
    core.snapshot(c, ref.Machine.from_program(prog))
    ex = SparseTensorCircuit(c)
    offset = len(text.bytes)
    dec = CodeDecoder(offset, text)
    print(f"circuit: {c.stats()} built in {time.time() - t0:.0f} s", flush=True)

    # the oracle: the same firmware, the same call, as a token stream
    oracle = ref.Machine.from_program(prog)
    bridge.park(oracle)
    sess = bridge.Session(text, oracle)
    body = f"<function=doom><parameter=keys>{args.keys}</parameter></function>".encode()
    call = [text.TCO, *body, text.TCC]
    try:
        sess.qwen(call, max_records=args.steps)
    except RuntimeError:  # the bound cut the run short; its tokens so far are in the stream
        pass
    want = sess.toks

    def model_id(st):  # circuit stream token -> fixture id (decoder space)
        return st + offset if st < S.TEXT_TOKEN else st - S.TEXT_TOKEN

    def step(mid):
        st = mid - offset if mid >= offset else S.TEXT_TOKEN | mid
        with torch.no_grad():
            h = ex.step_token(st)
            return int(dec(h[c.keep[:core.NFAM + core.NPAY]]))

    # boot: GO, then free-run to the first HALT (16 instructions)
    tok, n = offset + S.tok("GO"), 0
    while True:
        prev, tok = tok, step(tok)
        n += 1
        if prev == offset + S.tok("HALT"):
            break
        assert tok >= 0 and n < 1000, ("boot lost", tok, n)
    print(f"boot: parked after {n} circuit steps", flush=True)

    # the call: Qwen's tokens are teacher-forced (the circuit predicts GO at </tool_call>), then
    # the circuit's own predictions are fed back
    t1, i, bad = time.time(), 0, None
    for q in call:
        assert want[i] == S.TEXT_TOKEN | q
        pred = step(q)
        i += 1
    while i < len(want):
        got = pred - offset if pred >= offset else S.TEXT_TOKEN | pred if pred >= 0 else pred
        if got != want[i]:
            bad = (i, S.name(got) if 0 <= got < S.VOCAB else got, S.name(want[i])
                   if want[i] < S.VOCAB else want[i])
            break
        pred = step(model_id(want[i]))
        i += 1
        if i % 1000 == 0:
            print(f"  {i} tokens equal ({(time.time() - t1) / i * 1e3:.1f} ms/token)", flush=True)
    dsp = sum(1 for t in want[:i] if t < S.VOCAB and S.family(t)[0] == "DSP")
    if bad:
        raise SystemExit(f"FAIL at stream token {bad[0]}: circuit {bad[1]}, reference {bad[2]}")
    print(f"PASS: {i} free-running circuit tokens equal the reference ({dsp} display tokens); "
          f"{time.time() - t1:.0f} s on CPU", flush=True)


if __name__ == "__main__":
    main()
