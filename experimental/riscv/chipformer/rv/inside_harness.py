"""The harness inside: the guest drives the conversation, the host only runs the model.

    QWENPUTER_RV_MAP=inside python -m chipformer.rv.inside_harness --server --model PATH

The firmware (env/inside: wasm3 on rv32im, interpreting the harness module) runs in the circuit
layers of a merged "inside" checkpoint. It writes the whole prompt into Qwen's context (echoes),
hands positions to Qwen (YLD) or to the keyboard (KWT), is woken by every token that arrives
while it is parked, and draws the terminal itself (DSP tokens). This host has no chat logic: no
template, no tokenizer call, no stop rule, no formatting. It runs the layers, feeds each
predicted token back, turns DSP tokens into bytes on stdout and keys into tokens the circuit
reads (they go through the chip layers only, so Qwen never sees them).
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from chipformer.rv import spec as S

FIRMWARE = Path(__file__).parent / "env/inside/inside.elf"
QUIT = 0x1C  # Ctrl-\ closes the terminal session (the host's only key of its own)


class Pump:
    """Model execution and terminal bytes for one guest, on the executor path."""

    def __init__(self, chat, read_key, write_screen) -> None:
        self.chat, self.read_key, self.write = chat, read_key, write_screen
        self.boot_screen = bytearray()  # what the guest drew before its first key wait
        import time
        self.ticks, self.next_report, self.t0 = 0, 100_000, time.time()
        self.progress = os.environ.get("QWENPUTER_PROGRESS") == "1"  # stderr, every 100k
        self.trace = os.environ.get("QWENPUTER_TRACE") == "1"
        self.dsp_lo, n = S.FAMILIES["DSP"]
        self.dsp_lo += chat.v0
        self.dsp_hi = self.dsp_lo + n
        self.yld, self.kwt = chat.v0 + S.tok("YLD"), chat.v0 + S.tok("KWT")
        self.stats = {"chip": 0, "echo": 0, "qwen": 0, "keys": 0}

    def _screen(self, block) -> None:
        self.ticks += len(block)
        if self.progress and self.ticks >= self.next_report:
            import time
            self.next_report += 100_000
            print(f"[{time.time() - self.t0:.0f} s: {self.ticks:,} circuit tokens, "
                  f"{self.pos} in Qwen's context]", file=sys.stderr, flush=True)
        out = bytes(t - self.dsp_lo for t in block if self.dsp_lo <= t < self.dsp_hi)
        if out:
            if not self.stats["keys"]:
                self.boot_screen += out
            self.write(out)

    SHORT = 32  # guest runs start as single steps: most are a few records (a byte, a wake)

    def _run(self, first: int) -> tuple[list, int]:
        """A run of the guest from the chip token ``first``: single steps while it is short, then
        the device token loop (whose 256-step blocks would mostly be discarded on a short run).
        Returns (the tokens fed, the prediction after them) as ExecRow.run_chip does."""
        c = self.chat
        fed, t = [], first
        for _ in range(self.SHORT):
            fed.append(t)
            keep = c._exec_step(t)
            nxt = c.code_token(keep[:len(c.code_dims)])
            if nxt is None or nxt < c.v0:
                self._screen([t])
                return fed, -1 if nxt is None else nxt
            self._screen([t])
            t = nxt
        more, pred = c.erow.run_chip(t, 1 << 40, self._screen)
        return fed + more, pred

    def _qwen(self, q: int):
        """A text token into Qwen's context: the chip layers, then Qwen's layers. Returns the
        circuit's prediction for the next position (None: not owned)."""
        import torch

        c = self.chat
        keep = c._exec_step(q)
        hc = c._exec_hc(q, keep).view(1, 1, -1)
        p4 = torch.full((4, 1, 1), self.pos, device=c.dev, dtype=torch.long)
        pe = c.lm.rotary_emb(hc, p4[1:])
        h = hc
        for i in range(c.n_chip, len(c.lm.layers)):
            h = c.lm.layers[i](h, position_embeddings=pe, attention_mask=None,
                               position_ids=p4[0], past_key_values=self.cache)
        self.last = h
        self.pos += 1
        return c.code_token(keep[:len(c.code_dims)])

    def _qwen_logits(self):
        """Qwen's own prediction after its last text position. The circuit owned that position
        (a wake, or the guest's last echo), so the exit wrote the circuit's output code there
        and set tau as on an owned token; the tied head would score the printable tokens by
        that code. Qwen's layers never read these dims: the prediction is read with them as
        on a position nobody owns (no code, the text tau)."""
        from chipformer.rv import core

        c = self.chat
        h = self.last.clone()
        h[..., c.keep_cols[:len(c.code_dims)]] = 0
        h[..., c.meta["tau_dim"]] = core.TAU_TEXT
        return c._project(c.lm.norm(h), full_head=False)[0, -1]

    def _key(self, q: int):
        """A key's token: the chip layers only (Qwen's context does not change)."""
        c = self.chat
        keep = c._exec_step(q)
        return c.code_token(keep[:len(c.code_dims)])

    def _park_state(self) -> dict:
        """Everything that continues a parked guest: every circuit head's state and lag window,
        Qwen's cache (the context the guest wrote), the position, Qwen's last hidden state."""
        heads = [(st["S"].clone(), st["hist"].clone()) for st in self.chat.erow.heads]
        return {"heads": heads, "cache": self.cache, "pos": self.pos, "last": self.last,
                "screen": bytes(self.boot_screen), "stats": dict(self.stats)}

    def _resume(self, saved: dict) -> None:
        for st, (S_, h) in zip(self.chat.erow.heads, saved["heads"], strict=True):
            st["S"].copy_(S_)
            st["hist"].copy_(h)
        self.cache, self.pos, self.last = saved["cache"], saved["pos"], saved["last"]
        self.stats.update(saved["stats"])
        self.write(saved["screen"])

    def run(self, program, stop=lambda: False, state_path: Path | None = None) -> None:
        """Boot the guest (or resume it parked at its first key wait from ``state_path``, a
        cache of that park; written after the first boot) and serve it until a quit."""
        import torch
        from transformers import DynamicCache

        from chipformer.rv import core, ref

        c = self.chat
        core.snapshot(c.c, ref.Machine.from_program(program))
        c._exec_start([c.T.NL])  # the circuit's lag windows, as boot does; then the snapshot
        self.cache = DynamicCache(config=c.model.config)
        self.pos, self.last = 0, None
        nxt, booted = c.v0 + S.tok("GO"), False
        if state_path is not None and state_path.is_file():
            self._resume(torch.load(state_path, map_location=c.dev, weights_only=False))
            nxt, booted = self.kwt, True
        with torch.no_grad():
            while not stop():
                if nxt >= c.v0:  # a run of the guest
                    if nxt == self.kwt and booted and not self.stats["keys"]:
                        fed, pred = [nxt], -1  # resumed at the boot park
                    else:
                        fed, pred = self._run(nxt)
                        self.stats["chip"] += len(fed)
                    if pred >= 0:
                        nxt = pred  # an echo: the guest writes into Qwen's context
                        continue
                    if pred != -1 or fed[-1] not in (self.yld, self.kwt):
                        raise RuntimeError(f"the guest stopped without a park ({pred})")
                    if fed[-1] == self.kwt:
                        if not booted:  # the first key wait: the guest is booted
                            booted = True
                            if state_path is not None:
                                torch.save(self._park_state(), state_path)
                        b = self.read_key()
                        if b is None or b == QUIT:
                            return
                        self.stats["keys"] += 1
                        nxt = self._key(c.T.printable[b])
                    else:
                        if self.last is None:
                            raise RuntimeError("the guest yielded before writing any text")
                        logits = self._qwen_logits()
                        q = int(torch.argmax(logits[: c.v0]))
                        self.stats["qwen"] += 1
                        if self.trace:  # a log of Qwen's tokens (stderr); no effect on the run
                            print(f"[qwen {q} {c.T.bytes[q]!r}]", file=sys.stderr, flush=True)
                        nxt = self._qwen(q)
                else:  # an echo the circuit predicted: Qwen reads it
                    self.stats["echo"] += 1
                    nxt = self._qwen(nxt)
                if nxt is None:
                    raise RuntimeError("the circuit gave up a position")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", action="store_true", help="run on this CUDA machine")
    ap.add_argument("--model", required=True, help="merged inside checkpoint directory")
    ap.add_argument("--firmware", default=str(FIRMWARE))
    ap.add_argument("--keys", help="feed these keys instead of the terminal (\\r = enter)")
    ap.add_argument("--no-cache", action="store_true", help="always boot from the snapshot")
    args = ap.parse_args()
    if not S.INSIDE:
        raise SystemExit("set QWENPUTER_RV_MAP=inside")
    if not args.server:
        raise SystemExit("run with --server on the GPU host (see README)")

    from chipformer.rv import elf
    from chipformer.rv.chat import RvChat

    chat = RvChat(args.model, device="cuda", fast=False, executor=True, fuse=True)
    program = elf.program(Path(args.firmware))
    state = None
    if not args.no_cache:  # the booted guest, keyed by the firmware and the checkpoint
        key = hashlib.sha256(Path(args.firmware).read_bytes())
        key.update((Path(args.model) / "chipformer.json").read_bytes())
        state = Path(args.model) / f"inside-boot.{key.hexdigest()[:16]}.pt"

    def write(b: bytes) -> None:
        os.write(1, b)

    if args.keys is not None:
        keys = list(args.keys.encode().decode("unicode_escape").encode())
        pump = Pump(chat, lambda: keys.pop(0) if keys else None, write)
        pump.run(program, state_path=state)
        print(f"\n{pump.stats}", file=sys.stderr)
        return

    import termios
    import tty

    fd = 0
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        pump = Pump(chat, lambda: os.read(fd, 1)[0], write)
        pump.run(program, state_path=state)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print(f"\r\n{pump.stats}")


if __name__ == "__main__":
    main()
