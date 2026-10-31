"""Check the "inside" firmware (env/inside: wasm3 running the chat harness module) on the
reference machine. REFERENCE FIXTURE: no circuit and no language model. The "Qwen" replies are
fixed strings encoded with the Qwen3.5 tokenizer and played token by token; they are not a
live Qwen conversation.

  QWENPUTER_RV_MAP=inside python -m chipformer.rv.insidecheck     (from experimental/riscv)

Needs a local Qwen3.5 snapshot (tokenizer.json and chat_template.jinja): $QWEN_SNAPSHOT, or the
HF cache ($HF_HUB_CACHE or $HF_HOME/hub). Boots the firmware to its first key wait, types each
user line as key tokens, plays the canned replies, and checks Qwen's context against the chat
template's own rendering of the same conversation, and the terminal bytes. Reports the guest
instructions (ref.Machine.steps) per boot and per wake.
"""

import os
import statistics
import sys
import time
from pathlib import Path

assert os.environ.get("QWENPUTER_RV_MAP") == "inside", "run with QWENPUTER_RV_MAP=inside"

from chipformer.rv import bridge, elf, ref  # noqa: E402
from chipformer.rv import spec as S  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "env/inside"))
import template  # noqa: E402

from chipformer.rv.neural_harness import SYSTEM, TOOL  # noqa: E402

FIRMWARE = HERE / "env/inside/inside.elf"


def call(cmd: str) -> str:
    return (f"<tool_call>\n<function=harness>\n<parameter=command>\n{cmd}\n</parameter>\n"
            "</function>\n</tool_call>")


# (user line, [(assistant text before the call, command, tool output) ...], final answer)
TURNS = [
    ("hi", [("Let me compute that.", "2 3 + .", "5")], "2 + 3 = 5."),
    ("define square", [("", ": sq dup * ; 7 sq .", "defined sq\n49")], "sq is defined; 7 sq is 49."),
    ("use it again", [("", "9 sq .", "81")], "9 sq gives 81, so the dictionary persisted."),
]


class Check:
    def __init__(self):
        snap = template.snapshot()
        if snap is None:
            raise SystemExit("No Qwen3.5 snapshot: set QWEN_SNAPSHOT or HF_HUB_CACHE.")
        self.T = bridge.Text(snap / "tokenizer.json")
        self.m = ref.Machine.from_program(elf.program(FIRMWARE))
        self.s = bridge.Session(self.T, self.m)
        self.keys: set[int] = set()  # stream positions of key tokens (the terminal's, not Qwen's)
        self.wakes = {"key": [], "qwen": []}

    def wake(self, kind: str, qid: int) -> str:
        if kind == "key":
            self.keys.add(len(self.s.toks))
        steps, toks = self.m.steps, len(self.s.toks)
        self.s.wake(qid, max_records=10**8)
        self.wakes[kind].append((self.m.steps - steps, len(self.s.toks) - toks))
        return self.m.released

    def boot(self) -> dict:
        t0 = time.time()
        self.s.inside_run(max_records=10**8)
        assert self.m.released == "KWT", self.m.released
        return {"instructions": self.m.steps, "tokens": len(self.s.toks),
                "seconds": round(time.time() - t0, 1)}

    def type_line(self, line: str) -> None:
        for b in line.encode() + b"\r":
            assert self.wake("key", self.T.printable[b]) == "KWT" or b == 13
        assert self.m.released == "YLD", self.m.released

    def reply(self, text: str, end: str) -> None:
        ids = self.T.encode(text)
        for i, q in enumerate(ids):
            released = self.wake("qwen", q)
            assert released == (end if i == len(ids) - 1 else "YLD"), (i, released)

    def context(self) -> str:
        """What Qwen reads: the stream's text tokens, without the keys typed at a key wait."""
        qids = [t - S.TEXT_TOKEN for i, t in enumerate(self.s.toks)
                if S.is_text(t) and i not in self.keys]
        return self.T.decode_bytes(qids).decode()


def stats(xs: list[tuple[int, int]]) -> str:
    steps = [a for a, _ in xs]
    toks = [b for _, b in xs]
    return (f"{len(xs)} wakes; instructions mean {statistics.mean(steps):.0f}, median "
            f"{statistics.median(steps):.0f}, min {min(steps)}, max {max(steps)}; circuit tokens "
            f"mean {statistics.mean(toks):.0f}")


def main() -> None:
    print("REFERENCE FIXTURE: reference machine, canned Qwen replies, no model and no circuit")
    c = Check()
    print("boot:", c.boot())
    system = [{"role": "system", "content": SYSTEM}]
    head = template.system_turn(SYSTEM, TOOL)
    want, per_turn = head, []
    for user, calls, final in TURNS:
        n0 = len(c.wakes["qwen"])
        messages = list(system)
        c.type_line(user)
        messages.append({"role": "user", "content": user})
        for pre, cmd, result in calls:
            c.reply((pre + "\n\n" if pre else "") + call(cmd), "YLD")
            messages.append({"role": "assistant", "content": pre, "tool_calls": [
                {"type": "function", "function": {"name": "harness",
                                                  "arguments": {"command": cmd}}}]})
            messages.append({"role": "tool", "content": result})
        c.reply(final + "<|im_end|>", "KWT")
        messages.append({"role": "assistant", "content": final})
        per_turn.append(c.wakes["qwen"][n0:])
        # the turn as the template renders it when it is the latest query (its think block
        # kept: the stream is append-only, so earlier turns keep theirs too)
        turn = template.render(messages, tools=[TOOL])
        assert turn.startswith(head)
        want += turn[len(head):]

    ctx = c.context()
    if ctx != want:
        at = next((i for i, (a, b) in enumerate(zip(ctx, want)) if a != b), min(len(ctx), len(want)))
        print("context mismatch at", at, "\n guest:", repr(ctx[at - 80:at + 80]),
              "\n template:", repr(want[at - 80:at + 80]))
        raise SystemExit(1)
    print(f"Qwen's context equals the Qwen3.5 template's rendering, turn by turn ({len(ctx)} chars, "
          f"{len(c.s.qwen_text()) - len(c.keys)} Qwen tokens incl. the guest's byte tokens)")
    assert "<tool_response>\n5\n</tool_response>" in ctx

    screen = c.m.screen.decode()
    for part in ["\x1b[36m❯\x1b[0m hi\r\n", "Let me compute that.\r\n┌ harness · 2 3 + .\r\n"
                 "│ 5\r\n└ neural guest\r\n2 + 3 = 5.\r\n", "│ defined sq\r\n│ 49\r\n", "│ 81\r\n",
                 "9 sq gives 81, so the dictionary persisted.\r\n"]:
        assert part in screen, (part, screen)
    assert "<tool_call>" not in screen and "<parameter" not in screen
    print("screen (the terminal's bytes):")
    sys.stdout.write(screen.replace("\r\n", "\n"))
    print()
    print("key wakes:", stats(c.wakes["key"]))
    print("Qwen-token wakes:", stats(c.wakes["qwen"]))
    for i, w in enumerate(per_turn):
        print(f"  turn {i + 1}: {stats(w)}")
    big = sorted(c.wakes["qwen"], reverse=True)[:3]
    print("  largest Qwen wakes (tool runs + tool_response writes):", big)
    print(f"total instructions {c.m.steps:,}; stream {len(c.s.toks):,} tokens")
    print("OK (reference fixture)")


if __name__ == "__main__":
    main()
