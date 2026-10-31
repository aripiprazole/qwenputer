"""Qwen3.5's chat template, rendered as transformers renders it (no model is loaded).

Used at build time for the system turn the guest writes (prefix.h) and by insidecheck.py to
compare the guest-written context with the template's own rendering of the same conversation."""

import glob
import json
import os
from pathlib import Path

SPECIALS = ["<tool_response>", "</tool_response>", "<think>", "</think>", "<|im_start|>",
            "<|im_end|>", "<tool_call>", "</tool_call>"]  # putchar(256 + k), bridge.SPECIALS


def snapshot() -> Path | None:
    """A local Qwen3.5 snapshot directory (tokenizer.json, chat_template.jinja): $QWEN_SNAPSHOT,
    or the HF cache ($HF_HUB_CACHE, $HF_HOME/hub)."""
    if os.environ.get("QWEN_SNAPSHOT"):
        return Path(os.environ["QWEN_SNAPSHOT"])
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    for name in ("Qwen3.5-4B", "Qwen3.5-0.8B", "Qwen3.5-9B"):
        hits = sorted(glob.glob(str(hub / f"models--Qwen--{name}" / "snapshots" / "*" /
                                    "chat_template.jinja")))
        if hits:
            return Path(hits[-1]).parent
    return None


def render(messages, tools=None, add_generation_prompt=False, enable_thinking=False,
           template: Path | None = None) -> str:
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                          sort_keys=sort_keys)

    template = template or snapshot() / "chat_template.jinja"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(Path(template).read_text()).render(
        messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking)


def system_turn(system: str, tool: dict) -> str:
    """The template's system turn for this system prompt and one tool (through <|im_end|>\\n)."""
    text = render([{"role": "system", "content": system}, {"role": "user", "content": "x"}],
                  tools=[tool])
    end = text.index("<|im_end|>\n") + len("<|im_end|>\n")
    return text[:end]


def c_literal(text: str) -> str:
    """A C string literal of text's UTF-8 bytes, special tokens as the bytes 1..8."""
    for k, s in enumerate(SPECIALS):
        text = text.replace(s, chr(k + 1))
    out, line = [], ""
    for b in text.encode():
        if 32 <= b < 127 and chr(b) not in '"\\?':
            line += chr(b)
        else:
            line += f"\\{b:03o}"
        if b == 10:
            out.append(f'"{line}"')
            line = ""
    if line:
        out.append(f'"{line}"')
    return "\n    ".join(out)


# The fixed text the guest writes, as Qwen's own tokens (its long system turn read one byte per
# token degrades Qwen's tool calls). putchar values: 0..255 a byte's token, 256 + k a special,
# 264 + j the j-th token of the table (tokens.json, which the bridge appends to the printables).
PIECES = {  # the turn pieces the module writes around user text and tool output
    "USER_OPEN": "<|im_start|>user\n",
    "TO_ASSISTANT": "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "TOOL_OPEN": "<|im_end|>\n<|im_start|>user\n<tool_response>\n",
    "TOOL_CLOSE": "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "TURN_END": "<|im_end|>\n",
}


def token_arrays(system: str, tool: dict, snap: Path) -> tuple[dict, list[int]]:
    """{name: putchar values} for PREFIX (the system turn) and PIECES, and the token table."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    vocab = tok.get_vocab(with_added_tokens=True)
    special = {vocab[s]: 256 + k for k, s in enumerate(SPECIALS)}
    from chipformer.rv.bridge import bytes_to_unicode
    byte_of = {vocab[c]: b for b, c in bytes_to_unicode().items()}  # as bridge.Text.printable
    table: list[int] = []
    texts = {"PREFIX": system_turn(system, tool), **PIECES}
    arrays = {}
    for name, text in texts.items():
        out = []
        for i in tok.encode(text, add_special_tokens=False).ids:
            if i in special:
                out.append(special[i])
            elif i in byte_of:
                out.append(byte_of[i])
            else:
                if i not in table:
                    table.append(i)
                out.append(264 + table.index(i))
        arrays[name] = out
    if 264 + len(table) > 512:
        raise SystemExit(f"{len(table)} table tokens: more than the 248 free BYTE payloads")
    return arrays, table


def c_arrays(arrays: dict) -> str:
    out = []
    for name, vals in arrays.items():
        body = ", ".join(map(str, vals))
        out.append(f"static const unsigned short {name}[] = {{{body}}};\n"
                   f"#define {name}_N {len(vals)}\n")
    return "".join(out)
