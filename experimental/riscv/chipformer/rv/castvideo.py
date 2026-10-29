"""Render an asciicast v2 recording (termrec.py) as an MP4 in a Ghostty-style window.

  python -m chipformer.rv.castvideo IN.cast OUT.mp4 [--keys termrec.log] [--title TEXT]

Ghostty's default look (background #282c34, its 16-color palette, JetBrains Mono at 13 pt,
drawn at 2x for a retina-sharp video) in a macOS window with a transparent title bar. The
stretches where the circuit works and nobody types (the model loading, the game booting, each
move from its keys to its frame) are fast-forwarded to --ff seconds each, with a badge showing
how much (e.g. "▶▶ 21x"); everything else plays in real time (other waits longer than
--max-gap are shortened, a new game frame stays up --hold seconds). The real elapsed time is
always in the title bar. With --keys (the recorder's log) every key typed or sent is shown.
"""

import argparse
import ast
import json
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from chipformer.rv.castgif import CSI

FONT_DIR = ("/nix/store/9zxqksq18da8sn889px07m43k1nxvjif-nerd-fonts-jetbrains-mono-3.4.0+2.304/"
            "share/fonts/truetype/NerdFonts/JetBrainsMono/")
BG, FG = (0x28, 0x2C, 0x34), (0xFF, 0xFF, 0xFF)
PALETTE = ["#1d1f21", "#cc6666", "#b5bd68", "#f0c674", "#81a2be", "#b294bb", "#8abeb7",
           "#c5c8c6", "#666666", "#d54e53", "#b9ca4a", "#e7c547", "#7aa6da", "#c397d8",
           "#70c0b1", "#eaeaea"]  # Ghostty's default palette
PAL = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in PALETTE]
DIM = tuple((a + b) // 2 for a, b in zip(FG, BG))  # SGR 2: half way to the background


def xterm(n: int) -> tuple:
    """xterm's 256 colors: Ghostty's 16, the 6x6x6 cube, the 24 greys."""
    if n < 16:
        return PAL[n]
    if n < 232:
        n -= 16
        return tuple((0, 95, 135, 175, 215, 255)[v] for v in (n // 36, n // 6 % 6, n % 6))
    return (8 + 10 * (n - 232),) * 3
KEY_NAMES = {"\x1b[A": "↑", "\x1b[B": "↓", "\x1b[C": "→", "\x1b[D": "←", " ": "Space",
             "\t": "Tab", "\n": "Return", "\x1b": "Esc"}


class Screen:
    """The terminal state: cells of (char, color, bold); what the raw harness writes."""

    def __init__(self, cols: int, rows: int) -> None:
        self.cols, self.rows = cols, rows
        self.lines = [self._blank() for _ in range(rows)]
        self.x = self.y = 0
        self.fg, self.bold, self.bg = FG, False, None
        self.held = ""

    def _blank(self):
        return [(" ", FG, False, None)] * self.cols

    def _newline(self) -> None:
        self.y += 1
        if self.y >= self.rows:
            self.lines.pop(0)
            self.lines.append(self._blank())
            self.y = self.rows - 1

    def feed(self, text: str) -> None:
        text, self.held = self.held + text, ""
        i = 0
        while i < len(text):
            m = CSI.match(text, i)
            if m:
                self._csi(m[1], m[2])
                i = m.end()
                continue
            if text[i] == "\x1b" and re.fullmatch(r"\x1b(\[[?0-9;]*)?", text[i:]):
                self.held = text[i:]  # an escape code cut at the chunk's end: wait for the rest
                return
            c = text[i]
            i += 1
            if c == "\r":
                self.x = 0
            elif c == "\n":
                self._newline()
            elif c == "\b":
                self.x = max(0, self.x - 1)
            elif ord(c) >= 32 and c != "\x7f":
                if self.x >= self.cols:
                    self.x = 0
                    self._newline()
                self.lines[self.y][self.x] = (c, self.fg, self.bold, self.bg)
                self.x += 1

    def _csi(self, arg: str, op: str) -> None:
        if arg.startswith("?"):
            if arg == "?1049" and op == "h":  # the alternate screen (the game's)
                self.saved = (self.lines, self.x, self.y)
                self.lines = [self._blank() for _ in range(self.rows)]
            elif arg == "?1049" and op == "l" and getattr(self, "saved", None):
                self.lines, self.x, self.y = self.saved
                self.saved = None
            return
        nums = [int(n) for n in arg.split(";") if n] or [0]
        if op == "H":  # cursor position (1-based row;col)
            self.y = min(self.rows - 1, max(0, (nums[0] or 1) - 1))
            self.x = min(self.cols - 1, max(0, (nums[1] if len(nums) > 1 else 1) - 1))
        elif op == "J" and nums[0] == 2:
            self.lines = [self._blank() for _ in range(self.rows)]
        elif op == "K":
            start = 0 if nums[0] == 2 else self.x
            for x in range(start, self.cols):
                self.lines[self.y][x] = (" ", FG, False, None)
        elif op == "D":
            self.x = max(0, self.x - (nums[0] or 1))
        elif op == "C":
            self.x = min(self.cols - 1, self.x + (nums[0] or 1))
        elif op == "m":
            it = iter(nums)
            for n in it:
                if n in (38, 48) and next(it, None) == 5:  # 256 colors
                    color = xterm(next(it, 0))
                    if n == 38:
                        self.fg = color
                    else:
                        self.bg = color
                elif n == 0:
                    self.fg, self.bold, self.bg = FG, False, None
                elif n == 1:
                    self.bold = True
                elif n == 2:
                    self.fg = DIM
                elif 30 <= n <= 37:
                    self.fg = PAL[n - 30]
                elif 90 <= n <= 97:
                    self.fg = PAL[n - 82]


class Painter:
    def __init__(self, cols: int, rows: int, scale: int, title: str) -> None:
        s = self.s = scale
        self.font = ImageFont.truetype(FONT_DIR + "JetBrainsMonoNerdFontMono-Regular.ttf", 13 * s)
        self.bold = ImageFont.truetype(FONT_DIR + "JetBrainsMonoNerdFontMono-Bold.ttf", 13 * s)
        self.ui = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 12 * s)
        self.ui_bold = ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", 15 * s)
        asc, desc = self.font.getmetrics()
        self.cw, self.ch = self.font.getlength("M"), asc + desc + 2 * s
        self.title = title
        pad, bar = 10 * s, 28 * s
        self.term = (round(cols * self.cw) + 2 * pad, round(rows * self.ch) + 2 * pad)
        self.win = (self.term[0], self.term[1] + bar)
        margin = 36 * s
        w, h = self.win[0] + 2 * margin, self.win[1] + 2 * margin
        self.size = (w + w % 2, h + h % 2)  # even, for yuv420p
        self.origin = (margin, margin)
        self.pad, self.bar = pad, bar
        self.backdrop = self._backdrop()

    def _backdrop(self) -> Image.Image:
        s = self.s
        img = Image.new("RGB", self.size, (0x16, 0x18, 0x1d))
        shadow = Image.new("L", self.size, 0)
        ox, oy = self.origin
        ImageDraw.Draw(shadow).rounded_rectangle(
            [ox, oy + 8 * s, ox + self.win[0], oy + self.win[1] + 8 * s], 12 * s, fill=150)
        img.paste((0, 0, 0), mask=shadow.filter(ImageFilter.GaussianBlur(18 * s)))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([ox, oy, ox + self.win[0], oy + self.win[1]], 12 * s, fill=BG,
                            outline=(0x3a, 0x3f, 0x4a), width=s)
        for i, color in enumerate(((0xFF, 0x5F, 0x57), (0xFE, 0xBC, 0x2E), (0x28, 0xC8, 0x40))):
            cx, cy = ox + (20 + 20 * i) * s, oy + 14 * s
            d.ellipse([cx - 6 * s, cy - 6 * s, cx + 6 * s, cy + 6 * s], fill=color)
        tw = d.textlength(self.title, font=self.ui)
        d.text((ox + (self.win[0] - tw) / 2, oy + 7 * s), self.title, font=self.ui,
               fill=(0x9d, 0xa5, 0xb4))
        return img

    def frame(self, screen: Screen, clock: str, keys: list[str], badge: str = "") -> Image.Image:
        s = self.s
        img = self.backdrop.copy()
        d = ImageDraw.Draw(img)
        x0, y0 = self.origin[0] + self.pad, self.origin[1] + self.bar + self.pad
        for y, line in enumerate(screen.lines):
            for x, (c, fg, bold, bg) in enumerate(line):  # colored cells: blocks, not glyphs
                if bg is None and c != "▀":
                    continue
                # whole pixels, each block meeting the next (no hairline gaps)
                px, px2 = round(x0 + x * self.cw), round(x0 + (x + 1) * self.cw) - 1
                py, py2 = round(y0 + y * self.ch), round(y0 + (y + 1) * self.ch) - 1
                if bg is not None:
                    d.rectangle([px, py, px2, py2], fill=bg)
                if c == "▀":
                    d.rectangle([px, py, px2, (py + py2) // 2], fill=fg)
            run, start, style = "", 0, None
            for x, (c, fg, bold, bg) in enumerate(line + [(" ", None, False, None)]):
                if c == "▀":
                    c = " "
                if (fg, bold) != style:
                    if run.strip():
                        d.text((x0 + start * self.cw, y0 + y * self.ch), run,
                               font=self.bold if style[1] else self.font, fill=style[0])
                    run, start, style = "", x, (fg, bold)
                run += c
        # the real clock, right of the title bar
        ox, oy = self.origin
        tw = d.textlength(clock, font=self.ui)
        d.text((ox + self.win[0] - tw - 14 * s, oy + 7 * s), clock, font=self.ui,
               fill=(0xe5, 0xc0, 0x7b))
        if badge:  # fast-forward: how much, in the lower left corner
            tw = d.textlength(badge, font=self.ui_bold)
            x, y = ox + 24 * s, oy + self.win[1] - 54 * s
            d.rounded_rectangle([x, y, x + tw + 28 * s, y + 34 * s], 8 * s, fill=(0xe5, 0xc0, 0x7b))
            d.text((x + 14 * s, y + 7 * s), badge, font=self.ui_bold, fill=(0x28, 0x2c, 0x34))
        if keys:  # the keys just pressed, as key caps in the lower right corner
            caps = [k for k in keys][-8:]
            widths = [d.textlength(k, font=self.ui_bold) + 22 * s for k in caps]
            x = ox + self.win[0] - sum(widths) - 8 * s * (len(caps) - 1) - 24 * s
            y = oy + self.win[1] - 54 * s
            for k, w in zip(caps, widths):
                d.rounded_rectangle([x, y, x + w, y + 34 * s], 8 * s, fill=(0x3b, 0x41, 0x4d),
                                    outline=(0x5c, 0x63, 0x70), width=s)
                d.text((x + 11 * s, y + 7 * s), k, font=self.ui_bold, fill=FG)
                x += w + 8 * s
        return img


def mend_cut_blocks(events):
    """Recordings made before termrec decoded incrementally hold a character cut between two
    reads as replacement characters on both sides. In a frame row the cut character is a half
    block (its neighbours are half blocks or color codes); put it back."""
    out = [list(e) for e in events]
    for a, b in zip(out, out[1:]):
        if a[2].endswith("\ufffd") and b[2].startswith("\ufffd"):
            left, right = a[2].rstrip("\ufffd"), b[2].lstrip("\ufffd")
            if left.endswith(("▀", "m")) or right.startswith(("▀", "\x1b[")):
                a[2], b[2] = left, "▀" + right
    return [tuple(e) for e in out]


# where the circuit starts and stops working, in what the harness prints
BUSY_START = re.compile(r"working in the neural circuit|[▸⟳] move \d+:")
BUSY_END = re.compile(r"game started|move \d+ done|neural steps|stopped")


def busy_spans(events, ready="Ctrl-D to exit"):
    """(start, end) real times of the stretches where the circuit works: the start (loading)
    up to the prompt, and each turn or move from its start marker to its end marker."""
    spans, start = [], None
    out = [(t, d) for t, kind, d in events if kind == "o"]
    for t, d in out:
        if ready in d:
            spans.append((out[0][0], t))
            break
    for t, d in out:
        if start is None and BUSY_START.search(d):
            start = t
        elif start is not None and BUSY_END.search(d):
            spans.append((start, t))
            start = None
    return spans


def key_events(log: str) -> list[tuple[float, str]]:
    """(seconds, key cap) for what termrec typed (60 ms apart) or sent (at once)."""
    out = []
    for line in open(log):
        m = re.match(r"\[termrec\]\s+([\d.]+)s (typed|sent) (.*)$", line.strip())
        if not m:
            continue
        t, how, text = float(m[1]), m[2], ast.literal_eval(m[3])
        if how == "sent":
            caps, rest = [], text
            while rest:
                seq = next((k for k in KEY_NAMES if rest.startswith(k)), rest[0])
                caps.append(KEY_NAMES.get(seq, seq))
                rest = rest[len(seq):]
            out += [(t, c) for c in caps]
        elif text.endswith("\n"):  # a typed line: one cap, as it is sent
            out.append((t, text.rstrip("\n") + "  ⏎"))
        else:  # keys: termrec logs after typing, char i went out at t - 0.06 * (n - i)
            out += [(t - 0.06 * (len(text) - i), KEY_NAMES.get(c, c))
                    for i, c in enumerate(text)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cast")
    ap.add_argument("out")
    ap.add_argument("--keys", help="termrec's log: show the keys it typed")
    ap.add_argument("--title", default="qwenputer")
    ap.add_argument("--max-gap", type=float, default=0.25)
    ap.add_argument("--hold", type=float, default=3.0,
                    help="seconds a game frame stays up")
    ap.add_argument("--key-show", type=float, default=1.6, help="seconds a key cap stays up")
    ap.add_argument("--start-speed", type=float, default=1.0,
                    help="speed of the opening (until the game is up), outside fast-forwards")
    ap.add_argument("--ff", type=float, default=4.0,
                    help="video seconds for each fast-forwarded stretch (0: cut waits instead)")
    ap.add_argument("--ff-loading", type=float, default=2.0,
                    help="video seconds for the loading stretches (the model, the game's boot)")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    lines = open(args.cast).read().splitlines()
    head = json.loads(lines[0])
    events = [(t, "o", data) for t, kind, data in map(json.loads, lines[1:]) if kind == "o"]
    events = mend_cut_blocks(events)
    events += [(t, "k", cap) for t, cap in (key_events(args.keys) if args.keys else [])]
    events.sort(key=lambda e: e[0])

    # the video timeline: each event's video time. Inside a busy stretch time runs faster by
    # the stretch's factor (its length over --ff); elsewhere waits are cut and frames held.
    spans = busy_spans(events) if args.ff > 0 else []
    factor = [max(1.0, (b - a) / (args.ff_loading if k < 2 else args.ff))
              for k, (a, b) in enumerate(spans)]
    vt, v, held = [], 0.0, False
    ff = [None] * len(events)  # the factor of the gap after event i (None: real time)
    opening = max((b for a, b in spans[:2]), default=0.0)  # the game is up: loading + boot
    for i, (t, kind, data) in enumerate(events):
        if i:
            gap = t - events[i - 1][0]
            sp = next((k for k, (a, b) in enumerate(spans) if a <= events[i - 1][0] and t <= b),
                      None)
            if sp is not None and factor[sp] > 1.0:
                ff[i - 1] = factor[sp]
                v += gap / factor[sp]
            else:
                step = max(min(gap, args.max_gap), args.hold if held else 0.0) if gap > 0.05 else gap
                if t <= opening and args.start_speed > 1.0:
                    ff[i - 1] = args.start_speed
                    step = gap / args.start_speed if gap <= args.max_gap else step
                v += step
        vt.append(v)
        held = kind == "o" and data.count("▀") >= 400  # a game frame was drawn: hold it
    end = v + 3.0

    screen = Screen(head["width"], head["height"])
    painter = Painter(head["width"], head["height"], args.scale, args.title)
    shown: list[tuple[float, str]] = []  # (video time, key cap)
    tmp = Path(tempfile.mkdtemp(prefix="castvideo-"))
    concat, n_img, prev_key, i = [], 0, None, 0
    step = 1.0 / args.fps
    frame_t = 0.0
    real = 0.0
    while frame_t < end:
        changed = False
        while i < len(events) and vt[i] <= frame_t:
            t, kind, data = events[i]
            real = t
            if kind == "o":
                screen.feed(data)
            else:
                shown.append((vt[i], data))
            changed = True
            i += 1
        caps = [c for at, c in shown if frame_t - at < args.key_show]
        badge = ""
        if 0 < i < len(events) and ff[i - 1]:  # fast-forwarding: the clock runs with it
            real = min(events[i][0], events[i - 1][0] + (frame_t - vt[i - 1]) * ff[i - 1])
            badge = f"▶▶ {ff[i - 1]:.0f}× fast-forward"
        key = (tuple(caps), int(real), badge)
        if changed or key != prev_key or not concat:
            h, rem = divmod(int(real), 3600)
            clock = f"real time {h:d}:{rem // 60:02d}:{rem % 60:02d}"
            img = painter.frame(screen, clock, caps, badge)
            path = tmp / f"{n_img:06d}.png"
            img.save(path, compress_level=1)
            n_img += 1
            concat.append([path, step])
            prev_key = key
        else:
            concat[-1][1] += step
        frame_t += step
    with open(tmp / "list.ffconcat", "w") as f:
        f.write("ffconcat version 1.0\n")
        for path, dur in concat:
            f.write(f"file '{path}'\nduration {dur:.4f}\n")
        f.write(f"file '{concat[-1][0]}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i",
                    str(tmp / "list.ffconcat"), "-vf", f"fps={args.fps}", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
                    "-movflags", "+faststart", args.out], check=True)
    print(f"{args.out}: {end:.0f} s video of {events[-1][0]:.0f} s recorded, {n_img} images")


if __name__ == "__main__":
    main()
