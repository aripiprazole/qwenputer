"""Render an asciicast v2 recording (termrec.py) as a GIF with Pillow.

  python -m chipformer.rv.castgif IN.cast OUT.gif [--max-gap 1.5] [--hold 4] [--font PATH]

A small terminal emulator for what the raw harness writes (text, CR/LF, backspace, ESC[nD,
ESC[2K, SGR colors, scrolling). Waits longer than --max-gap are shortened to it, and a bar at the
bottom shows the real elapsed time of every frame, so the speed-up is visible in the GIF.
"""

import argparse
import json
import re

from PIL import Image, ImageDraw, ImageFont

COLORS = {30: (40, 40, 40), 31: (230, 90, 90), 32: (120, 200, 120), 33: (220, 200, 100),
          34: (100, 150, 230), 35: (200, 120, 200), 36: (90, 200, 220), 37: (220, 220, 220)}
FG, BG, DIM = (215, 215, 215), (18, 18, 22), (130, 130, 140)
CSI = re.compile(r"\x1b\[([?0-9;]*)([A-Za-z])")


class Screen:
    def __init__(self, cols: int, rows: int) -> None:
        self.cols, self.rows = cols, rows
        self.lines = [[(" ", FG, False)] * cols for _ in range(rows)]
        self.x = self.y = 0
        self.fg, self.bold = FG, False

    def _newline(self) -> None:
        self.y += 1
        if self.y >= self.rows:
            self.lines.pop(0)
            self.lines.append([(" ", FG, False)] * self.cols)
            self.y = self.rows - 1

    def feed(self, text: str) -> None:
        i = 0
        while i < len(text):
            m = CSI.match(text, i)
            if m:
                self._csi(m[1], m[2])
                i = m.end()
                continue
            ch = text[i]
            i += 1
            if ch == "\r":
                self.x = 0
            elif ch == "\n":
                self._newline()
            elif ch == "\b":
                self.x = max(0, self.x - 1)
            elif ch == "\x1b" or ord(ch) < 32:
                continue
            else:
                if self.x >= self.cols:
                    self.x = 0
                    self._newline()
                self.lines[self.y][self.x] = (ch, self.fg, self.bold)
                self.x += 1

    def _csi(self, arg: str, op: str) -> None:
        if arg.startswith("?"):
            return
        nums = [int(n) for n in arg.split(";") if n] or [0]
        if op == "K":
            if nums[0] == 2:
                self.lines[self.y] = [(" ", FG, False)] * self.cols
            else:
                for x in range(self.x, self.cols):
                    self.lines[self.y][x] = (" ", FG, False)
        elif op == "D":
            self.x = max(0, self.x - (nums[0] or 1))
        elif op == "C":
            self.x = min(self.cols - 1, self.x + (nums[0] or 1))
        elif op == "m":
            for n in nums:
                if n == 0:
                    self.fg, self.bold = FG, False
                elif n == 1:
                    self.bold = True
                elif n == 2:
                    self.fg = DIM
                elif n in COLORS:
                    self.fg = COLORS[n]


def render(screen: Screen, font, cw: int, ch: int, caption: str) -> Image.Image:
    img = Image.new("RGB", (screen.cols * cw + 16, (screen.rows + 1) * ch + 20), BG)
    d = ImageDraw.Draw(img)
    for y, line in enumerate(screen.lines):
        run, start, style = "", 0, None
        for x, (c, fg, bold) in enumerate(line + [(" ", None, False)]):
            if (fg, bold) != style:
                if run.strip():
                    d.text((8 + start * cw, 8 + y * ch), run, font=font, fill=style[0])
                run, start, style = "", x, (fg, bold)
            run += c
    d.rectangle([0, img.height - ch - 8, img.width, img.height], fill=(38, 38, 48))
    d.text((8, img.height - ch - 4), caption, font=font, fill=(240, 200, 90))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cast")
    ap.add_argument("gif")
    ap.add_argument("--max-gap", type=float, default=1.5)
    ap.add_argument("--font", default="/System/Library/Fonts/Menlo.ttc")
    ap.add_argument("--size", type=int, default=14)
    ap.add_argument("--title", default="")
    ap.add_argument("--hold", type=float, default=4.0,
                    help="seconds a frame stays up when its output holds a game frame")
    args = ap.parse_args()
    lines = open(args.cast).read().splitlines()
    head = json.loads(lines[0])
    events = [json.loads(x) for x in lines[1:]]
    font = ImageFont.truetype(args.font, args.size)
    cw = round(font.getlength("M"))
    ch = args.size + 4
    screen = Screen(head["width"], head["height"])
    frames, durations, last, held = [], [], 0.0, False

    def stamp(t: float) -> str:
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        return f"{args.title}  real time {h:d}:{m:02d}:{s:02d}" + (
            "  (waits compressed)" if args.max_gap else "")

    for t, kind, data in events:
        if kind != "o":
            continue
        if frames and t - last < 0.05:  # coalesce bursts into one frame
            screen.feed(data)
            frames[-1] = render(screen, font, cw, ch, stamp(t))
            held = held or "\u2502 \u00bb " in data
            continue
        if frames:
            durations.append(max(min(t - last, args.max_gap), args.hold if held else 0))
        screen.feed(data)
        frames.append(render(screen, font, cw, ch, stamp(t)))
        last, held = t, "\u2502 \u00bb " in data
    durations.append(3.0)
    frames[0].save(args.gif, save_all=True, append_images=frames[1:], optimize=True, loop=0,
                   duration=[max(20, int(d * 1000)) for d in durations])
    print(f"{args.gif}: {len(frames)} frames, {sum(durations):.0f} s GIF of "
          f"{events[-1][0]:.0f} s recorded")


if __name__ == "__main__":
    main()
