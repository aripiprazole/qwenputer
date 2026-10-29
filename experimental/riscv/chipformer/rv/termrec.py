"""Record a terminal program into an asciicast v2 file, typing scripted input (stdlib only).

  python -m chipformer.rv.termrec OUT.cast --script steps.json -- command args...

steps.json: a list of {"wait_for": "text", "timeout": s} (wait until output after this step
began contains it), {"type": "text", "delay": s} (type it, one character per 60 ms, then "\\r"
if the text ends with a newline) and {"send": "bytes", "delay": s} (written at once, as a
terminal sends an arrow key). Output is recorded as the program writes it, with real timestamps.
"""

import argparse
import codecs
import json
import os
import pty
import select
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--script", required=True)
    ap.add_argument("--cols", type=int, default=100)
    ap.add_argument("--rows", type=int, default=36)
    argv = sys.argv[1:]
    if "--" not in argv:
        ap.error("give the command after --")
    cut = argv.index("--")
    args, cmd = ap.parse_args(argv[:cut]), argv[cut + 1:]
    steps = json.load(open(args.script))

    pid, fd = pty.fork()
    if pid == 0:
        os.environ.update(COLUMNS=str(args.cols), LINES=str(args.rows), TERM="xterm-256color")
        os.execvp(cmd[0], cmd)
    import fcntl
    import struct
    import termios
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", args.rows, args.cols, 0, 0))

    t0 = time.time()
    seen = ""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    out = open(args.out, "w")
    out.write(json.dumps({"version": 2, "width": args.cols, "height": args.rows,
                          "timestamp": int(t0), "env": {"TERM": "xterm-256color"}}) + "\n")

    def pump(timeout: float) -> bool:
        nonlocal seen
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            return True
        try:
            data = os.read(fd, 65536)
        except OSError:
            return False
        if not data:
            return False
        text = decoder.decode(data)  # (a character cut between reads stays whole)
        seen = (seen + text)[-200000:]
        out.write(json.dumps([round(time.time() - t0, 3), "o", text]) + "\n")
        out.flush()
        return True

    alive = True
    for step in steps:
        if not alive:
            break
        if "wait_for" in step:
            mark, end = len(seen), time.time() + step.get("timeout", 600)
            while alive and step["wait_for"] not in seen[mark:] and time.time() < end:
                alive = pump(0.2)
            print(f"[termrec] {time.time() - t0:8.1f}s waited for {step['wait_for']!r}",
                  file=sys.stderr, flush=True)
        if "send" in step:
            end = time.time() + step.get("delay", 0.5)
            while alive and time.time() < end:
                alive = pump(0.05)
            os.write(fd, step["send"].encode())
            print(f"[termrec] {time.time() - t0:8.1f}s sent {step['send']!r}", file=sys.stderr,
                  flush=True)
        if "type" in step:
            end = time.time() + step.get("delay", 0.5)
            while alive and time.time() < end:
                alive = pump(0.05)
            text = step["type"]
            for ch in text.rstrip("\n"):
                os.write(fd, ch.encode())
                end = time.time() + 0.06
                while alive and time.time() < end:
                    alive = pump(0.02)
            if text.endswith("\n"):
                os.write(fd, b"\r")
            print(f"[termrec] {time.time() - t0:8.1f}s typed {text!r}", file=sys.stderr,
                  flush=True)
    end = time.time() + 3
    while alive and time.time() < end:
        alive = pump(0.1)
    out.close()


if __name__ == "__main__":
    main()
