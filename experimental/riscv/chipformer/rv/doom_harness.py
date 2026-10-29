"""Doom in the raw neural harness: Qwen plays through a ``doom`` tool whose game runs as RV32
guest firmware (env/doom) inside the compiled circuit layers of a "doom"-map checkpoint.

Qwen writes the tool call; the circuit captures it, runs Doom's tics and renderer as guest
instructions, and emits the frame as display tokens (DSP) that only the terminal reads, then a
one-line <tool_response> that Qwen reads. The chip layers run on the Triton executor (the
circuit's own tensor operations in fp32, fused across independent table sublayers, with a
device token loop); no host code executes guest instructions.
"""

import codecs
import os
from pathlib import Path

FIRMWARE = Path(__file__).parent / 'env/doom/doom.elf'
MODEL = os.environ.get('QWENPUTER_DOOM_MODEL', 'dist/q4-doom')
ROWS = 21  # qdoom.c: the frame's view rows (a status line follows)
TOOL = {'type': 'function', 'function': {
    'name': 'doom', 'description': 'Play DOOM (shareware E1M1), which runs inside your neural '
    'CPU. The first call starts the game. Each call presses keys and returns the player status; '
    'the user sees the frame. Keys: w forward, s back, a turn left, d turn right, q / e strafe, '
    'f fire, u use (open doors), enter, esc, 1-7 weapons, wait. Each key is held 4 tics; '
    'w*3 holds it 3 times as long. Example: w*4 d f',
    'parameters': {'type': 'object', 'properties': {'keys': {'type': 'string',
        'description': 'Space-separated keys, e.g. "w*4 d f". Empty to just look.'}},
        'required': ['keys']}}}
SYSTEM = ('You are Qwenputer. DOOM runs inside your own RISC-V neural layers, and you play it '
          'with the doom tool. When the user asks to start or play DOOM, call doom (keys "" '
          'starts the game). Turn requests such as "go forward" or "shoot" into keys. Keep '
          'calls short. After each call, report the status line in a few words; the user '
          'already sees the frame. Never invent game state.')


COLS = 80  # qdoom.c: cells per frame row


def halfblocks(cells: bytes) -> list[str]:
    """A color frame's cells (top, bottom xterm color index per cell) -> one line per row:
    upper half blocks, the foreground the top pixel and the background the bottom one. Only
    numbers go into the escape codes, so the result is safe to write to a terminal as is."""
    rows = []
    for r in range(ROWS):
        out, last = [], None
        for c in range(COLS):
            top, bot = cells[2 * (r * COLS + c)], cells[2 * (r * COLS + c) + 1]
            if (top, bot) != last:
                out.append(f'\x1b[38;5;{top};48;5;{bot}m')
                last = (top, bot)
            out.append('▀')
        rows.append(''.join(out) + '\x1b[0m')
    return rows


class Frames:
    """Presentation only: DSP bytes -> console lines and frames. A text frame (12) is ROWS
    lines and a status line; a color frame (14) is ROWS x COLS cells of two color bytes (any
    value, so read by count), then the status line. With ``draw`` a color frame goes there as
    (rows of half blocks with their escape codes, status) and with ``console`` the guest's
    console lines go there; otherwise both are written as scrollback text."""

    def __init__(self, write, draw=None, console=None):
        self.write, self.draw, self.console = write, draw, console
        self.line, self.frame, self.cells = bytearray(), None, None
        self.shown = 0  # frames written
        self.decode = codecs.getincrementaldecoder('latin-1')()

    def feed(self, byte: int) -> None:
        if self.cells is not None and len(self.cells) < 2 * ROWS * COLS:
            self.cells.append(byte)
            return
        if byte in (12, 14) and self.cells is None:  # a frame starts
            self.flush()
            self.frame = []
            self.cells = bytearray() if byte == 14 else None
            return
        if byte != 10:
            self.line.append(byte)
            return
        text, self.line = self.decode.decode(bytes(self.line)), bytearray()
        if self.frame is None:
            (self.console or (lambda line: self.write('│ ' + line + '\n')))(text)
            return
        if self.cells is not None:  # the color frame's status line
            rows = halfblocks(bytes(self.cells))
            self.cells, self.frame = None, None
            if self.draw:
                self.draw(rows, text)
            else:
                self.write(''.join('│ ' + row + '\n' for row in rows) + '│ » ' + text + '\n')
            self.shown += 1
            return
        self.frame.append(text)
        if len(self.frame) == ROWS + 1:
            body = ''.join('│ ' + row + '\n' for row in self.frame[:ROWS])
            self.write(body + '│ » ' + self.frame[ROWS] + '\n')
            self.frame = None
            self.shown += 1

    def flush(self) -> None:
        if self.line:
            self.feed(10)


class DoomSession:
    description = ('Qwenputer DOOM · E1M1 runs as RV32 guest code inside the circuit layers · '
                   'raw mode')
    reports_progress = True
    game_title = "DOOM · E1M1 · every frame computed by RV32 guest code in Qwen's circuit layers"
    draws_frames = True  # frames come as on_frame(rows, status), guest console as on_console

    def __init__(self, model=MODEL):
        import platform
        if platform.system() == 'Darwin':
            from chipformer.rv.neural_harness import guard_full_model
            guard_full_model()
        if os.environ.get('QWENPUTER_RV_MAP') != 'doom':
            raise RuntimeError('Set QWENPUTER_RV_MAP=doom (the Doom firmware needs that map).')
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('DOOM needs the remote CUDA GPU (the Triton executor).')
        from transformers.utils import logging as hf_logging

        from chipformer.rv.chat import RvChat
        hf_logging.set_verbosity_error()  # kernel-fallback notices would land in the terminal
        path = Path(model)
        if not (path / 'chipformer.json').is_file():
            raise ValueError(f'{path}: a merged doom-map checkpoint (chipformer.rv.merge) is '
                             'required')
        self.chat = RvChat(path, device='cuda', fast=False, executor=True, fuse=True)
        self.reset()

    def reset(self):
        from chipformer.rv import elf
        self.state = self.chat.boot(elf.program(FIRMWARE))
        self.messages = [{'role': 'system', 'content': SYSTEM}]
        self.playing = False  # a frame has been shown: the terminal may switch to play mode

    def _prompt(self):
        """The conversation's prompt ids; the oldest exchanges drop out past the input limit
        (the game itself lives in the guest's state, not in the conversation)."""
        while True:
            ids = self.chat.prompt_ids(self.messages, tools=[TOOL])
            if len(ids) <= 6144 or len(self.messages) <= 2:
                return ids
            del self.messages[1:3]

    def send(self, message, on_text, stop=None, on_progress=None, on_frame=None,
             on_console=None):
        """A turn of conversation: Qwen answers, and calls doom when it decides to."""
        self.messages.append({'role': 'user', 'content': message})
        return self._turn(on_text, stop, on_progress, on_frame=on_frame, on_console=on_console)

    def press(self, keys, on_text, stop=None, on_progress=None, on_frame=None, on_console=None):
        """Keys typed in play mode: the doom call is written into the conversation as the
        assistant's tool call (in place of Qwen choosing one), the circuit captures and runs it
        exactly as it runs Qwen's calls, and the turn ends when the circuit gives the position
        back (Qwen does not comment on every move)."""
        self.messages.append({'role': 'user', 'content': f'(pressed keys: {keys})'})
        call = ('<tool_call>\n<function=doom>\n<parameter=keys>\n' + keys +
                '\n</parameter>\n</function>\n</tool_call>')
        force = self.chat.tok.encode(call, add_special_tokens=False)
        return self._turn(on_text, stop, on_progress, force=force, on_frame=on_frame,
                          on_console=on_console)

    def _turn(self, on_text, stop, on_progress, force=None, on_frame=None, on_console=None):
        import time

        from chipformer.rv import spec as S
        from chipformer.rv.harness_display import Transcript
        display = Transcript(on_text)
        frames = Frames(on_text, on_frame, on_console)
        try:
            ids = self._prompt()
        except Exception:
            self.messages.pop()
            raise
        if len(ids) > 6144:
            self.messages.pop()
            raise ValueError('Conversation reached its input limit. Use /new for a fresh game.')
        decoder = codecs.getincrementaldecoder('utf-8')('replace')
        v0, chat = self.chat.v0, self.chat
        dsp0, dsp_n = S.FAMILIES['DSP']
        count, started = [0], time.perf_counter()

        def token(index):
            if index >= v0:
                count[0] += 1
                c = index - v0
                if dsp0 <= c < dsp0 + dsp_n:
                    frames.feed(c - dsp0)
                elif on_progress is not None and count[0] % 65536 == 0:
                    rate = count[0] / max(time.perf_counter() - started, 1e-9)
                    on_progress(f'… {count[0] / 1e6:.2f}M circuit tokens · {rate:,.0f} tok/s')
                return
            if index not in chat.eos:
                text = decoder.decode(chat.T.bytes[index])
                if text:
                    display.feed(text)
        try:
            result = chat.generate(ids, self.state, max_new=512, max_chip=40_000_000,
                                   on_token=token, stop=stop, force=force,
                                   until_released=force is not None)
            frames.flush()
            if result['circuit_paused']:
                raise RuntimeError('Guest trace budget exhausted; last parked state kept.')
            if stop is not None and stop.is_set():
                self.messages.pop()
                return {**result, 'stopped': True}
            self.state = chat.last_firmware_state
            self.playing = self.playing or frames.shown > 0
            text = chat.text_of([i for i in result['tokens'] if i not in chat.eos])
            self.messages.append({'role': 'assistant', 'content': text})
            return result
        except Exception:
            self.messages.pop()
            raise
        finally:
            display.finish()
