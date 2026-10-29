"""fx-style scrollback UI: raw input, streamed output, no alternate screen.

The one exception is a game (a session with press(), e.g. DOOM): once its guest starts drawing,
the game gets its own screen (the alternate screen, redrawn in place) and keys go to it; Tab
goes back to the chat's scrollback, untouched, and Tab again returns to the game."""

import codecs
from collections import deque
from contextlib import contextmanager
import os
import queue
import select
import shutil
import sys
import termios
import threading
import time
import tty


# play mode (a session with press(), e.g. DOOM): a key -> the word the session's game reads
PLAY_KEYS = {**{c: c for c in 'wasdqefu1234567'}, ' ': 'f', '\r': 'enter', '\n': 'enter'}
ARROWS = {'A': 'w', 'B': 's', 'C': 'd', 'D': 'a'}
PLAY_HINT = 'wasd/arrows move · q e strafe · space fire · u use · esc menu · tab chat · ^C stop'
SETTLE = 0.35  # seconds without a key before the keys typed so far go to the game
FRAME_ROWS = 21  # doom_harness.ROWS: cell rows of a frame (80 cells wide)


def batch(words):
    """['w', 'w', 'd'] -> 'w*2 d': repeats become one held key."""
    out = []
    for w in words:
        if out and out[-1][0] == w:
            out[-1][1] += 1
        else:
            out.append([w, 1])
    return ' '.join(w if n == 1 else f'{w}*{n}' for w, n in out)


def safe(text):
    return ''.join(c if c in '\n\t' or (ord(c) >= 32 and not 127 <= ord(c) < 160)
                   else '' for c in str(text))


class GameScreen:
    """The game's own screen, redrawn in place: a title, the frame (or the guest's console
    while it boots), the game's status line, what the circuit is doing, Qwen's last words,
    the keys typed and the key help. Frame rows come from the session with their color codes
    (made from numbers only); every other text goes through safe()."""

    def __init__(self, write, title):
        self.out, self.title, self.buf = write, title, []
        self.rows, self.status, self.console = [], '', deque(maxlen=FRAME_ROWS)
        self.activity, self.message, self.keys = '', '', ''
        self.shown = False

    def write(self, text):
        self.buf.append(text)

    def flush(self):
        """One write per redraw: the terminal shows a whole frame at once."""
        if self.buf:
            text, self.buf = ''.join(self.buf), []
            self.out(text)

    def _at(self, row, text):
        self.write(f'\x1b[{row};1H\x1b[2K' + text)

    def _left(self):
        cols = shutil.get_terminal_size((100, 40)).columns
        return ' ' * max(0, (cols - 80) // 2), cols

    def enter(self):
        self.shown = True
        self.write('\x1b[?1049h\x1b[?25l\x1b[2J')
        self.redraw()
        self.flush()

    def leave(self):
        self.shown = False
        self.write('\x1b[2J\x1b[?25h\x1b[?1049l')
        self.flush()

    def redraw(self):
        if not self.shown:
            return
        pad, cols = self._left()
        self._at(1, pad + '\x1b[1m' + safe(self.title)[:cols - len(pad)] + '\x1b[0m')
        self._frame()
        self._lines()

    def _frame(self):
        pad, cols = self._left()
        if self.rows:
            for i, row in enumerate(self.rows):
                self._at(3 + i, pad + row)
        else:  # booting: the guest's console
            lines = list(self.console)
            for i in range(FRAME_ROWS):
                text = safe(lines[i]) if i < len(lines) else ''
                self._at(3 + i, pad + '\x1b[2m' + text[:cols - len(pad)] + '\x1b[0m')

    def _lines(self):
        pad, cols = self._left()
        room = max(10, cols - len(pad))
        row = 4 + FRAME_ROWS
        self._at(row, pad + '\x1b[1m' + safe(self.status)[:room] + '\x1b[0m')
        self._at(row + 1, pad + '\x1b[33m' + safe(self.activity)[:room] + '\x1b[0m')
        self._at(row + 2, pad + safe(self.message)[:room])
        self._at(row + 4, pad + '\x1b[33m❯\x1b[0m ' + safe(self.keys)[:room - 2])
        self._at(row + 5, pad + '\x1b[2m' + PLAY_HINT[:room] + '\x1b[0m')

    def frame(self, rows, status):
        self.rows, self.status = rows, status
        if self.shown:
            self._frame()
            self._lines()
            self.flush()

    def add_console(self, line):
        self.console.append(line)
        if self.shown and not self.rows:
            self._frame()
            self.flush()

    def set(self, **lines):
        for name, value in lines.items():
            setattr(self, name, value)
        if self.shown:
            self._lines()
            self.flush()


@contextmanager
def raw_mode(stream):
    if not stream.isatty():
        raise ValueError('Interactive chat requires a terminal. Use --ask for a single request.')
    fd = stream.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, previous)


class RawTerminal:
    def __init__(self, session, *, stdin=None, stdout=None):
        self.session = session
        self.stdin, self.stdout = stdin or sys.stdin, stdout or sys.stdout
        self.events = queue.Queue()
        self.pending, self.history = deque(), []
        self.line, self.cursor, self.history_at = '', 0, 0
        self.stop = threading.Event()
        self.busy, self.closing = False, False
        self.worker = None
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.escape, self.pasting = '', False
        self.streaming = False
        self.progress = False  # a progress line is on screen (overwritten by the next output)
        self.play, self.keys, self.key_at = False, [], 0.0  # in the game: keys go to it
        self.game = GameScreen(self.write, getattr(session, 'game_title', 'game'))
        self.game_text = ''  # Qwen's text of the turn, while the game screen is up
        self.moves = 0
        self.left_game = False  # Tab took the user out: frames no longer bring the game up

    def write(self, text):
        self.stdout.write(text.replace('\n', '\r\n'))
        self.stdout.flush()

    def prompt(self):
        if self.play:
            self.game.set(keys=' '.join(self.keys))
            return
        width = max(12, shutil.get_terminal_size((80, 24)).columns - 4)
        left = max(0, self.cursor - width + 1)
        visible = self.line[left:left + width].replace('\n', '↵')
        self.write('\r\x1b[2K\x1b[36m❯\x1b[0m ' + safe(visible))
        after = len(visible) - (self.cursor - left)
        if after > 0:
            self.write(f'\x1b[{after}D')

    def output(self, text):
        if self.streaming:
            self.write('\n')
            self.streaming = False
        self.write('\r\x1b[2K' + safe(text))
        if not str(text).endswith('\n'):
            self.write('\n')
        self.prompt()

    def start(self, message, press=False):
        self.busy = True
        self.stop.clear()
        send = self.session.press if press else self.session.send

        def work():
            extra = {}
            if getattr(self.session, 'reports_progress', False):
                extra['on_progress'] = lambda text: self.events.put(('progress', text))
            if getattr(self.session, 'draws_frames', False):
                extra['on_frame'] = lambda rows, status: self.events.put(('frame', (rows, status)))
                extra['on_console'] = lambda line: self.events.put(('console', line))
            try:
                result = send(message, lambda text: self.events.put(('text', text)),
                              self.stop, **extra)
                self.events.put(('done', result))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def submit(self):
        message = self.line.strip()
        self.line, self.cursor = '', 0
        if not message:
            return
        self.history.append(message)
        self.history_at = len(self.history)
        if self.streaming:
            self.write('\n')
            self.streaming = False
        self.write('\r\x1b[2K\x1b[36m❯\x1b[0m ' + safe(message) + '\n\n')
        if message in ('/quit', '/exit'):
            self.closing = True
            self.stop.set()
        elif message == '/help':
            self.output('/help  /status  /new  /quit\nEnter sends · arrows edit/history · Ctrl-C stops · Ctrl-D exits'
                        + ('\nTab: play mode (keys go to the game)' if hasattr(self.session, 'press') else ''))
        elif message == '/status':
            self.output(self.session.description)
        elif message == '/new':
            if self.busy:
                self.output('Stop the current turn before starting a new session.')
            else:
                self.output('Booting a fresh neural guest…')
                self.session.reset()
                self.game = GameScreen(self.write, getattr(self.session, 'game_title', 'game'))
                self.moves, self.left_game = 0, False
                self.output('Guest ready. Dictionary and cells cleared.')
        elif self.busy:
            self.pending.append(message)
            self.output('Queued for the next turn.')
        else:
            self.write('\x1b[2m● working in the neural circuit…\x1b[0m\n')
            self.start(message)
            self.prompt()

    def can_play(self):
        return hasattr(self.session, 'press') and getattr(self.session, 'playing', False)

    def set_play(self, on):
        """Into the game's screen, or back to the chat's scrollback (with a one-line note)."""
        if on == self.play:
            return
        self.play, self.keys, self.left_game = on, [], not on
        if on:
            if self.streaming:
                self.write('\n')
                self.streaming = False
            self.write('\r\x1b[2K')
            self.game.enter()
            return
        self.game.leave()
        note = f'◆ game · {self.moves} moves · {self.game.status or "starting"}'
        self.output(note + '  (Tab returns to the game)')

    def play_key(self, char):
        """A key in play mode; True when it was taken."""
        if self.escape:
            if len(self.escape) == 2 and char in ARROWS and self.escape == '\x1b[':
                self.escape = ''
                self.keys.append(ARROWS[char])
                self.key_at = time.monotonic()
                return True
            return False
        if char == '\t':
            self.set_play(False)
        elif char == '\x04':
            self.set_play(False)
            self.closing = True
            self.stop.set()
        elif char == '\x03':
            self.keys = []
            if self.busy:
                self.stop.set()
                self.game.set(activity='stopping after the current guest call…')
        elif char == '/':
            self.set_play(False)
            self.line, self.cursor = '/', 1
        elif char.lower() in PLAY_KEYS or char in PLAY_KEYS:
            self.keys.append(PLAY_KEYS.get(char, PLAY_KEYS.get(char.lower())))
            self.key_at = time.monotonic()
        elif char in ('\x7f', '\x08'):
            if self.keys:
                self.keys.pop()
        else:
            return False
        return True

    def game_event(self, kind, value):
        """An event of the running turn while the game has the screen."""
        g = self.game
        if kind == 'frame':
            g.frame(*value)
        elif kind == 'console':
            g.add_console(value)
            g.set(activity='▸ booting the game in the circuit…')
        elif kind == 'progress':
            g.set(activity='▸ ' + value)
        elif kind == 'text':
            self.game_text += value
        else:  # done / error: the turn is over
            words = [ln.strip() for ln in self.game_text.splitlines()
                     if ln.strip() and ln.strip()[0] not in '┌│└●']
            self.game_text = ''
            if kind == 'error':
                g.set(activity='\x1b[31m' + safe(value), message='')
            else:
                steps = value.get('neural_steps', value.get('chip_tokens', 0))
                stopped = value.get('stopped')
                done = f'move {self.moves} done' if self.moves else 'game started'
                g.set(activity=('stopped · prior guest state kept' if stopped else
                                f'✓ {done} · {steps:,} circuit steps · ready for keys'),
                      message=('Qwen: ' + words[-1]) if words else g.message)
            self.busy = False
            if self.pending and not self.closing:
                self.start(self.pending.popleft())

    def send_keys(self):
        """Sends the keys typed so far once they settle and the game is free."""
        if (self.play and self.keys and not self.busy and not self.closing
                and time.monotonic() - self.key_at >= SETTLE):
            keys, self.keys = batch(self.keys), []
            self.moves += 1
            self.game.set(activity=f'▸ move {self.moves}: {keys} · running in the circuit…',
                          keys='')
            self.start(keys, press=True)

    def key(self, char):
        if self.play and self.play_key(char):
            return
        if char == '\t' and not self.play and not self.line and self.can_play():
            self.set_play(True)
            return
        if self.escape:
            self.escape += char
            if len(self.escape) > 12:
                self.escape = ''
                return
            sequence = self.escape
            if sequence == '\x1b[200~':
                self.pasting, self.escape = True, ''
            elif sequence == '\x1b[201~':
                self.pasting, self.escape = False, ''
            elif sequence in ('\x1b[A', '\x1b[B'):
                offset = -1 if char == 'A' else 1
                self.history_at = min(len(self.history), max(0, self.history_at + offset))
                self.line = self.history[self.history_at] if self.history_at < len(self.history) else ''
                self.cursor, self.escape = len(self.line), ''
            elif sequence in ('\x1b[C', '\x1b[D'):
                self.cursor = min(len(self.line), max(0, self.cursor + (1 if char == 'C' else -1)))
                self.escape = ''
            elif sequence in ('\x1b[H', '\x1b[F'):
                self.cursor, self.escape = (0 if char == 'H' else len(self.line)), ''
            elif char.isalpha() and len(sequence) > 2:
                self.escape = ''
            return
        if char == '\x1b':
            self.escape = char
        elif char == '\x03':
            self.pending.clear()
            if self.busy:
                self.stop.set()
                self.output('Stopping after the current neural guest call…')
            else:
                self.line, self.cursor = '', 0
                self.output('^C')
        elif char == '\x04':
            if not self.line:
                self.closing = True
                self.stop.set()
        elif char in ('\r', '\n'):
            if self.pasting:
                self.line = self.line[:self.cursor] + ' ' + self.line[self.cursor:]
                self.cursor += 1
            else:
                self.submit()
        elif char in ('\x7f', '\x08'):
            if self.cursor:
                self.line = self.line[:self.cursor - 1] + self.line[self.cursor:]
                self.cursor -= 1
        elif char == '\x01':
            self.cursor = 0
        elif char == '\x05':
            self.cursor = len(self.line)
        elif ord(char) >= 32 and len(self.line) < 4096:
            self.line = self.line[:self.cursor] + char + self.line[self.cursor:]
            self.cursor += 1

    def run(self):
        with raw_mode(self.stdin):
            self.write('\x1b[?2004h\x1b[1mqwenputer\x1b[0m  \x1b[2mneural harness\x1b[0m\n')
            self.write('\x1b[2m' + safe(self.session.description) + '\x1b[0m\n')
            self.write('\x1b[2m/help for commands · Ctrl-D to exit\x1b[0m\n\n')
            self.prompt()
            try:
                while not self.closing or self.busy:
                    while not self.events.empty():
                        kind, value = self.events.get()
                        if (kind in ('console', 'frame') and not self.play and not self.left_game
                                and hasattr(self.session, 'press')):
                            self.set_play(True)  # the guest started drawing: its screen
                        if self.play:
                            self.game_event(kind, value)
                            continue
                        if self.progress:
                            self.write('\r\x1b[2K')
                            self.progress = False
                        if kind == 'progress':
                            self.write('\r\x1b[2K\x1b[2m' + safe(value) + '\x1b[0m')
                            self.progress = True
                        elif kind in ('frame', 'console'):  # (back in the chat meanwhile)
                            self.game_event(kind, value)
                        elif kind == 'text':
                            if not self.streaming:
                                self.write('\r\x1b[2K\x1b[1m●\x1b[0m ')
                                self.streaming = True
                            self.write(safe(value))
                        else:
                            if self.streaming:
                                self.write('\n')
                            if kind == 'error':
                                self.write('\x1b[31m' + safe(value) + '\x1b[0m\n')
                            elif isinstance(value, dict):
                                details = value.get('neural_steps', value.get('chip_tokens', 0))
                                status = ('stopped · prior guest state restored' if value.get('stopped')
                                          else f'✓ {details} neural steps')
                                self.write(f'\x1b[2m{status}\x1b[0m\n')
                            self.busy, self.streaming = False, False
                            self.prompt()
                            if self.pending and not self.closing:
                                self.start(self.pending.popleft())
                    if self.closing and not self.busy:
                        break
                    ready, _, _ = select.select([self.stdin.fileno()], [], [], 0.05)
                    if ready:
                        data = os.read(self.stdin.fileno(), 4096)
                        if not data:
                            self.closing = True
                            self.stop.set()
                        for char in self.decoder.decode(data):
                            self.key(char)
                        if self.play and self.escape == '\x1b':  # a lone Esc: the game's menu
                            self.escape = ''
                            self.keys.append('esc')
                            self.key_at = time.monotonic()
                        if not self.streaming:
                            self.prompt()
                    self.send_keys()
            finally:
                self.stop.set()
                if self.play:
                    self.game.leave()
                self.write('\r\x1b[2K\x1b[?2004l\x1b[0m\n')
