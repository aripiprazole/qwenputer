"""Real PTYs validate raw input, scrollback output and terminal restoration."""

import os
from pathlib import Path
import select
import subprocess
import sys
import termios
import time

import pytest

from chipformer.rv.raw_terminal import safe


def test_sanitizes_terminal_escape_and_c1_controls():
    assert safe('hello\x1b[31m\x07\x9bworld\n') == 'hello[31mworld\n'


@pytest.mark.parametrize('outcome', ['complete', 'error', 'cancel'])
def test_raw_session_restores_terminal_and_uses_normal_scrollback(outcome):
    import pty
    master, slave = pty.openpty()
    before = termios.tcgetattr(slave)
    source = f'''
from chipformer.rv.raw_terminal import RawTerminal
class Session:
    description = "RAW TRANSPORT FIXTURE"
    def send(self, text, callback, stop):
        if {outcome!r} == 'error': raise RuntimeError("fixture failure")
        if {outcome!r} == 'cancel':
            callback("waiting for cancellation")
            if not stop.wait(5): raise RuntimeError("stop event was not delivered")
            return {{"stopped": True}}
        callback("answer:" + text)
        return {{"neural_steps": 7}}
    def reset(self): pass
RawTerminal(Session()).run()
'''
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen([sys.executable, '-B', '-c', source], cwd=root,
                               stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    output = bytearray()

    def until(marker):
        deadline = time.monotonic() + 10
        while marker not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                output.extend(os.read(master, 65536))
        assert marker in output, bytes(output)

    try:
        until(b'RAW TRANSPORT FIXTURE')
        assert termios.tcgetattr(slave) != before
        os.write(master, b'hello\r')
        if outcome == 'cancel':
            until(b'waiting for cancellation')
            os.write(master, b'\x03')
            until(b'stopped \xc2\xb7 prior guest state restored')
        else:
            until(b'fixture failure' if outcome == 'error' else b'answer:hello')
        os.write(master, b'\x04')
        process.wait(timeout=10)
        assert process.returncode == 0
        after = termios.tcgetattr(slave)
        # Darwin sets the transient PENDIN bit when canonical input is restored.
        after[3] &= ~getattr(termios, 'PENDIN', 0)
        before[3] &= ~getattr(termios, 'PENDIN', 0)
        assert after == before
        assert b'\x1b[?1049' not in output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)


def test_batches_repeated_play_keys():
    from chipformer.rv.raw_terminal import batch
    assert batch(['w', 'w', 'w', 'd', 'f', 'f']) == 'w*3 d f*2'
    assert batch(['esc']) == 'esc'


def test_game_takes_the_screen_and_keys_go_to_it_in_a_real_pty():
    """A fixture game session (no model): once its first turn draws, the game has its own
    screen (alternate screen, redrawn in place) and keys and arrows go to press() batched;
    Tab returns to the chat's scrollback with a summary; the terminal is restored on exit."""
    import pty
    master, slave = pty.openpty()
    source = '''
from chipformer.rv.raw_terminal import RawTerminal
class Session:
    description = "PLAY MODE FIXTURE"
    game_title = "FIXTURE GAME"
    reports_progress = False
    draws_frames = True
    playing = False
    def send(self, text, callback, stop, on_frame, on_console):
        callback("calling the game")
        on_console("booting the fixture")
        self.playing = True
        on_frame(["\\x1b[38;5;196;48;5;21m▀\\x1b[0m"] * 21, "alive health 100")
        callback("Game started.")
        return {"neural_steps": 3}
    def press(self, keys, callback, stop, on_frame, on_console):
        on_frame(["\\x1b[38;5;46;48;5;21m▀\\x1b[0m"] * 21, "pressed[" + keys + "]")
        return {"neural_steps": 5}
    def reset(self): self.playing = False
RawTerminal(Session()).run()
'''
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen([sys.executable, '-B', '-c', source], cwd=root,
                               stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    output = bytearray()

    def until(marker):
        deadline = time.monotonic() + 10
        while marker not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                output.extend(os.read(master, 65536))
        assert marker in output, bytes(output)

    try:
        until(b'PLAY MODE FIXTURE')
        os.write(master, b'start doom\r')
        until(b'\x1b[?1049h')  # the game's screen
        until(b'Qwen: Game started.')
        os.write(master, b'ww\x1b[C ')  # w w, right arrow (turn right), space (fire)
        until(b'pressed[w*2 d f]')
        os.write(master, b'\t')
        until(b'\x1b[?1049l')  # back to the chat
        until(b'1 moves')
        os.write(master, b'/quit\r')
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
        os.close(master)
        os.close(slave)
