from chipformer.rv.doom_harness import ROWS, Frames
from chipformer.rv.harness_display import Transcript


def test_frames_split_console_text_and_frames():
    out = []
    frames = Frames(out.append)
    for b in b"R_Init: Init DOOM refresh daemon\n":
        frames.feed(b)
    frames.feed(12)
    for r in range(ROWS):
        for b in (b"#" * 80 + b"\n"):
            frames.feed(b)
    for b in b"E1M1 alive health 100\n":
        frames.feed(b)
    assert out[0] == '│ R_Init: Init DOOM refresh daemon\n'
    frame = out[1].splitlines()
    assert len(frame) == ROWS + 1 and frame[0] == '│ ' + '#' * 80
    assert frame[-1] == '│ » E1M1 alive health 100'


def test_tool_header_names_the_function():
    out = []
    t = Transcript(out.append)
    t.feed('<tool_call>\n<function=doom>\n<parameter=keys>\nw*4 f\n</parameter>\n</function>\n'
           '</tool_call>')
    t.finish()
    assert '┌ doom · w*4 f' in ''.join(out)


def test_firmware_starts_and_moves_on_the_reference_model(monkeypatch):
    """The built firmware (a test oracle run; no circuit): a color frame of 21 x 80 cells (two
    xterm color bytes each) and the status line."""
    import importlib

    import pytest

    from chipformer.rv import doom_harness
    if not doom_harness.FIRMWARE.is_file():
        pytest.skip('doom.elf is not built (chipformer/rv/env/doom/build.py)')
    monkeypatch.setenv('QWENPUTER_RV_MAP', 'doom')
    from chipformer.rv import spec
    importlib.reload(spec)
    try:
        from chipformer.rv import elf, ref
        from chipformer.rv import doomcheck
        importlib.reload(ref)
        importlib.reload(doomcheck)
        m = ref.Machine.from_program(elf.program(doom_harness.FIRMWARE))
        m.text = doomcheck.call_rows('')
        doomcheck.run(m, 10**6)
        for keys in ('', 'w*2 d'):
            m.text = doomcheck.call_rows(keys)
            r = doomcheck.run(m, 10**9)
            screen = r['screen'].encode('latin-1')
            frame = screen[screen.rindex(14) + 1:]
            cells, status = frame[:2 * ROWS * 80], frame[2 * ROWS * 80:]
            assert len(cells) == 2 * ROWS * 80 and all(c >= 16 for c in cells)
            assert status.startswith(b'E1M1 alive health 100')
            assert 'E1M1 alive health 100' in r['out']
    finally:
        monkeypatch.delenv('QWENPUTER_RV_MAP')
        importlib.reload(spec)
        importlib.reload(ref)
        importlib.reload(doomcheck)


def test_color_frames_are_read_by_count_and_drawn_as_half_blocks():
    from chipformer.rv.doom_harness import COLS
    text, drawn = [], []
    cells = [16 + (i % 216) for i in range(2 * ROWS * COLS)]
    cells[0], cells[1] = 10, 12  # cell bytes may look like controls: read by count
    frames = Frames(text.append, lambda rows, status: drawn.append((rows, status)))
    for b in [14, *cells, *b'E1M1 alive health 100\n']:
        frames.feed(b)
    (rows, status), = drawn
    assert len(rows) == ROWS and frames.shown == 1 and status == 'E1M1 alive health 100'
    assert rows[0].startswith('\x1b[38;5;10;48;5;12m▀')
    assert rows[0].count('▀') == COLS and rows[-1].endswith('\x1b[0m')
    assert text == []
