"""Bounded offline neural-circuit fixture. No pretrained Qwen model is loaded.

The sparse PyTorch tensors evaluate the compiled neural circuit. The guest C
parser and dictionary run as RISC-V instructions in that circuit, never in Python.
"""

from pathlib import Path
import time


FIRMWARE = Path(__file__).parent / 'env/harness/harness.elf'


class ByteText:
    printable = list(range(260))
    TCO, TCC, NL = 260, 261, 10

    def __init__(self):
        from chipformer.rv.bridge import SPECIALS
        self.bytes = [bytes([i]) for i in range(256)] + [s.encode() for s in SPECIALS]
        self.bytes += [b'<tool_call>', b'</tool_call>']

    def printable_index(self, token):
        return token if 0 <= token < 260 else None

    def row_bytes(self, token):
        return self.bytes[token][:16].ljust(16, b'\0')

    def call(self, command):
        body = f'<function=harness><parameter=command>{command}</parameter></function>'
        return [self.TCO, *body.encode(), self.TCC]


class NeuralFixture:
    def __init__(self, max_steps=30000):
        import torch
        from chipformer.rv import core, elf, ref
        from chipformer.rv.decode import CodeDecoder
        from chipformer.rv.fixture_tensors import SparseTensorCircuit
        torch.set_num_threads(1)
        self.torch, self.core = torch, core
        self.text, self.offset, self.max_steps = ByteText(), 262, max_steps
        self.circuit = core.build(text=self.text)
        core.snapshot(self.circuit, ref.Machine.from_program(elf.program(FIRMWARE)))
        self.executor = SparseTensorCircuit(self.circuit)
        self.decoder = CodeDecoder(self.offset, self.text)
        self.steps = 0
        from chipformer.rv import spec as S
        token = self.offset + S.tok('GO')
        for _ in range(max_steps):
            previous = token
            token = self.step(token)
            if previous == self.offset + S.tok('HALT'):
                if token is not None and token < self.offset:
                    self.step(token)
                break
            if token is None:
                raise RuntimeError('neural firmware boot lost ownership')
        else:
            raise RuntimeError('neural firmware boot exceeded its budget')

    def step(self, token):
        from chipformer.rv import spec as S
        stream = token - self.offset if token >= self.offset else S.TEXT_TOKEN | token
        with self.torch.no_grad():
            hidden = self.executor.step_token(stream)
            token = int(self.decoder(hidden[self.circuit.keep[:self.core.NFAM + self.core.NPAY]]))
        self.steps += 1
        if token == -2:
            raise RuntimeError('invalid neural output code')
        return None if token == -1 else token

    def command(self, command, on_text=None, stop=None):
        if len(command.encode()) > 38:
            raise ValueError('offline byte-token fixture accepts at most 38 command bytes')
        if stop is not None and stop.is_set():
            return {'stopped': True, 'neural_steps': 0}
        parked = self.executor.snapshot()
        try:
            result = self._command(command, on_text)
        except Exception:
            self.executor.restore(parked)
            raise
        if stop is not None and stop.is_set():
            self.executor.restore(parked)
            return {**result, 'stopped': True}
        return result

    def _command(self, command, on_text):
        started, before = time.perf_counter(), self.steps
        token = None
        for text in self.text.call(command):
            token = self.step(text)
        output = bytearray()
        for _ in range(self.max_steps):
            if token is None:
                break
            if token < self.offset:
                chunk = self.text.bytes[token]
                output.extend(chunk)
                if on_text:
                    on_text(chunk.decode(errors='replace'))
            token = self.step(token)
        else:
            raise RuntimeError('neural guest exhausted its trace budget')
        raw = output.decode(errors='replace')
        if '<tool_response>' not in raw or '</tool_response>' not in raw:
            raise RuntimeError('guest did not return a complete response')
        answer = raw.split('<tool_response>', 1)[1].split('</tool_response>', 1)[0].strip()
        return {'output': answer, 'raw': raw, 'neural_steps': self.steps - before,
                'seconds': time.perf_counter() - started}


def self_test():
    import resource
    from chipformer.rv import ref
    # A fallback would fail the test immediately, including during boot.
    def forbidden(*args, **kwargs):
        raise AssertionError('host ISA execution is forbidden during neural inference')
    original = ref.Machine.run, ref.Machine.record, ref.Rv32.step
    ref.Machine.run = ref.Machine.record = ref.Rv32.step = forbidden
    try:
        fixture = NeuralFixture()
        cases = [(': square dup * ;', 'defined square'), ('9 square .', '81'),
                 ('words', 'square'), ('12 0 ! 0 @ .', '12')]
        rows = []
        for command, expected in cases:
            row = fixture.command(command)
            if expected not in row['output']:
                raise AssertionError((command, row))
            print(f"PASS {command!r} -> {row['output']!r} ({row['neural_steps']} neural steps)", flush=True)
            rows.append({'command': command, **row})
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys
        rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
        if rss_bytes > 768 * 1024**2:
            raise RuntimeError('Offline fixture exceeded its 768 MiB test memory budget')
        print('PASS: guest parser, dictionary, arithmetic and cells execute in neural tensors.', flush=True)
        print('No pretrained Qwen weights loaded. Host ISA execution was disabled.', flush=True)
        print(f'Peak process RSS: {rss_bytes} bytes (budget: 768 MiB).', flush=True)
        return rows
    finally:
        ref.Machine.run, ref.Machine.record, ref.Rv32.step = original
