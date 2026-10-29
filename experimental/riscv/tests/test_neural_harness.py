"""Offline tensor parity, actual guest customizations, and full-model refusal guards."""

import io
from pathlib import Path
import platform
import subprocess
from types import SimpleNamespace
import tarfile

import numpy as np
import pytest
import torch

from chipformer.q35 import ir
from chipformer.rv import core, ref
from chipformer.rv.fixture_tensors import SparseTensorCircuit
from chipformer.rv.harness_fixture import NeuralFixture
from chipformer.rv.neural_harness import QwenSession, archive, guard_full_model
from chipformer.rv.programs import program


def test_small_memory_tensor_lowering_matches_every_neural_residual_and_head():
    torch.set_num_threads(1)
    circuit = core.build()
    machine = ref.Machine.from_program(program('signext'))
    tokens = machine.run()[:12]
    core.snapshot(circuit, ref.Machine.from_program(program('signext')))
    network = SparseTensorCircuit(circuit)
    carry = {}
    expected = ir.evaluate(circuit, tokens, check=False, carry=carry)
    actual = torch.stack([network.step_token(token) for token in tokens]).numpy()
    np.testing.assert_array_equal(actual, expected)
    indices = {id(node): i for i, node in enumerate(circuit.nodes)}
    # State order follows the circuit's key heads at each scheduled level.
    flat = [stage[2]['state'] for stage in network.stages if stage[0] == 'head']
    level_states = dict(zip(sorted({n.level for n in circuit.nodes if isinstance(n, ir.Head)}),
                            flat, strict=True))
    for level, state in level_states.items():
        value = 0
        for node in (n for n in circuit.nodes if isinstance(n, ir.Head) and n.level == level):
            for vh in range(len(node.vheads())):
                expected_state = np.zeros((ir.HD, ir.HD))
                for row, data in carry[indices[id(node)], 'S', vh].items():
                    expected_state[row] = data
                np.testing.assert_array_equal(state[value].numpy(), expected_state)
                value += 1


@pytest.fixture(scope='module')
def neural_guest():
    return NeuralFixture()


def test_guest_dictionary_and_cells_live_in_neural_state(neural_guest, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('a host ISA interpreter was called')
    monkeypatch.setattr(ref.Machine, 'run', forbidden)
    monkeypatch.setattr(ref.Machine, 'record', forbidden)
    monkeypatch.setattr(ref.Rv32, 'step', forbidden)
    for command, expected in [(': square dup * ;', 'defined square'),
                              ('9 square .', '81'), ('words', 'square'),
                              ('12 0 ! 0 @ .', '12'),
                              ('0 if 9 . else 4 . then', '4')]:
        result = neural_guest.command(command)
        assert result['output'] == expected
        assert result['neural_steps'] > 0


def test_guest_error_is_a_real_neural_response(neural_guest):
    result = neural_guest.command('1 0 / .')
    assert 'error: division by zero' in result['output']
    assert neural_guest.command('2 3 + .')['output'] == '5'


def test_trace_budget_restores_recurrent_state_and_history(neural_guest):
    parked = neural_guest.executor.snapshot()
    limit = neural_guest.max_steps
    try:
        neural_guest.max_steps = 1
        with pytest.raises(RuntimeError, match='trace budget'):
            neural_guest.command('999 0 !')
    finally:
        neural_guest.max_steps = limit
    for actual, expected in zip(neural_guest.executor.snapshot(), parked, strict=True):
        for tensor, previous in zip(actual, expected, strict=True):
            torch.testing.assert_close(tensor, previous, rtol=0, atol=0)


def test_full_model_rejected_before_import_or_download_on_mac(monkeypatch):
    monkeypatch.setattr(platform, 'system', lambda: 'Darwin')
    with pytest.raises(RuntimeError, match='Full Qwen is disabled'):
        QwenSession()


@pytest.mark.parametrize('name, memory, host, error', [
    ('NVIDIA L4', 24 * 1024**3, 64 * 1024**3, '32 GiB'),
    ('NVIDIA A100', 40 * 1024**3, 16 * 1024**3, '40 GiB'),
])
def test_remote_model_guards(name, memory, host, error):
    with pytest.raises(RuntimeError, match=error):
        guard_full_model(system='Linux', cuda_available=True,
                         gpu_name=name, gpu_bytes=memory, host_bytes=host)


def test_b200_is_allowed_for_the_authorized_remote_test():
    guard_full_model(system='Linux', cuda_available=True, gpu_name='NVIDIA B200',
                     gpu_bytes=180 * 1024**3, host_bytes=128 * 1024**3)


def test_upload_contains_code_and_guest_firmware_only():
    with tarfile.open(fileobj=io.BytesIO(archive()), mode='r:gz') as tar:
        names = tar.getnames()
    assert 'chipformer/rv/env/harness/harness.elf' in names
    assert all(not name.startswith('/') and '..' not in Path(name).parts for name in names)
    assert not any(x in name for name in names for x in ('.safetensors', '.runpod', '.venv', '__pycache__'))


def test_launcher_does_not_load_a_model_without_configuration():
    root = Path(__file__).resolve().parents[3]
    env = __import__('os').environ.copy()
    env.pop('QWENPUTER_SSH', None)
    result = subprocess.run([str(root / 'qwen-harness')], cwd=root, env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert 'No remote GPU configured' in result.stderr


@pytest.mark.parametrize('paused', [False, True])
def test_conversation_persists_only_completed_neural_guest_state(paused):
    session = object.__new__(QwenSession)
    initial, updated = object(), object()
    passed = []

    def generate(ids, state, **kwargs):
        passed.append(state)
        kwargs['on_token'](0)
        kwargs['on_token'](1)
        return {'tokens': [0, 1], 'text_tokens': 2, 'chip_tokens': 12, 'circuit_paused': paused}

    session.chat = SimpleNamespace(v0=2, eos={1}, T=SimpleNamespace(bytes=[b'ok', b'<end>']),
                                   prompt_ids=lambda *args, **kwargs: [0], generate=generate,
                                   text_of=lambda ids: ''.join('ok' if i == 0 else '<end>' for i in ids),
                                   last_firmware_state=updated)
    session.state, session.messages = initial, []
    output = []
    if paused:
        with pytest.raises(RuntimeError, match='last parked state preserved'):
            session.send('test', output.append)
        assert session.state is initial and session.messages == []
    else:
        session.send('define a tool', output.append)
        session.send('call the tool', output.append)
        assert passed == [initial, updated]
        assert session.state is updated
        assert session.messages[-1] == {'role': 'assistant', 'content': 'ok'}
    assert '<end>' not in ''.join(output)
