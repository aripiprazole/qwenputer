"""Firmware startup and its continuation execute in stock HF transformer layers."""

import pytest
import torch

from chipformer.rv import bridge as BR, core, hf, ref
from chipformer.rv import spec as S
from chipformer.rv.asm import assemble
from chipformer.rv.chat import RvChat
from chipformer.rv.decode import CodeDecoder


class ByteText:
    bytes = [bytes([i]) for i in range(256)] + [s.encode() for s in BR.SPECIALS]
    bytes += [b"<tool_call>", b"</tool_call>"]
    printable = list(range(260))
    TCO, TCC, NL = 260, 261, 10

    def printable_index(self, token):
        return token if 0 <= token < len(self.printable) else None

    def row_bytes(self, token):
        data = self.bytes[token] if 0 <= token < len(self.bytes) else b""
        return data[:S.TEXT_ROW].ljust(S.TEXT_ROW, bytes(1))


class RemappedEmbedding(torch.nn.Module):
    """The emitted fixture stores only the rows its trace uses, with compact IDs."""

    def __init__(self, embedding, ids, offset):
        super().__init__()
        self.embedding = embedding
        mapping = torch.zeros(offset + S.VOCAB, dtype=torch.long)
        for token, index in ids.items():
            if token is not None:
                model_id = token - S.TEXT_TOKEN if S.is_text(token) else offset + token
                mapping[model_id] = index
        self.register_buffer("mapping", mapping)

    def forward(self, ids):
        return self.embedding(self.mapping[ids])


@pytest.fixture(scope="module")
def firmware():
    # Startup parks with a0=5. The next tool call resumes after that ecall and returns 12.
    program = assemble("li a0, 5\nli a7, 93\necall\naddi a0, a0, 7\necall")
    text = ByteText()
    oracle = ref.Machine.from_program(program)
    boot_trace = BR.park(oracle)
    session = BR.Session(text, oracle)
    session.qwen([text.TCC])
    assert oracle.exit_code == 12
    fault = assemble("ebreak")
    fault_trace = ref.Machine.from_program(fault).run()
    circuit = core.build(text=text)
    model, layout, ids = hf.build_model(
        circuit, boot_trace + session.toks + fault_trace + [BR.text_token(text.NL)])
    chat = object.__new__(RvChat)
    chat.model, chat.lm, chat.c, chat.T = model, model.model, circuit, text
    chat.v0, chat.n_chip, chat.dev = len(text.bytes), circuit.n_layers(), "cpu"
    chat.fast, chat.executor, chat.verify = False, False, False
    chat.eos = set()
    dims = []
    for expression in circuit.code:
        ((dim, weight),) = expression.terms.items()
        assert weight == 1
        dims.append(layout.dim(dim))
    chat.code_dims = torch.tensor(dims)
    chat.decoder = CodeDecoder(chat.v0, text)
    chat.lm.embed_tokens = RemappedEmbedding(chat.lm.embed_tokens, ids, chat.v0)
    return chat, program, session.toks, boot_trace, fault


@pytest.fixture
def no_reference_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the reference interpreter must not execute firmware")

    monkeypatch.setattr(ref.Machine, "run", forbidden)
    monkeypatch.setattr(ref.Machine, "record", forbidden)
    monkeypatch.setattr(ref.Rv32, "step", forbidden)


@torch.no_grad()
def test_neural_boot_parks_and_resumes_without_host_instruction_execution(firmware,
                                                                        no_reference_execution):
    chat, program, trace, boot_trace, _ = firmware
    parked = chat.boot(program, max_steps=len(boot_trace) + 4)
    assert parked.steps == len(boot_trace)
    saved = tuple(state.clone() for state in parked.heads)
    prefix = torch.tensor([[chat.T.NL]])
    cache = chat.lm(input_ids=prefix, use_cache=True).past_key_values
    parked.inject(cache)
    token = chat.T.TCC
    for expected in trace[1:]:
        hc, _ = chat._layers(token, cache, None)
        token = chat.circuit_token(hc)
        model_id = expected - S.TEXT_TOKEN if S.is_text(expected) else chat.v0 + expected
        assert token == model_id
    # Restoring and then running a conversation must not mutate the reusable boot state.
    for actual, expected in zip(parked.heads, saved, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_neural_boot_budget_and_faults_have_no_native_fallback(firmware, no_reference_execution):
    chat, program, _, _, fault = firmware
    with pytest.raises(RuntimeError, match="did not halt within 1"):
        chat.boot(program, max_steps=1)
    with pytest.raises(RuntimeError, match="faulted during neural boot"):
        chat.boot(fault)


@torch.no_grad()
def test_chat_reuses_neural_boot_state_and_runs_the_tool_in_weights(firmware,
                                                                 no_reference_execution):
    chat, program, trace, _, _ = firmware
    parked = chat.boot(program)
    original = chat.model.lm_head
    # Force a tool request at the language head. Its response must still come from
    # the actual RISC-V transformer layers, including resuming after startup.
    head = torch.nn.Linear(chat.model.config.hidden_size, chat.v0, bias=True).to(torch.bfloat16)
    head.weight.zero_()
    head.bias.zero_()
    head.bias[chat.T.TCC] = 1
    chat.model.lm_head = head
    expected = [token - S.TEXT_TOKEN if S.is_text(token) else chat.v0 + token for token in trace]
    try:
        for _ in range(2):
            received = []
            result = chat.generate([chat.T.NL], parked, max_new=2, on_token=received.append)
            assert result["tokens"] == expected
            assert received == expected
            assert result["text_tokens"] == 2
            assert result["chip_tokens"] == len(expected) - 2
    finally:
        chat.model.lm_head = original
