"""Device token decoding must preserve the entire circuit vocabulary and ownership."""

from types import SimpleNamespace

import pytest
import torch

from chipformer.rv import core
from chipformer.rv import spec as S
from chipformer.rv.chat import RvChat
from chipformer.rv.decode import CodeDecoder

OFFSET = 248320
TEXT = SimpleNamespace(printable=[(i * 13 + 17) % 1000 for i in range(260)])


def codes(family, payload):
    packed = (payload << core.NFAM) | family
    bits = (packed[..., None] >> torch.arange(core.NFAM + core.NPAY)) & 1
    return (bits * 2 - 1).to(torch.bfloat16) * 32


@pytest.mark.parametrize("name", core.OUT_NAMES)
def test_every_valid_token_and_control_payload(name):
    fid = core.FAM_ID[name]
    if name == "Q":
        payload = torch.arange(len(TEXT.printable))
        expected = torch.tensor(TEXT.printable)
    elif name in S.CONTROLS:
        payload = torch.arange(1 << core.NPAY)
        expected = torch.full_like(payload, OFFSET + S.tok(name))
    else:
        base, size = S.FAMILIES[name]
        payload = torch.arange(size)
        expected = OFFSET + base + payload
    actual = CodeDecoder(OFFSET, TEXT)(codes(fid, payload))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unowned_and_invalid_codes_never_become_tokens():
    decoder = CodeDecoder(OFFSET, TEXT)
    assert decoder(torch.zeros(core.NFAM + core.NPAY)).item() == CodeDecoder.UNOWNED
    for name, (_, size) in S.FAMILIES.items():
        if size < 1 << core.NPAY:
            assert decoder(codes(core.FAM_ID[name], torch.tensor(size))).item() == CodeDecoder.INVALID
    assert decoder(codes(core.FAM_ID["Q"], torch.tensor(260))).item() == CodeDecoder.INVALID
    assert decoder(codes((1 << core.NFAM) - 1, torch.tensor(0))).item() == CodeDecoder.INVALID
    assert CodeDecoder(OFFSET)(codes(core.FAM_ID["Q"], torch.tensor(0))).item() == CodeDecoder.INVALID
    # All-negative is an owned V(0), distinct from the all-zero unowned code.
    assert decoder(codes(0, torch.tensor(0))).item() == OFFSET + S.tok("V", 0)


def test_decoder_can_be_captured_without_host_control_flow():
    decoder = CodeDecoder(OFFSET, TEXT)
    compiled = torch.compile(decoder, backend="eager", fullgraph=True)
    payload = torch.tensor([0, 1, 259, 260, 65535])
    actual = compiled(codes(core.FAM_ID["Q"], payload))
    expected = torch.tensor([TEXT.printable[0], TEXT.printable[1], TEXT.printable[259], -2, -2])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_chat_preserves_invalid_code_diagnostic():
    chat = object.__new__(RvChat)
    chat.decoder = CodeDecoder(OFFSET, TEXT)
    assert chat.code_token(codes(core.FAM_ID["HALT"], torch.tensor(0))) == OFFSET + S.tok("HALT")
    assert chat.code_token(torch.zeros(core.NFAM + core.NPAY)) is None
    with pytest.raises(RuntimeError, match="family 31, payload 123"):
        chat.code_token(codes(31, torch.tensor(123)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for graph replay")
def test_decoder_cuda_graph_reads_new_codes_on_replay():
    decoder = CodeDecoder(OFFSET, TEXT, device="cuda")
    value = codes(core.FAM_ID["Q"], torch.tensor(10)).cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            decoder(value)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = decoder(value)
    for name, payload in [("GO", 0), ("V", 123), ("Q", 259), ("Q", 260)]:
        value.copy_(codes(core.FAM_ID[name], torch.tensor(payload)))
        graph.replay()
        token = core.decode_code(core.FAM_ID[name], payload, TEXT)
        expected = (-2 if token < 0 else token - S.TEXT_TOKEN if S.is_text(token)
                    else OFFSET + token)
        assert result.item() == expected
