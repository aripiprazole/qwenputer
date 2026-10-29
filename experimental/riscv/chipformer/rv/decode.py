"""Decode the circuit's output bits into vocabulary IDs using device tensors only.

This is token decoding, not RISC-V execution. The transformer layers produce every
output bit; these tables describe the same vocabulary as core.decode_code.
"""

import torch

from chipformer.rv import core
from chipformer.rv import spec as S


class CodeDecoder(torch.nn.Module):
    UNOWNED = -1
    INVALID = -2

    def __init__(self, vocab_offset: int, text=None, device=None):
        super().__init__()
        count = 1 << core.NFAM
        bases, limits, payloads = [0] * count, [0] * count, [0] * count
        printable = list(text.printable) if text is not None else []
        for fid, name in enumerate(core.OUT_NAMES):
            if name == "Q":
                limits[fid] = len(printable)
            elif name in S.CONTROLS:
                bases[fid] = vocab_offset + S.tok(name)
                limits[fid] = 1 << core.NPAY  # controls ignore their payload bits
            else:
                base, size = S.FAMILIES[name]
                bases[fid], limits[fid], payloads[fid] = vocab_offset + base, size, 1
        for name, values in (("bases", bases), ("limits", limits), ("payloads", payloads),
                             ("printable", printable or [0]),
                             ("bits", [1 << i for i in range(core.NFAM + core.NPAY)])):
            self.register_buffer(name, torch.tensor(values, dtype=torch.long, device=device))

    def forward(self, code: torch.Tensor) -> torch.Tensor:
        """IDs for [..., NFAM + NPAY] signed codes; -1 unowned, -2 invalid."""
        packed = ((code > 0).long() * self.bits).sum(-1)
        family = packed & ((1 << core.NFAM) - 1)
        payload = packed >> core.NFAM
        # torch.take, not tensor[index]: a 0-dim index would become a host .item() sync,
        # which breaks CUDA graph capture of a single-token step.
        token = torch.take(self.bases, family) + payload * torch.take(self.payloads, family)
        echoed = torch.take(self.printable, payload.clamp(max=self.printable.numel() - 1))
        token = torch.where(family == core.FAM_ID["Q"], echoed, token)
        token = torch.where(payload < torch.take(self.limits, family), token, self.INVALID)
        return torch.where((code != 0).any(-1), token, self.UNOWNED)
