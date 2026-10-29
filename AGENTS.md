# Active goal

Make Qwenputer faster while keeping RISC-V execution inside the compiled transformer
layers. Preserve this goal across turns unless the user explicitly changes it.

The current deliverable is a raw-terminal command for an agent-customizable harness
whose parser, dictionary and programs execute as guest firmware inside Qwen's circuit.
Do not load the full Qwen model on this Mac. The user has authorized a remote GPU test
on Community Cloud using the original published Qwenputer checkpoint. Use bounded offline
fixtures for local tests and clearly label them. Never present a fixture as a live pretrained
Qwen conversation.
Match Vercel fx's terminal presentation: normal scrollback, minimal raw-mode prompt,
streamed replies and compact tool records, with no alternate-screen dashboard.

## Execution constraints

- RISC-V instructions, registers, memory updates and program results must be computed
  by the neural circuit during model inference.
- Do not replace this execution with native C/Rust/Python executors, hand-written ISA
  handlers, external calculator tools, or precomputed program answers.
- GPU kernels and compiler optimizations may evaluate the circuit's tensor operations;
  they must preserve its semantics rather than implement the guest instructions directly.
- Reference interpreters are correctness oracles for tests only, never the production
  execution path or a hidden fallback.
- The native runtime and external harness are preserved on `codex/import-qwen-runtime`.
  Do not use that approach to satisfy this branch's performance goal.

## Optimization and evidence

- Focus on model kernels, unnecessary zero padding, graph capture, GPU token loops,
  caching and batching while retaining the in-model CPU.
- Verify circuit tokens and recurrent state against the reference before claiming gains.
- Compare the same model, workload, limits and hardware. Report circuit speed, text
  decoding and end-to-end latency separately; label unmeasured proposals as proposals.
