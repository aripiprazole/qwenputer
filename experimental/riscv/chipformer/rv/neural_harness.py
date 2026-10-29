"""Raw neural harness: remote Qwen runtime, or explicitly labeled offline tests."""

import argparse
import codecs
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tarfile
import threading

ROOT = Path(__file__).resolve().parents[2]
FIRMWARE = Path(__file__).parent / 'env/harness/harness.elf'
MODEL = 'divagr1925/Qwen3.5-4B-Qwenputer'
TOOL = {'type': 'function', 'function': {
    'name': 'harness', 'description': 'Execute or customize the harness inside your neural CPU. '
    'Its dictionary and cells persist. Define a word with : name body ; then call it. '
    'Example: : square dup * ; followed by 9 square . returns 81. '
    'Use short calls; the circuit capture window is 80 model tokens.',
    'parameters': {'type': 'object', 'properties': {'command': {'type': 'string',
        'description': 'Guest stack program: integers, + - * / % lt gt eq dup drop swap over, '
        '. emit cr, : name body ;, words forget, if else then, value cell ! and cell @.'}},
        'required': ['command']}}}
SYSTEM = ('You are Qwenputer. Your harness lives inside your RISC-V neural layers. '
          'Use the harness tool for execution. Its parser, dictionary, cells and programs run '
          'in your neural CPU; no external executor supplies results. You can customize it by '
          'defining guest words, then calling them in subsequent turns. Keep each call short. '
          'Read the actual tool_response before reporting results. Never invent a result. '
          'When asked to add a tool, define a guest word and verify it with a call.')


def guard_full_model(system=None, cuda_available=False, gpu_name='', gpu_bytes=0, host_bytes=0):
    if (system or platform.system()) == 'Darwin':
        raise RuntimeError('Full Qwen is disabled on this Mac. Use --self-test or --demo.')
    if not cuda_available:
        raise RuntimeError('The full neural harness requires a remote CUDA GPU. No CPU fallback.')
    if gpu_bytes < 32 * 1024**3 or host_bytes < 40 * 1024**3:
        raise RuntimeError('Need at least 32 GiB GPU memory and 40 GiB available host RAM.')


class QwenSession:
    description = 'Qwenputer · harness execution inside transformer layers · raw mode'

    def __init__(self, model=MODEL):
        # Refuse the Mac before importing torch, downloading weights or allocating a model.
        if platform.system() == 'Darwin':
            guard_full_model()
        import torch
        available = 0
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                available = int(line.split()[1]) * 1024
        props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        guard_full_model(cuda_available=props is not None,
                         gpu_name=props.name if props else '',
                         gpu_bytes=props.total_memory if props else 0, host_bytes=available)
        # A visible device can still need a host reset. Prove allocation and a kernel
        # work before downloading or loading any pretrained weights.
        try:
            torch.ones(1, device='cuda').sum().item()
        except RuntimeError as exc:
            raise RuntimeError(f'CUDA allocation/kernel probe failed before model loading: {exc}') from exc
        from huggingface_hub import snapshot_download
        from chipformer.rv.chat import RvChat
        path = Path(model)
        if not path.is_dir():
            path = Path(snapshot_download(model))
        if not (path / 'chipformer.json').is_file():
            raise ValueError('A merged Qwenputer checkpoint with chipformer.json is required.')
        self.chat = RvChat(path, device='cuda', fast=True, executor=False)
        self.reset()

    def reset(self):
        from chipformer.rv import elf
        self.state = self.chat.boot(elf.program(FIRMWARE))
        self.messages = [{'role': 'system', 'content': SYSTEM}]

    def send(self, message, on_text, stop=None):
        from chipformer.rv.harness_display import Transcript
        display = Transcript(on_text)
        self.messages.append({'role': 'user', 'content': message})
        ids = self.chat.prompt_ids(self.messages, tools=[TOOL])
        if len(ids) > 4096:
            self.messages.pop()
            raise ValueError('Conversation reached its input limit. Use /new to start a fresh guest.')
        decoder = codecs.getincrementaldecoder('utf-8')('replace')

        def token(index):
            if index < self.chat.v0 and index not in self.chat.eos:
                text = decoder.decode(self.chat.T.bytes[index])
                if text:
                    display.feed(text)
        try:
            result = self.chat.generate(ids, self.state, max_new=768, max_chip=12000,
                                        on_token=token, stop=stop)
            if result['circuit_paused']:
                raise RuntimeError('Guest trace budget exhausted; last parked state preserved. '
                                   'Ask for a smaller guest program.')
            if stop is not None and stop.is_set():
                self.messages.pop()
                return {**result, 'stopped': True}
            if result['text_tokens'] >= 768:
                raise RuntimeError('Model text budget exhausted; last parked state preserved.')
            self.state = self.chat.last_firmware_state
            text = self.chat.text_of([index for index in result['tokens'] if index not in self.chat.eos])
            self.messages.append({'role': 'assistant', 'content': text})
            return result
        except Exception:
            self.messages.pop()
            raise
        finally:
            display.finish()


class DemoSession:
    description = 'OFFLINE NEURAL FIXTURE · no pretrained Qwen · type guest programs'

    def __init__(self):
        self.reset()

    def reset(self):
        from chipformer.rv.harness_fixture import NeuralFixture
        self.fixture = NeuralFixture()

    def send(self, message, on_text, stop=None):
        from chipformer.rv.harness_display import Transcript
        display = Transcript(on_text)
        try:
            # Show the same call fed to the circuit; this formatter only renders text.
            display.feed(b''.join(self.fixture.text.bytes[token]
                                  for token in self.fixture.text.call(message)).decode())
            return self.fixture.command(message, on_text=display.feed, stop=stop)
        finally:
            display.finish()


def archive():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as tar:
        for path in (ROOT / 'chipformer').rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.elf', '.h', '.c', '.S', '.ld'):
                tar.add(path, arcname=str(path.relative_to(ROOT)))
    return data.getvalue()


def remote(args):
    target = args.ssh or os.environ.get('QWENPUTER_SSH', '')
    connection = {}
    config = ROOT.parents[1] / '.qwenputer.local.json'
    if not target and config.is_file():
        connection = json.loads(config.read_text())
        target = connection['ssh']
    if not target:
        raise ValueError('No remote GPU configured. Run ./qwen-harness --self-test or --demo '
                         'for offline tests. Full Qwen: ./qwen-harness --ssh user@gpu-host')
    if not re.fullmatch(r'[A-Za-z0-9_.@:-]+', target) or target.startswith('-'):
        raise ValueError('Invalid SSH destination')
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15']
    key, port = args.key or connection.get('key'), args.port or connection.get('port')
    if key:
        ssh += ['-i', key, '-o', 'IdentitiesOnly=yes']
    if port:
        ssh += ['-p', str(port)]
    if connection.get('known_hosts'):
        ssh += ['-o', 'UserKnownHostsFile=' + connection['known_hosts']]
    if connection.get('directory') and connection.get('python'):
        directory = connection.get('doom_directory') if args.doom else None
        command = ('cd ' + shlex.quote(directory or connection['directory']) + ' && exec '
                   + shlex.quote(connection['python']) + ' -m chipformer.rv.neural_harness --server')
        if args.model:
            command += ' --model ' + shlex.quote(args.model)
        if args.doom:
            command += ' --doom'
            if connection.get('doom_model') and not args.model:
                command += ' --model ' + shlex.quote(connection['doom_model'])
        if args.ask:
            command += ' --ask ' + shlex.quote(args.ask)
        os.execvp('ssh', [*ssh, '-tt', target, command])
        return
    # Hardware gate runs before installation/download. It cannot allocate the model.
    probe = ("import json,subprocess; from pathlib import Path; "
             "g=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total',"
             "'--format=csv,noheader,nounits'],text=True).splitlines()[0].split(','); "
             "m=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines()); "
             "print(json.dumps({'gpu':g[0].strip(),'bytes':int(g[1])*1024**2,"
             "'host':int(m['MemAvailable'].split()[0])*1024}))")
    result = subprocess.run([*ssh, target, 'python3 -c ' + shlex.quote(probe)],
                            check=True, text=True, capture_output=True)
    hardware = json.loads(result.stdout)
    guard_full_model(system='Linux', cuda_available=True, gpu_name=hardware['gpu'],
                     gpu_bytes=hardware['bytes'], host_bytes=hardware['host'])
    folder = '.cache/qwenputer-neural-harness'
    install = (f'mkdir -p {folder}/source && tar -xz -C {folder}/source && '
               f'python3 -m venv {folder}/venv && '
               f'{folder}/venv/bin/python -m pip install --disable-pip-version-check -q '
               "'numpy>=1.26' 'torch>=2.14.1' 'transformers>=5.19' safetensors tokenizers huggingface_hub")
    subprocess.run([*ssh, target, install], input=archive(), check=True)
    command = f'cd {folder}/source && ../venv/bin/python -m chipformer.rv.neural_harness --server'
    if args.doom:
        raise ValueError('--doom needs a prepared remote (doom_directory and doom_model in '
                         '.qwenputer.local.json): its checkpoint is built there by merge.')
    command += ' --model ' + shlex.quote(args.model)
    if args.ask:
        command += ' --ask ' + shlex.quote(args.ask)
    os.execvp('ssh', [*ssh, '-tt', target, command])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--self-test', action='store_true', help='bounded neural guest test; no full model')
    mode.add_argument('--demo', action='store_true', help='raw offline neural guest; no language model')
    mode.add_argument('--server', action='store_true', help='full merged Qwen on a suitable remote GPU')
    parser.add_argument('--ssh', help='remote GPU destination; also QWENPUTER_SSH')
    parser.add_argument('--key', help='SSH private-key path')
    parser.add_argument('--port', type=int)
    parser.add_argument('--model', help=f'merged checkpoint (default {MODEL}; with --doom, '
                        '$QWENPUTER_DOOM_MODEL or dist/q4-doom)')
    parser.add_argument('--doom', action='store_true',
                        help='play DOOM: its firmware runs in a doom-map checkpoint\'s circuit')
    parser.add_argument('--ask', help='single request; otherwise raw interactive mode')
    args = parser.parse_args()
    if args.doom:
        os.environ['QWENPUTER_RV_MAP'] = 'doom'  # before any chipformer.rv import reads spec
    if args.model is None and not args.doom:
        args.model = MODEL
    try:
        if args.self_test:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
            limit = min(180, hard) if hard != resource.RLIM_INFINITY else 180
            resource.setrlimit(resource.RLIMIT_CPU, (limit, hard))
            from chipformer.rv.harness_fixture import self_test
            self_test()
            return
        if not args.demo and not args.server:
            remote(args)
            return
        if not args.ask and not sys.stdin.isatty():
            raise ValueError('Raw mode needs a terminal; use --ask for a noninteractive request.')
        if args.doom and args.server:
            from chipformer.rv.doom_harness import MODEL as DOOM_MODEL, DoomSession
            session = DoomSession(args.model or DOOM_MODEL)
        elif args.doom:
            raise ValueError('--doom runs on the remote GPU only (with --server, or over --ssh).')
        else:
            session = DemoSession() if args.demo else QwenSession(args.model)
        if args.ask:
            session.send(args.ask, lambda text: print(text, end='', flush=True), threading.Event())
            print()
        else:
            from chipformer.rv.raw_terminal import RawTerminal
            RawTerminal(session).run()
    except (ValueError, RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print(f'qwen-harness: {exc}', file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    main()
