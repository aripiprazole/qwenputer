"""Fetch the shareware DOOM1.WAD and lay it out for the DROM; no model is loaded.

The DROM only allows aligned loads, and Doom reads lump fields in place, so every lump starts on
a 4-byte boundary. ``keep`` drops lumps the firmware never reads (sounds, music, demos) to make
the read-only memory, and so the circuit, smaller.
"""

import hashlib
import struct
import urllib.request
from pathlib import Path

URL = "https://github.com/nneonneo/universal-doom/raw/main/DOOM1.WAD"
MD5 = "f0cefca49926d00903cf57551d901abe"  # DOOM1.WAD v1.9 shareware (id Software)
CACHE = Path.home() / ".cache/qwenputer"


def fetch() -> Path:
    path = CACHE / "doom1.wad"
    if not path.is_file():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as r:
            blob = r.read()
        if hashlib.md5(blob).hexdigest() != MD5:
            raise SystemExit(f"{URL}: unexpected checksum")
        path.write_bytes(blob)
    if hashlib.md5(path.read_bytes()).hexdigest() != MD5:
        raise SystemExit(f"{path}: unexpected checksum; delete it to download again")
    return path


def lumps(blob: bytes) -> tuple[bytes, list[tuple[str, bytes]]]:
    kind, n, table = struct.unpack_from("<4sii", blob)
    out = []
    for i in range(n):
        pos, size, name = struct.unpack_from("<ii8s", blob, table + 16 * i)
        out.append((name.rstrip(b"\0").decode("latin-1"), blob[pos:pos + size]))
    return kind, out


def drop(name: str) -> bool:
    """Lumps the firmware never reads: sound effects, music and demos."""
    return name.startswith(("DS", "DP", "D_", "DEMO"))


def build(out: Path, trim: bool = True) -> Path:
    kind, entries = lumps(fetch().read_bytes())
    if trim:
        # an empty lump keeps each name, so lookups (W_GetNumForName) still succeed
        entries = [(name, b"" if drop(name) else data) for name, data in entries]
    body = bytearray(bytes(12))
    table = bytearray()
    for name, data in entries:
        body += bytes(-len(body) % 4)
        table += struct.pack("<ii8s", len(body) if data else 0, len(data), name.encode("latin-1"))
        body += data
    body += bytes(-len(body) % 4)
    struct.pack_into("<4sii", body, 0, kind, len(entries), len(body))
    out.write_bytes(bytes(body + table))
    return out


if __name__ == "__main__":
    import sys
    print(build(Path(sys.argv[1]), trim="--full" not in sys.argv))
