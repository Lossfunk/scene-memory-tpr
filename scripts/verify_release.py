#!/usr/bin/env python3
"""Verify the source, fitted models and archived evidence against the release hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'results/release_manifest.json').read_text())
for name, expected in manifest['sha256'].items():
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise SystemExit(f'Invalid manifest path: {name}')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Missing or changed release file: {name}')
print(f"Verified {len(manifest['sha256'])} release files.")
