"""Obtain two pinned upstream assets, verify SHA-256, and keep them untracked."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def fetch(source_directory=None):
    lock = json.loads((ROOT / "configs/upstream.json").read_text())
    for item in lock["assets"]:
        target = ROOT / "assets/minimal_world_model_interp" / item["path"]
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Existing asset has a different hash; left untouched: {target}")
            print(f"Verified existing {item['path']}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        try:
            if source_directory:
                shutil.copyfile(Path(source_directory) / item["path"], temporary)
            else:
                print(f"Downloading pinned {item['path']} ({item['bytes']:,} bytes)", flush=True)
                with urllib.request.urlopen(item["url"], timeout=120) as response, temporary.open("wb") as out:
                    shutil.copyfileobj(response, out)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Downloaded/copied asset failed SHA-256 verification: {item['path']}")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Verified {item['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-directory", type=Path, help="Use an existing upstream checkout instead of downloading")
    fetch(parser.parse_args().from_directory)
