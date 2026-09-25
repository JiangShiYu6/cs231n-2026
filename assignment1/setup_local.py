"""Local environment/data setup only; contains no assignment solutions."""
from pathlib import Path
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cs231n" / "datasets"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"


def setup():
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    expected = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    if all((DATA / "cifar-10-batches-py" / name).is_file() for name in expected):
        print("CIFAR-10 ready. Working directory:", ROOT)
        return
    archive = DATA / "cifar-10-python.tar.gz"
    if not archive.exists():
        print("Downloading CIFAR-10 (~163 MB)...", flush=True)
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(archive)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "md5").hexdigest()
    if digest != MD5:
        raise RuntimeError(f"CIFAR-10 checksum mismatch: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(DATA, filter="data")
    if not all((DATA / "cifar-10-batches-py" / name).is_file() for name in expected):
        raise RuntimeError("Incomplete CIFAR-10 extraction")
    print("CIFAR-10 downloaded and verified.", flush=True)


if __name__ == "__main__":
    setup()
