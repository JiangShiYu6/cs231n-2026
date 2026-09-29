"""Local setup helpers only; no assignment solutions."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cs231n" / "datasets"


def download(url, target):
    if target.exists():
        return
    partial = target.with_suffix(target.suffix + ".part")
    print("Downloading", url, flush=True)
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    partial.replace(target)


def setup(coco=False):
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    DATA.mkdir(exist_ok=True)
    cifar = DATA / "cifar-10-batches-py"
    source = ROOT.parent / "assignment1" / "cs231n" / "datasets" / cifar.name
    expected = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch", "batches.meta"]
    if not all((cifar / name).is_file() for name in expected) and source.exists():
        shutil.copytree(source, cifar, dirs_exist_ok=True)
    if not all((cifar / name).is_file() for name in expected):
        raise FileNotFoundError("CIFAR-10 missing; restore assignment1 dataset or download CIFAR-10 here.")
    if coco:
        required = ["coco2014_captions.h5", "coco2014_vocab.json", "train2014_urls.txt", "val2014_urls.txt",
                    "train2014_vgg16_fc7_pca.h5", "val2014_vgg16_fc7_pca.h5"]
        if not all((DATA / "coco_captioning" / name).is_file() for name in required):
            archive = DATA / "coco_captioning.zip"
            download("https://cs231n.stanford.edu/coco_captioning.zip", archive)
            with zipfile.ZipFile(archive) as z:
                bad = z.testzip()
                if bad:
                    raise RuntimeError(f"Corrupt ZIP member: {bad}")
                for name in z.namelist():
                    if not (DATA / name).resolve().is_relative_to(DATA.resolve()):
                        raise ValueError("Unsafe archive path")
                z.extractall(DATA)
            if not all((DATA / "coco_captioning" / name).is_file() for name in required):
                raise RuntimeError("COCO extraction is incomplete")
    print("Assignment 2 data ready:", ROOT)


def build_extensions():
    import numpy as np
    folder = ROOT / "cs231n"
    target = folder / ("im2col_cython" + sysconfig.get_config_var("EXT_SUFFIX"))
    if target.exists() and target.stat().st_mtime >= (folder / "im2col_cython.pyx").stat().st_mtime:
        print("Cython extension ready:", target.name)
        return
    if os.name == "nt" and shutil.which("gcc"):
        # Compile the supplied Cython source with the locally installed MinGW compiler.
        build = folder / "build"
        build.mkdir(exist_ok=True)
        c_file = build / "im2col_cython.c"
        subprocess.run([sys.executable, "-m", "cython", "-3", str(folder / "im2col_cython.pyx"), "-o", str(c_file)], check=True)
        base = Path(sys.base_prefix)
        subprocess.run([shutil.which("gcc"), "-shared", "-O2", "-static-libgcc",
                        "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
                        "-I" + str(base / "include"), "-I" + np.get_include(),
                        str(c_file), str(base / "libs" / f"python{sys.version_info.major}{sys.version_info.minor}.lib"),
                        "-o", str(target)], check=True)
    else:
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=folder, check=True)
    print("Cython extension built:", target.name)


if __name__ == "__main__":
    setup(coco="--coco" in sys.argv)
    if "--build" in sys.argv:
        build_extensions()
