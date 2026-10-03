"""Local environment and data preparation; assignment solutions stay untouched."""
from pathlib import Path
import os
import shutil
import sys
import urllib.request
import http.client

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cs231n" / "datasets"
CACHE = ROOT / ".cache"


def download(url, target, force=False):
    target = Path(target)
    if not force and target.is_file() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    print("Downloading:", target.name, flush=True)
    # Bounded range requests avoid truncated large transfers on this network.
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as response:
        size = int(response.headers["Content-Length"])
    offset = 0
    chunk = 8 * 1024 * 1024
    with partial.open("wb") as out:
        while offset < size:
            end = min(offset + chunk, size) - 1
            for attempt in range(4):
                try:
                    separator = "&" if "?" in url else "?"
                    request_url = url + separator + f"cs231n_range={offset}"
                    req = urllib.request.Request(request_url, headers={"Range": f"bytes={offset}-{end}"})
                    with urllib.request.urlopen(req, timeout=120) as response:
                        payload = response.read()
                        if response.status == 200:
                            if len(payload) != size:
                                raise IOError("Incomplete download")
                            out.seek(0)
                            out.truncate()
                            out.write(payload)
                            offset = size
                        else:
                            expected_range = f"bytes {offset}-{end}/{size}"
                            if response.headers.get("Content-Range") != expected_range or len(payload) != end - offset + 1:
                                raise IOError("Incomplete download range")
                            out.write(payload)
                            offset += len(payload)
                    break
                except (OSError, ValueError, http.client.HTTPException):
                    if attempt == 3:
                        raise
    if partial.stat().st_size != size:
        raise IOError("Download size mismatch")
    partial.replace(target)
    print("Ready:", target.name, flush=True)
    return target


def _reuse_tree(source, target):
    if not source.is_dir():
        raise FileNotFoundError(f"Dataset missing: {source}")
    for path in source.rglob("*"):
        if path.is_file():
            dest = target / path.relative_to(source)
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, dest)
            except OSError:
                shutil.copy2(path, dest)


def setup(coco=False, cifar=False):
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    DATA.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
    os.environ.setdefault("TFDS_DATA_DIR", str(DATA / "tensorflow_datasets"))
    # TFDS temporary download names exceed Windows MAX_PATH under a deep repo.
    os.environ.setdefault("CS231N_TFDS_DOWNLOAD_DIR", str(Path(sys.prefix).parent / "tmp/a3-downloads"))
    os.environ.setdefault("TFDS_DISABLE_GCS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    if coco and not (DATA / "coco_captioning/coco2014_captions.h5").exists():
        _reuse_tree(ROOT.parent / "assignment2/cs231n/datasets/coco_captioning",
                   DATA / "coco_captioning")
    if cifar and not (ROOT / "data/cifar-10-batches-py/test_batch").exists():
        _reuse_tree(ROOT.parent / "assignment1/cs231n/datasets/cifar-10-batches-py",
                   ROOT / "data/cifar-10-batches-py")
    import torch
    torch.set_num_threads(2)
    torch.hub.set_dir(str(CACHE / "torch/hub"))
    print("Assignment 3 ready; device:", "cuda" if torch.cuda.is_available() else "cpu")
    return ROOT


def prepare_simclr():
    return download(
        "https://cs231n.stanford.edu/2025/storage/a3/pretrained_simclr_model.pth",
        ROOT / "pretrained_model/pretrained_simclr_model.pth")


def prepare_ddpm():
    for name in ("emoji_data.npz", "text_embeddings.pt"):
        download("https://cs231n.stanford.edu/2025/storage/a3/" + name, DATA / name)
    download("https://cs231n.stanford.edu/2025/storage/a3/model-70000.pt",
             ROOT / "cs231n/exp/pretrained/model-70000.pt")


def load_dino():
    import torch
    cached = CACHE / "torch/hub/facebookresearch_dino_main"
    if cached.is_dir():
        return torch.hub.load(str(cached), "dino_vits8", source="local")
    return torch.hub.load("facebookresearch/dino:main", "dino_vits8",
                          trust_repo=True, skip_validation=True)


def load_clip_examples(data, preferred, count=10, max_attempts=50):
    """Keep caption/image pairs aligned when old Flickr URLs have disappeared."""
    import numpy as np
    from cs231n.coco_utils import decode_captions
    from cs231n.image_utils import image_from_url
    candidates = list(preferred) + np.random.default_rng(231).permutation(
        len(data["val_captions"])).tolist()
    captions, images, seen = [], [], set()
    for row in candidates:
        image_idx = data["val_image_idxs"][row]
        url = str(data["val_urls"][image_idx])
        if url in seen:
            continue
        seen.add(url)
        image = image_from_url(url, quiet=True)
        if image is not None:
            caption = decode_captions(data["val_captions"][row], data["idx_to_word"])
            for token in ("<START>", "<END>", "<UNK>"):
                caption = caption.replace(token, "")
            captions.append(caption.strip())
            images.append(image)
        if len(images) == count or len(seen) >= max_attempts:
            break
    if len(images) < count:
        raise RuntimeError(f"Only {len(images)}/{count} images available. Check the network and retry.")
    print(f"Loaded {count} image/caption pairs; successful images are cached locally.")
    return captions, images


if __name__ == "__main__":
    setup(coco=True, cifar=True)
    prepare_simclr()
    prepare_ddpm()
