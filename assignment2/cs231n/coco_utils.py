import os, json
import numpy as np
import h5py

dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(dir_path, "datasets/coco_captioning")

def load_coco_data(base_dir=BASE_DIR, max_train=None, pca_features=True):
    print('base dir ', base_dir)
    data = {}
    caption_file = os.path.join(base_dir, "coco2014_captions.h5")
    with h5py.File(caption_file, "r") as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7_pca.h5")
    else:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7.h5")
    with h5py.File(train_feat_file, "r") as f:
        data["train_features"] = np.asarray(f["features"])

    if pca_features:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7_pca.h5")
    else:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7.h5")
    with h5py.File(val_feat_file, "r") as f:
        data["val_features"] = np.asarray(f["features"])

    dict_file = os.path.join(base_dir, "coco2014_vocab.json")
    with open(dict_file, "r") as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, "train2014_urls.txt")
    with open(train_url_file, "r") as f:
        train_urls = np.asarray([line.strip() for line in f])
    data["train_urls"] = train_urls

    val_url_file = os.path.join(base_dir, "val2014_urls.txt")
    with open(val_url_file, "r") as f:
        val_urls = np.asarray([line.strip() for line in f])
    data["val_urls"] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data["train_captions"].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data["train_captions"] = data["train_captions"][mask]
        data["train_image_idxs"] = data["train_image_idxs"][mask]
#         data["train_features"] = data["train_features"][data["train_image_idxs"]]
    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def sample_coco_minibatch(data, batch_size=100, split="train"):
    split_size = data["%s_captions" % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data["%s_captions" % split][mask]
    image_idxs = data["%s_image_idxs" % split][mask]
    image_features = data["%s_features" % split][image_idxs]
    urls = data["%s_urls" % split][image_idxs]
    return captions, image_features, urls


def sample_coco_preview(data, batch_size=3, split="train", max_attempts=20):
    """Sample reachable images for display only, keeping all rows aligned.

    Training continues to use sample_coco_minibatch and never needs image URLs.
    Returns captions, features, URLs, and decoded images; may return fewer rows
    if the download budget is exhausted. Each distinct URL is tried only once.
    """
    from .image_utils import image_from_url

    if batch_size < 1 or max_attempts < 1:
        raise ValueError("batch_size and max_attempts must be positive")
    captions = data[split + "_captions"]
    selected, images, seen = [], [], set()
    for row in np.random.permutation(len(captions)):
        image_idx = data[split + "_image_idxs"][row]
        url = str(data[split + "_urls"][image_idx])
        if url in seen:
            continue
        seen.add(url)
        image = image_from_url(url, timeout=5, quiet=True)
        if image is not None:
            selected.append(row)
            images.append(image)
        if len(images) >= batch_size or len(seen) >= max_attempts:
            break
    rows = np.asarray(selected, dtype=np.int64)
    image_idxs = data[split + "_image_idxs"][rows]
    if len(images) < batch_size:
        print("Preview: loaded %d/%d images after trying %d URLs. "
              "Some links may be unavailable or the network may be unreachable. "
              "Training uses local features and is unaffected."
              % (len(images), batch_size, len(seen)))
    return (captions[rows], data[split + "_features"][image_idxs],
            data[split + "_urls"][image_idxs], images)
