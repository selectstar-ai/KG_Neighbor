"""Text and image encoders (Section 4.1).

Requires the ``embed`` extra. Paper encoders: BERT-base and Qwen3-Embedding-8B
(text), ResNet50 and CLIP ViT-L/14 (image). Outputs are L2-normalized float32.
"""

from __future__ import annotations

import numpy as np


def resolve_device(requested: str = "auto") -> str:
    """``"auto"`` picks cuda > mps > cpu."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def length_batches(
    lengths: list[int],
    *,
    batch_size: int,
    token_budget: int | None = None,
    max_seq_length: int | None = None,
) -> list[list[int]]:
    """Group indices longest first, at most ``batch_size`` per batch.

    With ``token_budget``, a batch also keeps ``len(batch) * longest length <= token_budget``
    (lengths capped at ``max_seq_length``); a single text that exceeds the budget forms its
    own batch. Longest first so that an out-of-memory batch fails at the start.
    """
    capped = [min(n, max_seq_length) if max_seq_length else n for n in lengths]
    order = sorted(range(len(capped)), key=lambda i: -capped[i])
    batches, start = [], 0
    while start < len(order):
        size = batch_size
        if token_budget is not None:
            size = max(1, min(batch_size, token_budget // max(capped[order[start]], 1)))
        batches.append(order[start : start + size])
        start += size
    return batches


def load_text_encoder(
    model: str,
    *,
    device: str = "auto",
    dtype=None,
    max_seq_length: int | None = None,
):
    """Load a sentence-transformers text encoder once; pass it to :func:`encode_texts`.

    ``max_seq_length`` truncates inputs to this many tokens (default: the model's own limit).
    """
    from sentence_transformers import SentenceTransformer

    kwargs = {"model_kwargs": {"torch_dtype": dtype}} if dtype is not None else {}
    st = SentenceTransformer(model, device=resolve_device(device), **kwargs)
    if max_seq_length is not None:
        st.max_seq_length = max_seq_length
    return st


def embed_texts(
    texts: list[str],
    model: str,
    *,
    device: str = "auto",
    dtype=None,
    batch_size: int = 32,
    max_seq_length: int | None = None,
    token_budget: int | None = None,
) -> np.ndarray:
    """(N, d) embeddings: :func:`load_text_encoder` then :func:`encode_texts`."""
    st = load_text_encoder(model, device=device, dtype=dtype, max_seq_length=max_seq_length)
    return encode_texts(st, texts, batch_size=batch_size, token_budget=token_budget)


def encode_texts(
    st,
    texts: list[str],
    *,
    batch_size: int = 32,
    token_budget: int | None = None,
) -> np.ndarray:
    """(N, d) L2-normalized float32 embeddings with an encoder from :func:`load_text_encoder`.

    ``token_budget``: if set, batch by length so that a batch holds at most this many tokens
    (``batch_size`` stays the upper bound); for long documents. Otherwise every batch has
    ``batch_size`` texts.
    """
    if token_budget is None:
        emb = st.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    from tqdm import tqdm

    lengths: list[int] = []
    for i in range(0, len(texts), 1000):  # chunks: only lengths are kept
        ids = st.tokenizer(
            texts[i : i + 1000], add_special_tokens=True, truncation=True,
            max_length=st.max_seq_length,
        )["input_ids"]
        lengths.extend(len(x) for x in ids)
    batches = length_batches(lengths, batch_size=batch_size, token_budget=token_budget,
                             max_seq_length=st.max_seq_length)
    out = None
    for b in tqdm(batches, desc="embed"):
        emb = st.encode(
            [texts[i] for i in b],
            batch_size=len(b),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if out is None:
            out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
        out[b] = emb
    return out


def embed_images(
    image_paths: list[str],
    model: str,
    *,
    device: str = "auto",
    batch_size: int = 32,
) -> np.ndarray:
    """(N, d) image embeddings.

    ``model`` is a CLIP name for sentence-transformers (e.g. ``"clip-ViT-L-14"``)
    or ``"resnet50"`` (torchvision ImageNet weights, 2048-d penultimate features).
    """
    from PIL import Image

    device = resolve_device(device)
    if "clip" in model.lower():
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model, device=device)
        images = [Image.open(p).convert("RGB") for p in image_paths]
        emb = st.encode(
            images,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    if model != "resnet50":
        raise ValueError(f"unsupported image model {model!r}; use a CLIP model or 'resnet50'")
    import torch
    from torchvision import models, transforms

    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net.eval().to(device)
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    out = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = torch.stack(
                [tf(Image.open(p).convert("RGB")) for p in image_paths[i : i + batch_size]]
            )
            out.append(net(batch.to(device)).float().cpu().numpy())
    emb = np.concatenate(out).astype(np.float32)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
