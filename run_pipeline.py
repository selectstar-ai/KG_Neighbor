"""KG-Neighbor pipeline: preprocess -> embed -> kg_neighbor (+ plots) -> idp.

Stages run in this order whatever order the flags are given in.

Examples
--------
python run_pipeline.py --preprocess --dataset stackoverflow
python run_pipeline.py --preprocess --dataset cub200
python run_pipeline.py --embed --kg_neighbor --dataset cub200 --encoder resnet50
python run_pipeline.py --idp --dataset cub200 --encoder resnet50 --dry_run

Inputs and outputs
------------------
data/<dataset>/                                texts/sentences.jsonl, meta.json, images/, kg/triples.tsv
results/<dataset>/embeddings/<encoder>/        embedding_{vectors,labels,ids}.npy, embedding_meta.json, parts/
results/<dataset>/layouts/<encoder>/default/   layout_knn.npy, layout_<measure>_a<alpha>.npy
results/<dataset>/metrics/<encoder>/metric_default.json
results/<dataset>/plot/<encoder>/default/      plot_knn.png, plot_<measure>_a<alpha>.png
results/<dataset>/analysis/<encoder>/default/  analysis_plot_grid.png, analysis_alpha_sweep.png, analysis_retention.png
results/<dataset>/idp/idp_questions.json       questions the layouts need, keyed by "<sample_id>|<class_name>"
results/<dataset>/idp/<judge_model>/           idp_meta.json: judge settings and the state of every batch
                                               batches/batch_<n>.jsonl: requests, saved before they are sent
                                               responses/batch_<n>.jsonl: the judge's result rows
                                               idp_judgments.json: answers, keyed like the questions
                                               <encoder>/idp_default.json: the diagnosis

``default`` is the KG of the dataset (kg/triples.tsv); the pipeline knows this one KG only.
The judge model names its idp folder: gpt-5.4 (paper), claude-opus-5-5 with ``--judge anthropic``.
A question judged there is never sent again, and an interrupted run resumes from idp_meta.json.
API keys come from the environment or .env (see .env.example).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from kg_neighbor import (
    KGNeighbors,
    kg_similarity,
    load_dataset,
    lt_2d,
    neighbor_retention,
    nmi_kmeans,
    sample_classes,
    umap_layout,
)
from kg_neighbor.utils.data import read_jsonl

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config" / "datasets.json").read_text(encoding="utf-8"))
MEASURES = ["jaccard", "overlap", "idf_cosine", "adamic_adar"]
ALPHAS = [round(0.1 * i, 1) for i in range(11)]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tag(alpha: float) -> str:
    return f"{alpha:g}"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


KG_NAME = "default"  # outputs of the dataset's one KG (kg/triples.tsv)


def emb_dir(args, name: str, enc: str) -> Path:
    root = args.emb_root if args.emb_root else args.results_root / name / "embeddings"
    return root / enc


def layout_dir(args, name: str, enc: str) -> Path:
    return args.results_root / name / "layouts" / enc / KG_NAME


def layout_file(layouts: Path, key: str) -> Path:
    return layouts / f"layout_{key}.npy"


def metrics_path(args, name: str, enc: str) -> Path:
    return args.results_root / name / "metrics" / enc / f"metric_{KG_NAME}.json"


def idp_dir(args, name: str) -> Path:
    return args.results_root / name / "idp"


def questions_path(args, name: str) -> Path:
    return idp_dir(args, name) / "idp_questions.json"


def judge_dir(args, name: str, model: str) -> Path:
    """What one judge model made: its batches, responses, judgments and diagnoses."""
    return idp_dir(args, name) / model


def idp_path(args, name: str, enc: str, model: str) -> Path:
    return judge_dir(args, name, model) / enc / f"idp_{KG_NAME}.json"


def plot_dir(args, name: str, enc: str) -> Path:
    return args.results_root / name / "plot" / enc / KG_NAME


def analysis_dir(args, name: str, enc: str) -> Path:
    return args.results_root / name / "analysis" / enc / KG_NAME


def load(args, name: str, enc: str):
    ds = load_dataset(name, enc, root=args.data_root, emb_dir=emb_dir(args, name, enc))
    return sample_classes(ds, args.n_classes, seed=args.class_seed)


# stages -----------------------------------------------------------------------
def preprocess(args, name: str) -> None:
    from kg_neighbor.utils import preprocess as pp

    out = args.data_root / name
    if name == "stackoverflow":
        n = pp.build_stackoverflow(out)
    elif name == "banking77":
        n = pp.build_banking77(out)
    elif name not in ("cub200", "flowers102"):
        raise SystemExit(f"[preprocess] no builder for {name!r}: prepare data/{name}/ separately")
    else:
        # Flowers 102: the archives have class numbers only
        names = pp.FLOWERS102_CLASSES if name == "flowers102" else None
        n = pp.build_reed(name, out, archive=args.reed_archive, class_names=names)
        if not args.skip_images:
            pp.fetch_images(name, out, archive=args.image_archive)
    log(f"[preprocess] {name}: {n} instances -> {out}")


def kg_collect(args, name: str) -> None:
    from kg_neighbor.utils.preprocess import collect_triples

    stats = collect_triples(args.data_root / name)
    log(f"[kg_collect] {name}: {stats}")


def embed_chunks(
    encode,
    texts: list[str],
    ids: list[str],
    parts: Path,
    *,
    chunk_size: int,
    settings: dict,
    overwrite: bool = False,
) -> np.ndarray:
    """Embed ``texts`` ``chunk_size`` at a time with ``encode``, saving each part in ``parts/``.

    A rerun reuses the saved parts, so an interrupted run resumes where it stopped. Parts
    made with other ``settings`` or holding other ids stop the run; ``overwrite`` recomputes
    every part instead.
    """
    parts.mkdir(parents=True, exist_ok=True)
    sig = parts / "settings.json"
    if sig.exists() and not overwrite:
        saved = json.loads(sig.read_text(encoding="utf-8"))
        if saved != settings:
            raise SystemExit(f"[embed] {parts} was made with {saved}, not {settings}: use --overwrite")
    write_json(sig, settings)
    n_parts = -(-len(texts) // chunk_size)
    out = []
    for k in range(n_parts):
        lo, hi = k * chunk_size, min((k + 1) * chunk_size, len(texts))
        part = parts / f"part_{k:05d}.npz"
        if part.exists() and not overwrite:
            saved = np.load(part)
            if saved["ids"].tolist() != list(ids[lo:hi]):
                raise SystemExit(f"[embed] {part} holds other ids: use --overwrite")
            out.append(saved["emb"])
            continue
        emb = np.asarray(encode(texts[lo:hi]), dtype=np.float32)
        tmp = parts / f"part_{k:05d}.tmp.npz"  # renamed when complete: no half-written part
        np.savez(tmp, emb=emb, ids=np.array(ids[lo:hi]))
        tmp.replace(part)
        out.append(emb)
        log(f"[embed] part {k + 1}/{n_parts} ({hi}/{len(texts)})")
    return np.concatenate(out)


def embed(args, name: str, enc: str) -> None:
    from kg_neighbor.utils.embed import embed_images, encode_texts, load_text_encoder, resolve_device

    cfg = CONFIG[name]["encoders"][enc]
    rows = read_jsonl(args.data_root / name / "texts" / "sentences.jsonl")
    device = resolve_device(args.device)
    out = emb_dir(args, name, enc)
    out.mkdir(parents=True, exist_ok=True)
    options = {k: cfg[k] for k in ("max_seq_length", "token_budget", "dtype", "chunk_size") if k in cfg}
    t0 = time.time()
    if CONFIG[name]["modality"] == "text":
        import torch

        dtype = torch.bfloat16 if device.startswith("cuda") else None
        if cfg.get("dtype"):  # optional, e.g. "bfloat16" on mps
            dtype = getattr(torch, cfg["dtype"])
        st = load_text_encoder(cfg["model"], device=device, dtype=dtype,
                               max_seq_length=cfg.get("max_seq_length"))

        def encode(texts: list[str]) -> np.ndarray:
            return encode_texts(st, texts, batch_size=cfg["batch_size"],
                                token_budget=cfg.get("token_budget"))

        texts = [r["text"] for r in rows]
        if cfg.get("chunk_size"):  # optional: save parts, resume after an interruption
            settings = {"model": cfg["model"], "batch_size": cfg["batch_size"], **options}
            Z = embed_chunks(encode, texts, [str(r["id"]) for r in rows], out / "parts",
                             chunk_size=cfg["chunk_size"], settings=settings,
                             overwrite=args.overwrite)
        else:
            Z = encode(texts)
    else:
        paths = [str(args.data_root / name / "images" / r["image_file"]) for r in rows]
        Z = embed_images(paths, cfg["model"], device=device, batch_size=cfg["batch_size"])
    np.save(out / "embedding_vectors.npy", Z)
    np.save(out / "embedding_labels.npy", np.array([r["class_id"] for r in rows], dtype=np.int64))
    np.save(out / "embedding_ids.npy", np.array([r["id"] for r in rows]))
    write_json(out / "embedding_meta.json", {"model": cfg["model"], "device": device, "n": len(Z),
                                   "dim": int(Z.shape[1]), "batch_size": cfg["batch_size"],
                                   **options, "seconds": round(time.time() - t0, 1)})
    log(f"[embed] {name}/{enc}: {Z.shape} -> {out}")


def run_kg_neighbor(args, name: str, enc: str) -> None:
    ds = load(args, name, enc)
    layouts = layout_dir(args, name, enc)
    layouts.mkdir(parents=True, exist_ok=True)
    out = metrics_path(args, name, enc)
    log(f"[kg_neighbor] {name}/{enc}: N={len(ds)}, classes={len(ds.class_names)}")

    def layout(nn, key: str) -> np.ndarray:
        f = layout_file(layouts, key)
        if f.exists() and not args.overwrite:
            return np.load(f)
        Y = umap_layout(ds.Z, nn, seed=args.seed)
        np.save(f, Y)
        return Y

    def scores(Y: np.ndarray) -> dict:
        return {"nmi": nmi_kmeans(Y, ds.labels, seed=args.seed),
                "lt_2d": lt_2d(Y, ds.labels, k=args.lt_k)}

    metrics = {"settings": {"k": args.k, "lt_k": args.lt_k, "n_classes": args.n_classes,
                            "class_seed": args.class_seed, "seed": args.seed,
                            "measures": args.measures, "alphas": args.alphas}}
    knn = KGNeighbors(args.k).fit(ds.Z)
    _, idx0 = knn.kneighbors()
    metrics["knn"] = scores(layout(knn, "knn"))
    log(f"  knn: {metrics['knn']}")
    for m in args.measures:
        S_kg = kg_similarity(ds.triples, m)
        metrics[m] = {"0": {**metrics["knn"], "retention": 1.0}}
        for a in (a for a in args.alphas if a > 0):
            nn = KGNeighbors(args.k, alpha=a, S_kg=S_kg).fit(ds.Z)
            _, idx = nn.kneighbors()
            metrics[m][tag(a)] = {**scores(layout(nn, f"{m}_a{tag(a)}")),
                                  "retention": neighbor_retention(idx, idx0)}
            write_json(out, metrics)
            log(f"  {m} a={tag(a)}: {metrics[m][tag(a)]}")
        del S_kg
    write_json(out, metrics)


def _figures(args, name: str, enc: str):
    """matplotlib (off-screen, paper fonts), the dataset and its saved layouts {key: file}."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kg_neighbor.utils.plot import use_fonts

    use_fonts()
    files = sorted(layout_dir(args, name, enc).glob("layout_*.npy"))
    if not files:
        raise FileNotFoundError(f"no layouts in {layout_dir(args, name, enc)}; run --kg_neighbor first")
    return plt, load(args, name, enc), {f.stem.removeprefix("layout_"): f for f in files}


def run_plot(args, name: str, enc: str) -> None:
    from kg_neighbor.utils.plot import save, scatter

    plt, ds, layouts = _figures(args, name, enc)
    out = plot_dir(args, name, enc)
    out.mkdir(parents=True, exist_ok=True)
    for key, f in layouts.items():
        title = "kNN" if key == "knn" else "KG-Neighbor, {}, α = {}".format(*key.rsplit("_a", 1))
        fig, ax = plt.subplots(figsize=(2.6, 2.6))
        scatter(np.load(f), ds.labels, ax=ax, title=title, class_names=ds.class_names)
        save(fig, out / f"plot_{key}.png")
        plt.close(fig)
    log(f"[plot] {name}/{enc}: {len(layouts)} layouts -> {out}")


def run_analysis(args, name: str, enc: str) -> None:
    from kg_neighbor.utils.plot import alpha_sweep, grid, retention, save

    plt, ds, layouts = _figures(args, name, enc)
    out = analysis_dir(args, name, enc)
    out.mkdir(parents=True, exist_ok=True)
    # kNN and three anchor guidance strengths side by side (paper Figure 3), overlap if computed
    measures = [m for m in MEASURES if any(k.startswith(f"{m}_a") for k in layouts)]
    if "knn" in layouts and measures:
        m = "overlap" if "overlap" in measures else measures[0]
        anchors = ["knn"] + [f"{m}_a{tag(a)}" for a in (0.3, 0.7, 1.0)]
        panels = [("kNN" if k == "knn" else f"α = {k.rsplit('_a', 1)[1]}", np.load(layouts[k]))
                  for k in anchors if k in layouts]
        fig = grid(panels, ds.labels, class_names=ds.class_names, suptitle=f"{name} · {enc} · KG-Neighbor ({m})")
        save(fig, out / "analysis_plot_grid.png")
        plt.close(fig)
    metrics = metrics_path(args, name, enc)
    if metrics.exists():
        scores = json.loads(metrics.read_text(encoding="utf-8"))
        for fname, draw in (("analysis_alpha_sweep.png", alpha_sweep), ("analysis_retention.png", retention)):
            fig = draw(scores, dataset=name, encoder=enc)
            save(fig, out / fname)
            plt.close(fig)
    log(f"[analysis] {name}/{enc} -> {out}")


def run_judge(folder: Path, judge, questions: list[dict], keys) -> dict[str, dict]:
    """Judgments in ``folder`` once its open batches are collected and the unanswered ``questions`` judged.

    idp_meta.json keeps the judge settings and, per batch, the job ids and the status
    (composed -> sent -> received, or failed), so a rerun picks up where the last one stopped
    and a question is sent once. ``keys`` are all question keys of the dataset.
    """
    meta_path, judgments_path = folder / "idp_meta.json", folder / "idp_judgments.json"
    settings = judge.settings()
    meta = read_json(meta_path, {**settings, "batches": {}})
    saved = {k: v for k, v in meta.items() if k != "batches"}
    if saved != settings:
        raise SystemExit(f"[idp] {folder} holds judgments made with {saved}, not {settings}")
    judgments = read_json(judgments_path, {})
    ids = {judge.request_id(k): k for k in keys}

    def finish(name: str) -> None:
        batch = meta["batches"][name]
        if batch["status"] == "composed":
            batch.update(job_ids=judge.submit(read_jsonl(folder / "batches" / f"{name}.jsonl")), status="sent")
            write_json(meta_path, meta)
            log(f"  {name}: {batch['n_requests']} requests sent, jobs {batch['job_ids']}")
        try:
            rows = judge.wait(batch["job_ids"])
        except RuntimeError:  # the job itself failed: its questions go into a new batch on the next run
            batch["status"] = "failed"
            write_json(meta_path, meta)
            raise
        write_jsonl(folder / "responses" / f"{name}.jsonl", rows)
        judgments.update({ids[i]: a for i, a in judge.parse(rows).items()})
        write_json(judgments_path, judgments)
        batch.update(n_responses=len(rows), status="received")
        write_json(meta_path, meta)
        log(f"  {name}: {len(rows)} responses received")

    for name, batch in meta["batches"].items():
        if batch["status"] in ("composed", "sent"):
            finish(name)
    todo = [q for q in questions if q["key"] not in judgments]
    if todo:
        name = f"batch_{len(meta['batches']) + 1:04d}"
        write_jsonl(folder / "batches" / f"{name}.jsonl", [judge.build_request(q) for q in todo])
        meta["batches"][name] = {"job_ids": None, "n_requests": len(todo), "n_responses": 0, "status": "composed"}
        write_json(meta_path, meta)
        finish(name)
    return judgments


def run_idp(args, name: str, enc: str) -> None:
    from kg_neighbor.idp import IDP, AnthropicBatch, OpenAIBatch

    ds = load(args, name, enc)
    layouts = layout_dir(args, name, enc)
    keys = ["knn"] + [f"{m}_a{tag(a)}" for m in args.idp_measures for a in args.alphas if a > 0]
    missing = [k for k in keys if not layout_file(layouts, k).exists()]
    if missing:
        raise FileNotFoundError(f"layouts missing {missing[:3]}; run --kg_neighbor first")

    idp = IDP(k=args.idp_k)
    mismatches = {k: idp.detect(ds, np.load(layout_file(layouts, k))) for k in keys}
    questions = {q["key"]: q for mm in mismatches.values() for q in idp.questions(ds, mm)}
    listed = read_json(questions_path(args, name), {})  # every question the dataset's layouts have needed
    listed.update({k: {"sample_id": q["sample_id"], "class_name": q["class_name"]} for k, q in questions.items()})
    write_json(questions_path(args, name), listed)

    judge = {"openai": OpenAIBatch, "anthropic": AnthropicBatch}[args.judge]
    judge = judge(args.judge_model) if args.judge_model else judge()
    folder = judge_dir(args, name, judge.model)
    judgments = read_json(folder / "idp_judgments.json", {})
    batches = read_json(folder / "idp_meta.json", {"batches": {}})["batches"]
    n_open = sum(b["status"] in ("composed", "sent") for b in batches.values())
    n_new = sum(k not in judgments for k in questions)
    log(f"[idp] {name}/{enc} ({judge.model}): {len(keys)} layouts, {len(questions)} questions, "
        f"{len(questions) - n_new} judged, {n_new} new, {n_open} open batches")
    if n_new or n_open:
        if args.dry_run:
            log("  --dry_run: judge not called")
            return
        judgments = run_judge(folder, judge, list(questions.values()), listed)
    write_json(idp_path(args, name, enc, judge.model),
               {k: idp.diagnose(ds, mm, judgments) for k, mm in mismatches.items()})
    for k in keys:
        log(f"  {k}: {len(mismatches[k]['idx'])} mismatches")


# CLI -------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = p.add_argument_group("stages")
    stages.add_argument("--preprocess", action="store_true",
                        help="raw files placed under data/<dataset>/raw/ -> data/<dataset>/")
    stages.add_argument("--kg_collect", action="store_true",
                        help="per-instance .triples from kg_extractor -> kg/triples.tsv")
    stages.add_argument("--embed", action="store_true", help="encode texts or images")
    stages.add_argument("--kg_neighbor", action="store_true",
                        help="measure x alpha sweep: layouts, metrics and their plots")
    stages.add_argument("--plot", action="store_true",
                        help="redraw plot/ (each layout) and analysis/ (grid, alpha curves, retention) from saved "
                             "layouts and metrics")
    stages.add_argument("--idp", action="store_true", help="diagnose residual mismatches")

    p.add_argument("--dataset", required=True, choices=sorted(CONFIG))
    p.add_argument("--encoder", nargs="+", help="default: both encoders of the dataset")
    p.add_argument("--data_root", type=Path, default=ROOT / "data")
    p.add_argument("--emb_root", type=Path, help="default: <results_root>/<dataset>/embeddings")
    p.add_argument("--results_root", type=Path, default=ROOT / "results")

    g = p.add_argument_group("preprocess")
    g.add_argument("--reed_archive", type=Path, help="Reed caption archive at another path than raw/")
    g.add_argument("--image_archive", type=Path, help="image archive at another path than raw/")
    g.add_argument("--skip_images", action="store_true")

    g = p.add_argument_group("embed")
    g.add_argument("--device", default="auto")

    g = p.add_argument_group("kg_neighbor (defaults = paper settings)")
    g.add_argument("--measures", nargs="+", default=MEASURES, choices=MEASURES)
    g.add_argument("--alphas", nargs="+", type=float, default=ALPHAS)
    g.add_argument("--k", type=int, default=15, help="neighbors of the graph given to UMAP")
    g.add_argument("--lt_k", type=int, default=20, help="neighbors for LT_2D")
    g.add_argument("--n_classes", type=int, default=20)
    g.add_argument("--class_seed", type=int, default=0)
    g.add_argument("--seed", type=int, default=0, help="UMAP and k-means seed")
    g.add_argument("--overwrite", action="store_true",
                   help="recompute saved layouts and embedding parts")

    g = p.add_argument_group("idp")
    g.add_argument("--idp_measures", nargs="+", default=["overlap"], choices=MEASURES)
    g.add_argument("--idp_k", type=int, default=15, help="neighbors for mismatch detection")
    g.add_argument("--dry_run", action="store_true", help="count new questions without calling the judge")
    g.add_argument("--judge", choices=["openai", "anthropic"], default="openai",
                   help="openai: GPT-5.4 (paper), anthropic: Claude Opus 5.5; each model has its own idp/<model>/ folder")
    g.add_argument("--judge_model", help="override the judge model")

    args = p.parse_args(argv)
    if not (args.preprocess or args.kg_collect or args.embed or args.kg_neighbor or args.plot or args.idp):
        p.error("give at least one stage: --preprocess --kg_collect --embed --kg_neighbor --plot --idp")
    args.alphas = sorted(set(round(a, 6) for a in args.alphas))
    args.encoder = args.encoder or list(CONFIG[args.dataset]["encoders"])
    unknown = set(args.encoder) - set(CONFIG[args.dataset]["encoders"])
    if unknown:
        p.error(f"unknown encoder for {args.dataset}: {sorted(unknown)}")
    return args


def load_env(path: Path) -> None:
    """``KEY=VALUE`` lines of ``path`` into the environment; values already set and empty values are skipped.

    Values are never logged.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip().strip("'\"")
        if value and not os.environ.get(key):
            os.environ[key] = value


def main(argv=None) -> None:
    load_env(Path(__file__).resolve().parent / ".env")
    args = parse_args(argv)
    name = args.dataset
    if args.preprocess:
        preprocess(args, name)
    if args.kg_collect:
        kg_collect(args, name)
    for enc in args.encoder:
        if args.embed:
            embed(args, name, enc)
        if args.kg_neighbor:
            run_kg_neighbor(args, name, enc)
        if args.kg_neighbor or args.plot:  # plots follow the layouts
            try:
                run_plot(args, name, enc)
                run_analysis(args, name, enc)
            except ImportError as e:
                if args.plot:
                    raise
                log(f"[plot] skipped ({e}); pip install -e '.[plot]' and rerun with --plot")
        if args.idp:
            run_idp(args, name, enc)


if __name__ == "__main__":
    main()
