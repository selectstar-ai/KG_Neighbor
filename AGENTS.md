# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this repository is

Code for the paper *KG-Neighbor: Knowledge-Guided Neighborhood Construction for Embedding Visualization*
(WSDM 2027 Findings). `run_pipeline.py` runs the stages in this order:
preprocess → embed → kg_neighbor (with plots) → idp. Usage is described in `README.md`.

## Layout

```
run_pipeline.py          command-line entry point, one function per stage
config/datasets.json     datasets and their encoders
kg_neighbor/
├── similarity.py        KG similarity between instance-specific triple sets
├── neighbors.py         knowledge-guided distance and top-k neighbor selection
├── layout.py            UMAP layout from a given neighbor search
├── metrics.py           NMI, LT_2D, Neighbor Retention
├── idp/                 residual mismatch detection, judge requests, diagnostic states
└── utils/               data loading, preprocessing, encoders, plots, class sampling
kg_extractor/            prompts and schemas used to construct the KGs (not Python code)
tests/                   pytest tests
docs/assets/             figures of the paper (the READMEs use 01 and 02)
data/, results/          inputs and outputs, not tracked
```

## Commands

```
pip install -e ".[all]"          # embed, plot, llm extras
pip install -e ".[dev]"          # pytest
pytest                           # run before finishing any change
python run_pipeline.py --help    # all stages and options
```

## Rules

- **No downloads in preprocessing.** `--preprocess` only converts the source files that the user placed
  under `data/<dataset>/raw/`. A missing file must stop with a message that names the file and where to
  get it (`SOURCES` in `kg_neighbor/utils/preprocess.py`).
- **Instance ids and class indices are fixed.** The KG triples are keyed by instance id, and the 20
  evaluated classes are sampled by class index. Do not change the order in which preprocessing reads
  instances or numbers classes.
- **Defaults are the settings used in the paper.** Four KG similarity measures, `α ∈ {0, 0.1, …, 1}`,
  `--k 15`, `--lt_k 20`, `--n_classes 20`, `--class_seed 0`, `--seed 0`. Add an option instead of changing
  a default.
- **The IDP judge is a paid API.** Run `--idp` with `--dry_run` first and do not send requests unless the
  user asks for it. Judgments are stored under `results/<dataset>/idp/` and are never requested twice.
- **Secrets stay local.** API keys are read from `.env` (see `.env.example`). Never print or commit `.env`.
- **Do not commit `data/` or `results/`.** They are ignored. The only tracked files under `data/` are the
  released KG triples (`data/<dataset>/kg/triples.tar.gz`). `docs/` is ignored as well, and only the
  figures in `docs/assets/` are tracked.
- **Optional dependencies are imported inside functions** (torch, sentence-transformers, matplotlib,
  openai, anthropic), so that the core modules run with the base dependencies. Keep it that way.
- **The prompts and schemas in `kg_extractor/` are a record.** `AGENT.md`, the `SKILLS.md` files and
  `domains/*.json` are the ones used to construct the KGs of the paper. Do not edit them to improve the
  extraction.
- **Tests.** Every change in behavior comes with a simple test in `tests/`. Tests must not need the
  network or a paid API.

## Documentation

- `README.md` and `README_ko.md` have the same sections, code blocks and links. Update both together.
- Use the terms of the paper: instance content, relational knowledge, KG similarity, guidance strength,
  residual mismatch, assigned class, neighborhood-majority class, and the four diagnostic states
  (Placement Issue, Label Issue, Boundary Case, Unresolved Case).
- Neighborhood construction does not use class labels. Labels serve as the evaluation reference and as
  input to the IDP diagnosis.
- Module docstrings cite the sections and equations of the paper. Keep them in line with the code.
