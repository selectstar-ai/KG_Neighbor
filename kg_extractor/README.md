# Knowledge Graph Construction

Each instance gets its own knowledge graph, extracted from its content. The input
record given to the extractor holds only the id and the text of the instance; it has
no label field. The wording of the schema and the way the extraction is run can still
carry label cues: [Robustness ablation](#robustness-ablation) lists them and describes
the experiment that varies them.

## Three passes

The orchestrator ([`AGENT.md`](AGENT.md)) runs three passes in order. Each pass has its own
specification, in the directory layout `AGENT.md` refers to:

| Pass | Specification | Produces |
|---|---|---|
| 1. Entity extraction | [`entity_extraction/SKILLS.md`](entity_extraction/SKILLS.md) | Canonical entity names and types |
| 2. Relation linking | [`relation_linking/SKILLS.md`](relation_linking/SKILLS.md) | Candidate triples over the domain predicate vocabulary |
| 3. Critique | [`critique_knowledge_triple/SKILLS.md`](critique_knowledge_triple/SKILLS.md) | Verified triples — violations fixed or dropped, missing core triples added |

The released output is the post-critique triple set.

## Why canonicalization is the load-bearing part

Cross-instance KG similarity is computed by set overlap over triples. Two instances
about the same thing therefore have to produce *the same strings*. If one document
yields `SVN` and another `Subversion`, the two share nothing and the similarity is
zero regardless of how well each graph describes its own document. The naming rules
in the domain schema and the canonicalization checks in pass 3 exist for this
reason.

## Domain schemas

One per corpus, in [`domains/`](domains/). The schema carries everything
domain-specific so the three skill specifications stay domain-neutral: entity and
predicate naming rules, the closed predicate vocabulary, entity types,
canonicalization hints, worked examples, and constraints.

| Schema | Corpus |
|---|---|
| `stackoverflow_qa.json` | StackOverflow |
| `banking_intent.json` | BANKING77 |
| `flowers_captions.json` | Flowers 102 |
| `birds_captions.json` | CUB-200-2011 |

For the two image corpora the extractor reads human-written visual descriptions
rather than the image itself, so the KG is built from a linguistic description of
image content and does not reuse the image embedding's internal structure.

## Output format

Per document, one triple per line:

```
(Subject, predicate, Object)
```

These are concatenated into a corpus-level 4-column TSV keyed by instance id:

```
so_00000	Question	type	Artifact
so_00000	Question	about_technology	LINQ
so_00000	LINQ	type	Technology
```

## Worked examples

[`examples/`](examples/) holds three extractions.

`so_11794.triples` and `so_15035.triples` are the StackOverflow instances that
appear in Table 2 of the paper. The first is shown below:

```
Instance: "Trying to get rsync to copy files back to svn"
Label:    osx

(Question, type, Artifact)
(Question, about_technology, Rsync)
(Rsync, type, Tool)
(Question, performs_operation, Copy)
(Rsync, integrates_with, SVN)
(SVN, type, Tool)
```

The paper reports this instance as a Label Issue in the Qwen3-Embedding-8B layout at
`α = 0.1`: all 15 of its neighbors are labeled `svn`, and the judge finds that the
content supports `svn` but not `osx`.

`n_0059155.*` is a generic example from the agent's own specification, retained
because `AGENT.md` refers to it.

## Scale

Extraction was run over the full corpus for each dataset. The evaluation subset
(20 classes, sampled per the paper) is a subset of that:

| Dataset | Documents extracted | Triples in eval subset | Per instance | Distinct predicates |
|---|---:|---:|---:|---:|
| StackOverflow | 20,000 | 122,625 | 6.1 | 9 |
| BANKING77 | 13,083 | 20,832 | 6.4 | 10 |
| Flowers 102 | 8,189 | 21,950 | 15.9 | 6 |
| CUB-200-2011 | 11,788 | 26,904 | 22.8 | 5 |

Every instance in every dataset received at least one triple.

## Robustness ablation

The results in the paper come from one KG. A KG extracted by an LLM depends on the
extraction model, the prompt, and the way the extraction is run, so this experiment
checks whether the gains of KG-Neighbor hold when the KG is extracted differently.
3,000 StackOverflow questions (150 per class) were extracted again while varying three
factors, which gives 2 × 2 × 3 = 12 KGs.

| Factor | Levels | What it checks |
|---|---|---|
| Extraction model | Claude Opus 4.8 (paper), Claude Opus 5.5 | Dependence on one LLM |
| Prompt | original schema, neutral schema | Label cues in the wording of the prompt |
| Batch execution | R (random), C (per class), X (per class, class name exposed) | Label cues from the way the extraction is run |

- **Prompt.** The original StackOverflow schema says that the class is a technology tag,
  asks that questions about the same technology collapse to an overlapping KG, and names
  five of the 20 classes (spring, hibernate, scala, oracle, linq) in its rules and
  examples. The neutral schema changes only this wording; the predicates, the output
  format, and the `about_technology` rule stay the same.
- **Batch execution.** In the extraction for the paper, one batch held 50 questions of one
  class, and the batch file name and the output path contained the class name (level X).
  Level C keeps the same batches but hides the class name and tells the agent not to
  guess labels. Level R also mixes the classes within a batch.
- At every level the input record given to the extractor holds only the id and the title
  of the question; it has no label field.

The files of this experiment are in [`ablation/`](ablation/):

| File | Content |
|---|---|
| `stackoverflow_qa_neutral.json` | Neutral schema. The original schema is `domains/stackoverflow_qa.json` |
| `DISPATCH_<condition>.md` | Instruction given to the extraction agent in each of the 11 re-extracted conditions |

A condition name joins the prompt (O original, N neutral), the model (48 Claude Opus 4.8, 55 Claude Opus
5.5), and the batch-execution level (R, C, X). O48X is the KG of the paper and was not extracted again.
In the instructions, `{BASE}`, `{DATASET}`, and `{BATCH}` stand for the repository path, the data folder,
and the batch file, and `stackoverflow_qa_original.json` is a copy of `domains/stackoverflow_qa.json`.

See the supplementary material of the paper for details.
