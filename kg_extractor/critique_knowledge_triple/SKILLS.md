# SKILLS — Critique Knowledge Triples (Pass 3 of 3)

## Purpose

Final adversarial pass. Given the **candidate triples from Pass 2** + the
original text + the domain spec, verify every triple, fix what's
fixable, drop what isn't, and add a SMALL number of critical missing
triples. The output of this pass IS the agent's final output.

This pass exists because the first two passes optimize for *coverage*,
which makes them prone to over-extraction, surface-form drift, and
vocab leakage. Critique optimizes for *precision* and *canonicalization*.

This is ONE-SHOT. You do not bounce back to Pass 1 or Pass 2; you produce
the final corrected list directly.

## Inputs

- The document `text`
- The candidate triple list from Pass 2
- The entity list from Pass 1 (for evidence cross-check)
- The `domain_spec` (entity types, predicate vocab, quality checklist,
  constraints)

## Output (final, emitted to the user)

```
(Subject, predicate, Object)
(Subject, predicate, Object)
...
# critique: <one-line note per substantive change> (optional)
# new_predicate <name>: <reason> (optional, from Pass 2; keep if still warranted)
```

The post-critique triples are the **only** thing the downstream parser
sees. Working notes stay as `# critique: ...` comments.

## The four critique dimensions

For each candidate triple, apply the four checks in order. The first one
to fail decides the action.

### 1. Surface fidelity — is this triple supported by the text?

- The relation must trace to an explicit text span (the Pass-1 evidence
  list is your reference).
- Common knowledge that the doc DOES NOT state → **DROP**. Example: in a
  paper about LSTM for VSR, do not emit `(LSTM_Network, is_a,
  Recurrent_Neural_Network)` unless the text says so.
- Hallucinated entities (entity not in Pass-1 list) → **DROP**.
- Predicate is paraphrasing the text but loses fidelity ("LSTM beats
  baseline" → `(LSTM, defeats, Baseline)` when the text said "achieves
  +0.6% over baseline") → **REWRITE** as the canonical reification.

### 2. Canonical-form discipline — is naming consistent?

- Entity string differs from its canonical form → **RENAME**.
  Example: `(LSTMs, uses, …)` → `(LSTM_Network, uses, …)`.
- Predicate is a synonym not in vocab → **REWRITE** with the canonical
  predicate (`employs` → `uses`).
- Person name uses an honorific or role suffix → strip
  (`Prime_Minister_Tony_Blair` → `Tony_Blair` + a separate role triple
  if not already present).
- Duplicate triple after renaming → **DROP** the duplicate.

### 3. Closed-vocabulary discipline — is type / predicate in the spec?

- Type label is not in `entity_types` → if the closest canonical type
  exists, **RENAME** the type; if none fits, **MARK** with
  `# critique: unknown type <X>` and emit a `(Entity, type, UNKNOWN)`
  triple OR drop the type triple, per domain spec.
- Predicate is not in `predicate_vocab` → if the relation is real,
  **REWRITE** with the closest match; if the relation is novel,
  **KEEP** the triple AND append a `# new_predicate <name>: <reason>`
  line.

### 4. Completeness — is anything critical missing?

A short positive-check list — ADD a missing triple only if obviously
required by the spec:

- **Title task / main subject** must connect to the rest of the graph.
  If isolated → ADD the missing link.
- **Reification nodes** must have all their attribute triples per the
  spec. Missing `has_value` for an Evaluation node → ADD if the number
  appears in the text.
- **Event-centric domains** (`require_type_triples = false`): every
  utterance must have one canonical Event hub + at least one User-to-Event
  predicate + at least one `hasStatus` (or `hasOccurrenceCount` /
  `hasIssueType`). ADD if missing.

Do NOT add triples beyond this short list. Critique is precision-first.

## Procedure

1. **Load the spec's quality checklist.** It is your final acceptance test.
2. **Walk the candidate triple list once**, applying the four checks per
   triple. Maintain an in-place corrected list.
3. **Walk the spec's mandatory-element list** and ADD missing items only
   if Section 4 above warrants it.
4. **Sort** the final list into the spec's canonical emission order
   (entity types → top-level wiring → reification attributes → cross-entity
   relations).
5. **Verify constraints**: total triple count within
   `constraints.min_triples` … `max_triples`. If under min, you missed
   coverage (only legitimate fix: re-walk text for clear omissions); if
   over max, drop the LEAST informative triples (boilerplate
   self-types, redundant restatements).
6. **Emit** the final triple list + optional `# critique:` /
   `# new_predicate` lines.

## Decision rubric — DROP vs REWRITE vs KEEP

| Situation | Action |
|---|---|
| Triple has no text evidence | DROP |
| Triple's evidence is real but predicate is a synonym | REWRITE predicate |
| Triple's evidence is real but subject/object is a surface variant | RENAME entity |
| Triple is duplicated after rename/rewrite | DROP duplicate |
| Triple uses an unknown type/predicate AND no canonical equivalent | KEEP + annotate (`# new_predicate` or `# critique: unknown type`) |
| Triple is technically correct but trivially restates another triple | DROP the weaker one |
| Triple is missing but mandated by the spec (title task, event hub, reification attribute) | ADD |

## Quality checklist (final acceptance)

- [ ] Every emitted triple matches `^\((.+?), (.+?), (.+?)\)$`.
- [ ] Every named entity has a `type` triple (unless the spec disables them).
- [ ] All predicates are in the domain vocab OR annotated as `# new_predicate`.
- [ ] All entity strings are in canonical form.
- [ ] No duplicate triples.
- [ ] Mandatory elements present per spec (title task, event hub, reification
      attribute completeness).
- [ ] Triple count within constraint range.
- [ ] No prose, no markdown, no JSON — only triple lines and optional
      `# critique` / `# new_predicate` comments.

## Output discipline

The user / parser only sees:

```
(triple lines...)
# critique: optional one-line notes
# new_predicate: optional declarations
```

No JSON wrapper, no preamble, no "Here is the result" text. The downstream
W_kg builder reads this file directly and tolerates `#`-prefixed lines as
comments.
