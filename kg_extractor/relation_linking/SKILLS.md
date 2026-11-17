# SKILLS — Relation Linking (Pass 2 of 3)

## Purpose

Given the **entity list from Pass 1** + the original text + the domain
predicate vocabulary, produce the **candidate triples**. This is where
relations are picked, reification nodes are filled out, and `type`
triples are emitted.

This pass does NOT verify against the text — that is Pass 3's job. Your
goal here is *complete and structured*; let critique police *correct*.

## Inputs

- The document `text`
- The entity list `[(canonical_name, type, evidence)]` from Pass 1
- The `domain_spec` (`predicate_vocab`, `reification.patterns`)

## Output (internal)

A flat triple list, plus running notes if you had to invent a predicate:

```
(Subject, predicate, Object)
(Subject, predicate, Object)
...
# new_predicate <name>: <reason>
```

## Procedure

1. **Re-read the domain spec's canonical examples.** Note the typical
   structures — how `Proposed_System` is linked to `Research_Task` (via
   `performs`), how `<Dataset>_Evaluation` reification nodes attach
   `evaluated_on` / `has_value` / `has_unit`, etc. Replicate those
   patterns.
2. **Emit `type` triples** for every entity in the Pass-1 list (unless the
   domain spec says `require_type_triples = false`, e.g. the event-centric
   banking domain). One `(Entity, type, Type)` per entity. Skip bare
   numbers and the pre-named reification nodes — those get
   `(<Node>, type, <Reification_Type>)` from the reification template.
3. **Walk the text again, sentence by sentence**, this time linking
   entity pairs:
   a. For each verb / preposition / "of" / "by" construct that ties two
      Pass-1 entities, pick the **closest matching predicate** from
      `predicate_vocab`. Do not invent synonyms — `uses`, not
      `employs` / `leverages` / `utilizes`.
   b. Multi-entity sentences may produce multiple triples; emit them all.
   c. Negations / hedges ("did not", "may not") usually mean DO NOT emit
      the positive triple. If the negation IS the knowledge, the domain
      spec must have a predicate for it.
4. **Fill in reification nodes.** Every reification node pre-named in
   Pass 1 needs its attribute triples from the spec's reification template.
   Walk the relevant sentence and emit one triple per attribute.
   - For an Evaluation node: `evaluated_on`, `compared_against`,
     `reports_change`, `has_value`, `has_unit`.
   - For an Event node: `event_type`, `has_actor`, `has_target`,
     `on_date`, `at_venue`, `has_outcome`, `has_amount`.
   - For a Quantity node: `has_value`, `has_unit`, `of_entity`,
     `on_period`, `change_from`.
5. **Wire reification nodes back to the main entity** via the spec's
   linking predicate (`has_evaluation`, `has_amount`, `has_status`, etc.).
6. **Stop at the doc's stated content.** Common knowledge ("the UK is in
   Europe") should NOT be added — it isn't grounded in the text.

## Predicate-selection heuristics

| Surface pattern | Predicate (default) |
|---|---|
| "X is built on Y" / "X uses Y" / "X is based on Y" | `uses` |
| "X is applied to Y" / "X performs Y" | `applied_to` / `performs` |
| "X achieves Z% on Y" | reification: `<X>_<Y>_Evaluation` + `has_value` |
| "X acquired Y" (completed deal) | `acquired` (subject = buyer) |
| "X met with Y" | `met_with` (symmetric — emit once) |
| "X owns Y" / "X is owned by Y" | `owns` (subject = owner) |
| "X is in Y" (location) | `located_in` |
| "X scored N" / "X scored against Y" | `scored` |
| "X starred in Y" | `starred_in` |
| "User waiting for X" / "X has not arrived" (banking-style) | `waitsFor` + `<Event>` reification |

When two predicates feel applicable, **pick the more specific one** from
the vocab.

## Reification ordering

Emit triples in this canonical order — Pass 3's critique scans more
cleanly when shape is uniform:

```
1. All entity type triples
2. Top-level (main entity → reification node) wiring
3. Reification node attribute triples (grouped per node)
4. Cross-entity relation triples
```

(For tight event-centric domains where `require_type_triples = false`,
skip step 1 and order is: top-level wiring → state attributes →
differentiator triples.)

## Quality checklist

- [ ] Every triple uses a predicate from the domain vocab (or has a
      trailing `# new_predicate ...` line with the reason).
- [ ] Every reification node from Pass 1 has at least one attribute triple
      AND a linking triple back to the main entity.
- [ ] No predicate synonyms (one canonical predicate per relation type).
- [ ] No duplicate triples.
- [ ] Title task / main subject is connected to the rest of the graph (not
      isolated).
- [ ] Triple count is within the domain spec's `constraints.min_triples` /
      `max_triples` range (rough guide).

## Hand-off

The triple list is the input to Pass 3 (Critique). Carry the entity list
and text along — Pass 3 needs both to verify surface fidelity.
