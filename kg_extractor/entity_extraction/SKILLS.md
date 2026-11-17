# SKILLS — Entity Extraction (Pass 1 of 3)

## Purpose

From the input text + domain spec, produce a candidate **entity list** —
each entity with its canonical name, type, and the surface evidence that
warranted extracting it. NO predicates / triples yet — that is Pass 2.

This pass exists because mixing entity recognition with relation linking
in a single pass degrades both. Get the entities right first.

## Inputs

- The document `text`
- The `domain_spec` (entity types, naming convention, canonical examples)

## Output (internal, not emitted)

A working list, one row per entity:

```
canonical_name | type | surface evidence span(s)
```

Example for a scientific paper:

```
Visual_Speech_Recognition | Research_Task   | "visual speech recognition"
LSTM_Network              | Neural_Network_Component | "LSTM", "Long-Short Memory"
OuluVS2                   | Visual_Speech_Recognition_Dataset | "OuluVS2"
End_to_End_VSR_System     | Proposed_System | "end-to-end visual speech recognition system"
```

## Procedure

1. **Read the domain spec's canonical examples first.** Skim their entities
   to calibrate your *naming convention* (TitleCase_With_Underscores vs
   CamelCase), *granularity* (do datasets get a Subtype suffix?), and
   *typical density* (how many entities per doc).
2. **Walk the text top-to-bottom.** For each sentence:
   a. List every concrete noun phrase that names a real thing — task,
      method, dataset, person, organization, location, event, product,
      concept.
   b. Skip pronouns and the document author / narrator. Resolve coreference
      now: `the firm`, `he`, `she`, `it`, `the dataset` → the canonical
      entity that was most recently named.
   c. Skip generic boilerplate (`the model`, `the system`, `a number of
      methods`) UNLESS the boilerplate has no replacement and the entity
      is load-bearing.
3. **Assign each entity a type** from the domain spec's `entity_types`
   list. Prefer the most specific applicable type.
   - If no type fits, mark as `UNKNOWN_TYPE` and continue. Pass 3 will
     surface this.
4. **Apply the canonical naming convention**:
   - Use the dominant acronym if it is universally standard in the
     subfield (`LSTM`, `MIMO`, `BER`). Spell out paper-coined acronyms.
   - Strip suffixes (`Inc`, `Ltd`, `Corp`) unless part of the brand.
   - Strip honorifics (`Mr`, `Mrs`, `Dr`).
   - Numbers NEVER appear inside entity names — they belong in
     reification nodes (Pass 2 handles that).
5. **Deduplicate within the document.** Two surface forms that refer to
   the same real entity collapse to one row with merged evidence.
6. **Order the list by appearance** in the text — Pass 2 walks it
   sequentially.

## Heuristics

| When… | Do this |
|---|---|
| Two surface forms differ in case / hyphenation only | Collapse to the canonical form |
| An entity is mentioned with a role suffix ("Tony Blair, the prime minister") | Two entities: `Tony_Blair` (Person) AND `Prime_Minister` (Role) — link them via Pass 2 |
| A number is part of a product name (`iPhone 4`, `Windows XP`) | Number is part of the entity string |
| A number is a value (`£300m`, `3.4%`, `100 cafés`) | DO NOT create an entity for the number now; mark "evidence pending Pass-2 reification" |
| A general topic is mentioned without a named anchor ("climate change", "the internet") | Topic entity is fine if the doc references it more than once OR the topic is the document's main subject |
| A datum / event lacks a clear actor (passive constructions) | Skip if no actor is recoverable; surface fidelity matters |

## Reification-aware extraction

Some entities exist ONLY as reification nodes (Evaluation_Result, Event,
Quantity). Examples:

- A reported number like "£300m valuation of Deluxe Films" implies a
  `Deluxe_Film_Valuation` Quantity node.
- "Microsoft acquired Nokia" implies a `Microsoft_Nokia_Acquisition` Event
  node.

**Pre-name these reification nodes in this pass** using the convention
`<Actor>_<EventType>` or `<Entity>_<Aspect>`. Pass 2 will fill in their
attributes. If you don't pre-name them now, Pass 2 will, but consistency
slips.

## Quality checklist

- [ ] Every entity has a canonical name + a type (or `UNKNOWN_TYPE`).
- [ ] No pronouns or generic boilerplate in the list.
- [ ] Naming convention matches the domain spec.
- [ ] Title entity / main subject is included.
- [ ] Reification nodes have been pre-named where evidence is clear.
- [ ] The list is deduplicated.

## Hand-off

The deduplicated entity list is the input to Pass 2 (Relation Linking).
Carry it forward verbatim; do not lose evidence spans — Pass 3 uses them
for surface-fidelity check.
