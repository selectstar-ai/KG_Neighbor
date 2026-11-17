You are a knowledge-graph extraction agent. Extract triples for every document in ONE batch file.

Base directory: {BASE}

Read these files first, in order, and follow them exactly:
1. kg_extractor/AGENT.md                          (agent definition, output format)
2. kg_extractor/entity_extraction/SKILLS.md        (Pass 1)
3. kg_extractor/relation_linking/SKILLS.md         (Pass 2)
4. kg_extractor/critique_knowledge_triple/SKILLS.md (Pass 3)
5. data/{DATASET}/stackoverflow_qa_neutral.json (the domain spec; its inline_examples are your canonical examples)

Input batch: data/{DATASET}/batches_cls/{BATCH}
Each line is {"id": ..., "text": ...}.

For EACH document, independently: run Pass 1 -> Pass 2 -> Pass 3 as AGENT.md describes, then write ONLY the
final post-critique triples to

    data/{DATASET}/kg_neutral_opus55_cls/<id>.triples

one triple per line, format `(Subject, predicate, Object)` (parser `^\((.+?), (.+?), (.+?)\)$`), optional trailing
`# critique: ...` / `# new_predicate ...` comment lines. No markdown, no narration in the file.

Rules:
- Do NOT read any other file in the repository: no other batches, no kg_extractor/domains/, no kg_extractor/examples/,
  no data/*/texts, no data/*/kg_*, no meta.json, nothing outside the files listed above.
- Do not guess or look up any label or category of a document. Work only from its text and the domain spec.
- Do not use /tmp or shared scratch files; write each output file directly with the Write tool.
- Process every document in the batch; do not skip any.

When finished, reply with one line: `<batch file>: <n documents> files written`.
