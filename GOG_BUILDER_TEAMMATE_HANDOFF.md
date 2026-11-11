# GoG builder and evaluation handoff

This handoff concerns `primekg_llm_gog_colab_code_updated.py` (the Colab export supplied by the project owner), the current GoG model in `gog_model/`, and the training code in `training/`. The objective is a leakage-audited comparison of a flat KG, a flat context-aware model, and the GoG encoder. The existing MRR reports are development results, not the benchmark for this comparison.

## 1. What the current builder actually does

- It reads all of PrimeKG `kg.csv` into one edge array and builds adjacency from all retained edges (builder lines 362-426). There are no train/validation/test labels at this point.
- For a drug, gene/protein, or pathway owner, it retrieves up to 15 one-hop and 15 two-hop candidates from that adjacency (469-503), gives the LLM selected node and edge descriptions (518-585), then chooses up to 20 nodes and 40 edges for an internal graph (601-648). The LLM ranks candidates from PrimeKG; it does not establish that its selections are independent new facts.
- Each internal member also receives an occurrence in ROOT (709-722). This is intentional shared entity identity, **not graph nesting**. An owner entity appearing inside another internal graph does not recursively expand its own graph there. Only ROOT plus one level of internal graphs is modeled.
- The builder outputs ROOT relations, selected internal relations, and extra cross-graph relations (725-755). For every ROOT relation `(A,r,B)`, lines 741-754 iterate over internal memberships of A and B and create `(A@G1,r,B@G2)` for different graphs. The source did not assert those particular contextual endpoints. These are derived copies, not independent observed cross-context facts.
- The builder exports no stable source-row/fact ID, source-to-derived-record mapping, split manifest, graph-membership provenance, dataset fingerprint, or prompt/version/response record. `llm_reason` is an explanation, not provenance.
- Configured limits currently allow at most 25 graphs per owner type, 20 nodes per graph, 15 candidates per hop, and 30 candidate edges per owner. `QUICK_TEST=True` and `False` currently use the same graph limits (109-112). The script does not measure input/output token use or cache completed owner decisions. The exported `.py` includes notebook-only material (`!pip`) and references variables before initialization at its top; make changes in the notebook or convert it to a runnable module before automated use.

## 2. Freeze the experimental question before changing the builder

Define one or more prediction tasks and their candidate universes. A recommended first primary task is ROOT link prediction across selected relation types, with per-relation and per-endpoint-type results. Drug-disease `indication` can be one task, not the entire evaluation. Internal and cross-context link prediction are separate tasks and should be reported separately. For each task specify: the known query inputs, valid head/tail entity types, whether endpoint graph context is known at inference, the candidate universe, split unit, raw/filtered rank definition, and handling of unscorable queries.

Do not compare MRRs when the candidate sets differ. A ROOT query over thousands of candidates and an internal query over dozens are different tasks. The current Stage 9 decoder scores exact endpoint occurrences; an internal/cross test fact is only scorable if its endpoints have independently justified occurrences in those contexts. Never retain a membership solely to make a test fact scorable. Global entities may remain in ROOT after their relation is held out.

## 3. Split the source facts before constructing any GoG inputs

1. Ingest a fixed, versioned PrimeKG source snapshot. Give each raw row a stable source ID **before** deduplication. Preserve source relation direction, display relation, provenance, and any source time/version fields. Review builder lines 410-417: the current unordered-pair deduplication may collapse directional records. Define a deliberate duplicate policy.
2. Decide the split unit. At minimum group exact duplicates and semantic inverse records across directions and contexts. If the intended task is an unseen *entity pair*, group every relation between the unordered pair, as the current loader does. If the task is an unseen *relation for a known pair*, grouping all relations by pair is too strict; still group copies/inverses of the same semantic fact. Document the choice.
3. Freeze and save a train/validation/test **source-fact manifest** with seed, counts per relation/type, source dataset hash, and grouping rule. Do not split contextual copies as independent rows.
4. Construct adjacency, degrees, owner queues, two-hop candidates, LLM payloads, memberships, selected internal edges, ROOT inputs, and any feature statistics from **training-available information only**. The present builder's all-fact adjacency at 422-426 must become train-only. A held-out source fact must not influence graph selection indirectly through degree, two-hop retrieval, prompt text, pruning, or graph-size caps.
5. Independently available external structures may be included if their source and availability are documented and they do not contain the hidden target label. Apply the same policy to drug, gene, and pathway owners. No owner type is automatically safe or unsafe.

## 4. Make every derived object auditable

For each internal membership, record graph ID, entity ID, builder rule, supporting **training source IDs** or external source/version, LLM prompt/model version if used, and selection score. Do the same for each internal edge. Preserve the semantic source ID on ROOT facts. A source fact copied into a computational channel must remain linked to its parent ID and split; it must not become a new supervised fact.

Do not automatically export the current cross-graph copies as semantic facts. If PrimeKG has no explicit context-specific endpoint evidence, cross-context **semantic target facts** may be absent in this experiment. That is preferable to inventing them. Identity and ownership can still support computational information flow in the encoder. If the team deliberately introduces derived cross-context channels, label them as model input channels, track their parent training fact, exclude them from validation/test targets, and do not count them as additional observed relations. This is a modeling change requiring explicit documentation and tests.

The full relation universe may be retained as task labels, but only training facts enter message passing, negative-training exclusion, LLM prompts, or graph construction. Test labels are used only by the final evaluator and filtering bookkeeping under the declared protocol.

## 5. Address token and scale limits without changing the experiment

Keep bounded per-owner retrieval. The LLM should rank a compact, source-grounded candidate list, not ingest the whole KG. Cache each owner decision by source snapshot hash, split hash, prompt/model version, and candidate-list hash so a run can resume without paying for repeated calls. Log actual input/output tokens, truncations, retry counts, rejected IDs, and owner-level failures. Validate every returned ID against the offered candidates; never accept a new LLM-invented fact. Save raw responses for audit, separately from model input features. Fix `QUICK_TEST` to genuinely use a tiny owner set. Increase graph count only after split safety and output validation pass.

Implement a deterministic candidate-ranking builder using the **same train-only candidate pools and graph-size budgets**. Comparing deterministic and LLM-assisted graph selection will reveal whether LLM judgment improves the graph data, rather than conflating builder and encoder effects.

## 6. Train and compare fairly

Use the same frozen source-fact split, same selected entity universe, same target queries, candidate lists, filtering rules, and decoder/ranking code for all models. A useful comparison ladder is: entity-only ComplEx; flat relational GNN over canonical entities; flat context-aware GNN given the permitted occurrences/membership/ownership; GoG given that same information. The flat context-aware model may encode membership as computational connections; those are not extra supervised target facts. Tune each model on validation under comparable compute/parameter budgets. Train beyond the current 10-epoch development run with validation-based early stopping, several seeds, and one sealed final test evaluation.

Report per-relation and per-scope MRR, Hits@1/3/10, head/tail results, candidate counts, coverage/unscorable counts, micro and clearly defined macro aggregates, training time, memory, and seed variation. Specify filtered known-positive sets and tie handling. Run GoG ablations for local processing, graph readout/fusion, cross attention, and identity synchronization. Interpret B>A as evidence that context information helps, and C>B as evidence that the GoG encoder uses the same context better; neither conclusion follows from one unreplicated score.

## 7. Acceptance checks before any MRR claim

- Every model input edge, membership, candidate-selection feature, and LLM prompt item traces to a permitted source ID or documented independent source.
- No validation/test source fact, duplicate, inverse, or derived contextual copy is in training adjacency or graph-construction inputs.
- The semantic fact count is based on unique source facts, not contextual copies; every copy carries its parent ID.
- Two clean builds with the same source snapshot, split, prompt/model version, and cached decisions produce identical manifests and graph objects.
- All compared models rank identical test queries against identical valid candidate sets; unsupported internal/cross queries are counted, never silently dropped.
- A membership-only diagnostic baseline is evaluated. If it predicts held-out labels unusually well, inspect provenance before trusting the result.
- The released artifacts include builder code, prompts/version, source and split hashes, provenance manifest, model configuration, checkpoints, evaluator configuration, and per-scope results.

**First implementation milestone:** produce a frozen source-fact split and a train-only, provenance-producing builder on a small owner subset. Do not spend more tokens or scale to more owners until its leakage audit and evaluator pass.
