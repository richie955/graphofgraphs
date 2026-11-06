# Stage 9 — Decoder, Training, and Leakage-Safe Evaluation

**Purpose:** Score typed directed relation candidates from final canonical and contextual states, train the encoder end-to-end, and evaluate without exposing held-out facts to message passing or candidate filtering.

**Status:** Dataset-independent training contract. Benchmark-specific split and metric settings belong in configuration.

---

## 1. Decoder inputs and scope resolution

The decoder receives the final canonical states `b_entity`, final occurrence states `z_occurrence`, occurrence/entity/graph mappings, and candidate triples with exact source and target context. A fact is represented as `(source_node_id, relation_id, target_node_id)`; global entity IDs alone are insufficient for context-specific facts.

Use the scope-specific endpoint occurrence states for scoring. Candidate identity is a global `entity_id`, but it is resolved to the unique GraphNode in the required graph context. With the Stage 0 invariant of at most one occurrence per entity per graph, resolution is deterministic. Canonical entity states are not substituted for a missing context occurrence.

## 2. Directional ComplEx scoring

Use complex dimension `d_c=128` by default to produce 256D real states. A shared projection maps each 256D endpoint occurrence state to a complex vector:

\[
q_s=Pz_s,\qquad q_t=Pz_t,\qquad q_r=R_r,
\]

where `P: R^256 -> C^128` is implemented as a real projection to 256 values split into real and imaginary halves, and `R_r` is a learned complex relation embedding. The directional score is:

\[
f(s,r,t)=\operatorname{Re}\sum_{k=1}^{128}q_{s,k}\,q_{r,k}\,\overline{q_{t,k}}.
\]

The relation vocabulary is global; relation embeddings are shared across scopes. The scoring function is directed. Do not assume symmetry or add an inverse fact in the decoder. If separate inverse relation IDs exist in the source, score them as distinct predicates.

For source/target corruption, preserve the positive relation and endpoint graph contexts. ROOT facts draw candidates with a ROOT occurrence. Same-internal-graph facts draw both corrupted endpoints from the fact's internal graph. Cross-context facts retain the original source and target graph IDs; a head candidate must occur in the source graph and a tail candidate in the target graph. The candidate entity must also satisfy the relation's configured endpoint type constraint. Candidates without the required occurrence are outside the candidate universe; do not synthesize occurrences or fall back to canonical states. A positive fact whose true endpoint cannot be resolved under these rules is invalid for that scope and must be counted and reported, not silently dropped.

## 3. Training objective and negatives

For every scorable positive training fact, generate head- and tail-corrupted negatives from the type- and context-valid candidate sets defined above. Exclude the positive itself and training-known true facts from negatives. Never consult validation/test positives to train the model. If a side has fewer eligible candidates than the configured negative count, sample all eligible candidates without replacement; if none remain, omit that side's negative term and record the shortfall. If neither side yields a negative, the positive may still contribute its positive term, and the shortfall must be reported.

Use a configurable logistic pairwise objective as the default:

\[
\mathcal L_{fact}=-\log\sigma(f^+)-\frac{1}{K}\sum_{j=1}^{K}\log\sigma(-f^-_j).
\]

Average over facts and configured negative sets. Record negative ratio, head/tail corruption mix, type constraints, context-resolution policy, and random seed. If using a benchmark-required alternative (e.g. softmax ranking or filtered 1-vs-all), specify its exact formula and candidate/filter policy in the run config.

Train end-to-end through all encoder stages and decoder. Do not detach intermediate states. Configure optimizer, learning rate, weight decay, gradient clipping, dropout, batch size/sampler, and early stopping explicitly. Save the selected checkpoint based only on validation metrics.

## 4. Split and adjacency isolation

Stage 0 assigns semantic facts to disjoint train/validation/test split units before constructing adjacency. Encoder message passing sees training facts only for every split. Validation is used for checkpoint selection and hyperparameter decisions; test is evaluated once under the declared final protocol.

Candidate filters for ranking metrics may use the benchmark's declared known-positive set, but report whether filtering uses train-only positives or train+validation+test positives. Such filtering is evaluation bookkeeping only: it must never add those facts to encoder adjacency, negative-training exclusions, feature construction, or model inputs. Prefer benchmark-prescribed filtered ranking when reproducing a benchmark; otherwise report both raw and filtered metrics with exact filter scope.

Ensure reciprocal/inverse facts are grouped into one split unit when required to prevent a held-out semantic edge from leaking through a separately stored inverse. The preprocessing report must prove that no held-out fact or generated channel entered training adjacency.

## 5. Evaluation

Evaluate each held-out directed fact by ranking valid head and tail candidates under its relation type and required graph context. The true endpoint must be in that candidate set; otherwise mark the query unscorable and report it in coverage counts. Report MRR, Hits@1/3/10, and mean rank, separately by ROOT, within-internal, and cross-context scope, plus macro and micro aggregates. State whether ranks are raw or filtered and define tie handling, candidate universe, type constraints, and unscorable-query counts.

Ranking groups queries that share a relation and endpoint context, projects each occurrence once, and scores candidate vectors in batches. This avoids recomputing endpoint projections for every query-candidate pair. Training also batches decoder loss over facts and negatives to reduce GPU memory, while accumulating one encoder gradient per full-graph pass. Run `python train_stage9.py DATASET.json --model baseline` for the entity-and-type baseline only, `--model gog` to train GoG only, or `--model compare` to train both.

Use validation for early stopping (default selection metric: macro validation MRR across scopes with a documented rule when a scope has no scorable queries). Report test metrics once for the selected checkpoint. Include per-scope total queries, scorable queries, unscorable queries, and candidate coverage so scores cannot hide unsupported scopes. If a benchmark mandates different metrics or split procedures, its protocol overrides these defaults and must be cited in experiment metadata.

## 6. Reproducibility and saved artifacts

Save model/optimizer state as applicable, configuration, random seeds, software versions, dataset/source version and fingerprint, Stage 0 mappings/vocabularies, split manifest, feature provenance, negative-sampling policy, candidate/filter policy, checkpoint-selection metric, and evaluation report. Scores must be traceable to fact IDs and raw entity/graph/occurrence IDs.

## 7. Acceptance criteria

Stage 9 is ready when:

1. Every candidate triple resolves to a valid relation and exact endpoint contexts.
2. ComplEx scoring is directed and uses the declared occurrence states.
3. Negative candidates obey relation type and context constraints and do not include training-known positives.
4. Validation/test facts and their reciprocal leakage units are absent from all training encoder channels.
5. Evaluation filtering is explicit and isolated from model inputs.
6. Metrics are reported overall and by scope with candidate coverage and reproducible ranking rules.
7. Validation alone selects the checkpoint; test is not used for tuning.
8. Checkpoints and reports contain enough metadata to reproduce preprocessing, training, and evaluation.
