# Reproduction map

Commands below run from the repository root. `/authorized` denotes access-controlled storage you are permitted to use; it is not a supplied dataset. Each executable module lists its options with `--help`. Keep protected outputs outside a public checkout.

## CPU workflow and fitting settings

Install `.[verifier]` for the proposal/verifier/router workflow and `.[analysis]` for NumPy-based aggregate analyses. `python -B -m unittest discover -s tests -v` runs the shipped tests; `python -B -m scfr_release demo --full --output-dir outputs/demo` exercises the complete fitting and inference path on invented text. The demo uses a reduced verifier search grid and does not reproduce a paper experiment.

The portable `train` command fits a new bundle, schema `scfr-portable-model-v2`. The two proposal perceptrons use FP weights .2 and .8, FN weight .1, 50 epochs, learning rate .05, and averaged coefficients. The final router uses FP/FN weights .2/.4 with the same epochs, learning rate, and averaging. Ties select the first candidate in this order: `current_only`, `cand_fp02`, `cand_fp08`, `verifier_t0`, `verifier_t2`.

Each verifier budget independently selects its logistic model through the original `liblinear` fitting, interaction features, and per-finding threshold search. The exported JSON includes both models' coefficients, intercepts, thresholds, hyperparameters, and constraint results. Budget 0 or 2 limits appended text. If filtering leaves no text, the original realization returns its canonical fallback outside that append limit; the tests preserve this behavior.

Proposal models fit on `component_train`; the final router fits on `meta_train`. Subjects must not cross splits. The verifier search tunes on the component-training split, following the included OOF component configuration. Test references do not enter fitting. An unsatisfied verifier constraint is recorded in the bundle and must not be described as a passed clinical or paper gate. The fit is a refitting protocol, not a reconstruction of every archived paper checkpoint.

### Archived router inference

`scfr_release route` applies one bundled final-router vector without fitting or evaluating against references. Input rows contain a unique string `id` (or `sample_id`), the three base-report strings `current_only`, `real_prior`, `prior_report_only`, all four proposal strings `cand_fp02`, `cand_fp08`, `verifier_t0`, `verifier_t2`, and optional string `prior_report`/`context`. Extra fields, references, and wrong-history arms are rejected. All five final candidates must be nonempty. The three base reports supply source-consistency features even when a proposal is selected.

```bash
python -B -m scfr_release route --input /authorized/candidates.json --panel locked1000 --output /authorized/selected.json
```

Use `--weights PATH` to supply a compatible archive; the default is the bundled `models/scfr_controller_vectors_full_precision.json`. The 19 feature names and candidate order must match the implementation. The supplied vectors are for `seed911`, `seed1221`, and `locked1000`; they do not contain proposal/verifier or pretrained generator weights. Exact archived routing also requires the matching historical proposal reports. The older `scripts.apply_frozen_scfr_controller` combines archived routing and reference-based evaluation for complete experiment inputs.

## Generator environments

Generator execution is separate from the CPU router. Acquire authorized model snapshots and data before running generation. The adapters retain model-specific dependencies and some upstream fallback behavior; inspect the selected adapter and its pinned upstream code. Where provided, use `--local_files_only`/`--local-files-only` with local snapshots. Some adapters use `trust_remote_code=True`, which executes upstream code.

Use separate environments for model extras: `maira2` pins Transformers 4.51.3, `medgemma` pins 5.6.2, and `chexagent` pins 4.40.0. Do not combine these incompatible extras. The `generators` extra has shared dependencies only. The original MedGemma environment uses PyTorch 2.4.1 and TorchVision 0.19.1; install the matching CUDA build separately. Third-party repositories and their weights are not bundled; see [external/README.md](external/README.md). The `corrector` extra is an optional neural-corrector training path, separate from the final proposal/router method. GPU generation is not covered by the CPU tests.

## Experiment entrypoints

| Component or display | Original executable entrypoints | Required inputs and boundary |
|---|---|---|
| Longitudinal source/temporal audit | `python -m scripts.audit_longitudinal_manifest --manifest /authorized/source.json --require-official-split --require-temporal-gap --output /authorized/audit.json` | Authorized source manifest and image roots; original code checks provenance and chronology. |
| Current, true-prior, random-wrong, matched-wrong, prior-text arms | `python -m scripts.materialize_counterfactual_arms --source-manifest /authorized/source.json --output-dir /authorized/arms --allow-reference-conditioned-matching` | Frozen source manifest; matching is split-scoped by default. Matched wrong-history uses target reference findings and is analysis-only. |
| Frozen generator reports and safety-prompt control | `python -m src.eval.evaluate_generators --generator maira2 --dataset /authorized/arm.json --dataset_base /authorized/images --model-revision PINNED_REVISION --local_files_only --require-all-images --preserve-input-metadata --record-generation-provenance --maira2-prompt-style scfr_safety_v1` | Authorized local model snapshot and pixels. Repeat on the same input with `--maira2-prompt-style default`. The included original MAIRA-2 adapter contains the fixed prompt. Outputs stay protected. No GPU execution was performed for this release. |
| Arm merge | `python -m scripts.merge_counterfactual_arm_predictions --help` | Repeat `--arm-manifest NAME=PATH` and `--arm-results NAME=PATH`; use `--require-complete`. |
| Two reranker proposals | `python -m scripts.source_causal_candidate_reranker --help` | Three source-valid reports, references on component-training split, FP .2/.8, FN .1. Portable fitting: `python -m scfr_release train --help`; inference: `python -m scfr_release infer --help`. |
| Logistic verifier proposals and compact realization | `python -m scripts.source_causal_verifier --help` | Source-valid feature arms only; trained coefficients, thresholds, original finding extractor; natural base current-only, compact append budgets 0/2. Requires `verifier` extra. |
| Final router, coefficient table, route counts | `python -m scripts.source_causal_meta_controller --help`; `python -m scripts.apply_frozen_scfr_controller --help` | Subject-disjoint final training/panel inputs and the four matching fitted proposals. `models/scfr_controller_vectors_full_precision.json` supplies final vectors, not proposal assets. |
| Main controller contrasts and direct-current CI table | `python -m scripts.analyze_source_causal_arms --help` | Paired per-case report arms, `--n-bootstrap 5000 --seed 13 --cluster-by subject_id`, exact panel membership. Original paired cluster resampling preserves within-subject rows. |
| Main corrected evidence table | `python -m scripts.source_causal_confirmatory_gate --target PANEL=CURRENT_JSON,MATCHED_JSON --output-json /authorized/confirm.json --output-md /authorized/confirm.md` | Clustered current/matched contrast aggregates for every panel; Holm correction applies within the original declared families. |
| Fixed-control appendix table (complete 18,716-study snapshot) | `python -m scripts.analyze_scfr_full_stress_baselines --inventory /authorized/inventory.json --n-bootstrap 5000 --output-json /authorized/fixed.json` | Exact protected inventory and full complete-case arm outputs; this population is distinct from the three 300-case controller panels. |
| Matched-budget selector controls | `python -m scripts.analyze_scfr_selector_baselines --help` | Frozen panel predictions and local subject labels; rows/patients gates retained. NumPy analysis extra. |
| Cross-generator OOF table | `python -m scripts.assign_scfr_subject_transfer_splits --help`; `python -m scripts.run_scfr_cross_generator_oof_rotation --help`; `python -m scripts.analyze_scfr_cross_generator_transfer --help` | Original RadInfer/Libra merged inputs, five subject folds, component/meta/test separation, archived router vectors, protected output root. Specify `--output-dir` for aggregate results; `--weights-receipt` defaults to the bundled vectors. |
| Reader positive-addition appendix table (Task A) | `python -m scripts.analyze_scfr_r01_task_a --help` | Locked reader labels, private mapping, original generated arms and protocol integrity evidence; 10,000 bootstrap draws, seed 20260728. Inputs are deliberately absent. |
| Active-route item appendix table (Task B) | `python -m scripts.analyze_scfr_r01_task_b --private-root /authorized/reader --output-json /authorized/task-b.json --output-md /authorized/task-b.md --bootstrap-replicates 100000 --bootstrap-seed 20260728` | Original signed sequential reader state, local integrity key, private mapping, and locked responses. Source is provided; credentials/responses cannot be redistributed. |
| Copy/context decomposition | `python -m scripts.analyze_scfr_copy_context_decomposition --inventory /authorized/inventory.json --output-json /authorized/copy.json --output-md /authorized/copy.md --n-bootstrap 5000 --seed 20260728` | Protected inventory and per-case texts, aggregate-only output; NumPy analysis extra. |
| Same-panel safety-prompt comparison | `python -m scripts.analyze_maira2_same_panel_prompt_results --help` | Paired default/safety predictions, original source-input and provenance receipts. Historical infrastructure submission scripts are excluded; use the included generator adapter directly in authorized compute. |
| Optional learned corrector training | `python -m train.train_corrector_medgemma_multimodal --help`; `python -m scripts.run_source_causal_corrector_eval --help` | Authorized preference drafts, images, locally available MedGemma snapshot; separate optional model path, not a replacement for the paper's proposal/router method. |

Utility/epoch sensitivity tables require the original fixed panels and component candidates. The final router's actual training function is `train_structured_perceptron` in `scripts/source_causal_candidate_reranker.py`; vary only the declared FP/FN or epoch setting and retain the same component features, subject exclusion, seed, and evaluation population. Aggregate sensitivity receipts and the final manuscript evidence records are curated separately by the authors. A fresh parameter sweep is not included among this release's validation claims.

Historical main-table LaTeX can be generated with `python -m scripts.build_scfr_publication_tables --raw-json /authorized/raw.json --confirm-json /authorized/confirm.json --output-tex /authorized/tables.tex --output-md /authorized/tables.md`. The builder was recovered from the archived submission revision. Its exact original JSON inputs are not included; `evidence/` preserves reviewed historical aggregate Markdown with source/transform manifests. These summaries verify reported aggregate values without recreating absent per-case inputs.

### Cross-generator OOF protocol

The archived orchestration assigns each subject to `SHA256(seed + NUL + subject_id)[:8]` interpreted big-endian modulo 5, with seed 20260728. Rotation `f` tests fold `f`, uses `(f+1) mod 5` for `meta_train`, and fits proposal components on the other three folds. The final ranker is the frozen `locked1000` vector and is never refitted on the target generator. The four meta-fold parity conditions are SCFR-minus-current F1 >= -.02, status accuracy >= -.02, FP <= +.10, and FN <= +.10. If any condition fails, every outer-test case falls back to its current-only report. Wrong-history repair direction and its interval are recorded but do not determine this deployment decision. This point-estimate gate does not establish statistical non-inferiority.

Place the authorized complete-case inputs at `PROTECTED/work/radinfer_pilot400_frozen_4arm_merged.json` and `PROTECTED/work/libra_pilot400_frozen_4arm_completecase_merged.json`. These historical filenames identify the expected input role, not supplied files. Run:

```bash
python -B -m scripts.run_scfr_cross_generator_oof_rotation --protected-root /authorized/transfer --output-dir /authorized/transfer-aggregates
```

The script loads bundled router vectors unless overridden with `--weights-receipt`. Per-case outputs and execution logs remain under `PROTECTED/oof5`; only aggregate analyses and the validation receipt go to `--output-dir`. Historical panel membership and matching proposal assets are not reconstructed by this command. New OOF fits on newly assembled inputs must be reported as a new experiment.

## Expected input schemas

Base-arm rows are JSON objects with unique `id` or `sample_id`, `subject_id`, `split`, `current_only`, `real_prior`, `prior_report_only`, optional `prior_report`, and a `reference` used only during fitting/evaluation. Counterfactual construction and image generation additionally use `image_paths`, `prior_image_paths`, chronology, and official-split metadata as documented by their original parsers.

Candidate files for the meta-controller are JSON lists with the same sample key and a nonempty `prediction`. The original materializer checks duplicate candidate keys and reports missing coverage. Never silently join by row order or substitute another panel.

The portable inference CLI accepts a strict projection of base-arm rows with no `reference` or wrong-history arms. Generated proposals and selected text can still be restricted patient-derived data: protection is required even when identifiers have been removed.

## Open reproduction requirements

Exact end-to-end replay requires licensed data access, immutable generator snapshots, historical split/panel membership, matching fitted proposal/verifier assets, and authorized reader-study artifacts. The supplied final vectors and source code do not replace these inputs. The historical aggregate evidence supports checking the reported arithmetic, but cannot independently rerun protected-data experiments.
