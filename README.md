# SCFR: Source-Causal Finding Router

Public code for **Source-Causal Control of Historical Context in Longitudinal Radiology Report Generation**, accepted at NeurIPS 2026 Main Track as a poster (paper 21649), by Dmitry Lvov and Ilya Pershin. SCFR constructs source-valid report proposals and routes among them without a reference report at inference. Repository: [LvovDmitro/SCFR](https://github.com/LvovDmitro/SCFR).

The records under `verification/` describe the pre-publication code checks performed on 19 October 2026, including 41 passing tests. Their local-release-candidate wording records that earlier verification stage. The publication changes update this introduction and `CITATION.cff`; the verified algorithm and test files are unchanged.

The original code snapshot is preserved under `scripts/`, `src/`, `train/`, and `utils/`. The `scfr_release` wrapper adds explicit fitting, reference-free inference, and separate evaluation interfaces. `SOURCE_MANIFEST.json` records the source revision, dirty-worktree file hashes, and release paths. Source line endings are normalized to UTF-8/LF; algorithm files are otherwise unchanged.

## Install and run

Python 3.10 or newer is required. Run from this directory. The base reranker demo uses only the standard library:

```bash
python -B -m scfr_release demo --output-dir outputs/base-demo
```

For the full proposal/verifier/router path:

```bash
python -m pip install -e ".[verifier]"
python -B -m scfr_release demo --full --output-dir outputs/full-demo
python -B -m unittest discover -s tests -v
```

The fixture contains wholly invented radiology-like text. It contains no patient records or images. Its deliberately small fitting grid checks software execution; it does not reproduce paper results, estimate clinical performance, or validate a clinical system. The fixture and smoke artifacts are generated locally, not downloaded.

Optional extras are `analysis` for NumPy-based aggregate analyses and `corrector` for optional neural-corrector training. Model extras must be installed in separate environments: `maira2` pins Transformers 4.51.3, `medgemma` pins 5.6.2, and `chexagent` pins 4.40.0, following the original model-specific requirements. The `generators` extra contains shared adapter dependencies only; choose the model's Transformers environment separately. Do not combine the incompatible model extras. For MedGemma, the original GPU environment supplies PyTorch 2.4.1 and TorchVision 0.19.1; install the appropriate CUDA build separately. These GPU stacks were not installed or tested during release preparation. Additional model-specific dependencies and third-party repositories are not bundled; see `external/README.md`. There is no GPU or network requirement for the CPU demos after installing dependencies.

## Fit, construct proposals, infer, evaluate

Training input is a JSON list. Each row needs a unique `id`, `subject_id`, `split`, `reference`, `current_only`, `real_prior`, `prior_report_only`, and optional valid `prior_report`. Use explicit, subject-disjoint `component_train`, `meta_train`, and `test` splits. The wrapper rejects subjects shared across splits and missing report coverage.

```bash
python -B -m scfr_release train --input /authorized/merged.json --output /authorized/model.json
python -B -m scfr_release infer --input /authorized/inference.json --model /authorized/model.json --output /authorized/predictions.json
python -B -m scfr_release evaluate --predictions /authorized/predictions.json --references /authorized/references.json --output /authorized/aggregate.json
```

Inference accepts only a string `id` (or `sample_id`), the three source-valid base reports, and optional string `prior_report`/`context`. Structured context or identifier payloads are rejected. References, wrong-history reports, images, clinical metadata, and other fields are rejected at this boundary. Keep the inference input separate from the reference file. Evaluation requires unique identical sample-ID sets. It reports the original 11-finding rule-based proxy metrics, which are not clinical ground truth.

The five candidates, in tie-breaking order, are current-only, a perceptron proposal fitted with FP weight .2, a perceptron proposal fitted with FP weight .8, and verifier realizations with append budgets 0 and 2. Both proposal perceptrons use FN weight .1. The final router uses FP/FN weights .2/.4. Perceptrons use 50 epochs, learning rate .05, averaged coefficients, and first-maximum tie-breaking. Proposal features and router features call the original implementation. Each verifier budget independently selects its logistic model and thresholds through the original `liblinear` fitting, interactions, threshold search, constraints, and natural-language realization. Model schema `scfr-portable-model-v2` explicitly exports `verifiers.verifier_t0` and `verifiers.verifier_t2`, including their budgets, coefficients, selected hyperparameters, and thresholds. It does not load pickle files. The earlier local-only shared-verifier schema is not accepted.

Component models fit on `component_train`; the final router fits on `meta_train`. The portable wrapper follows the included OOF component configuration, including tuning verifier thresholds on the component-training split. It does not inspect `test` references during fitting. This configuration is a runnable refitting protocol, not a claim that newly fitted coefficients equal every archived paper model. Verifier constraint results are recorded in the model; an unsatisfied constraint is a diagnostic result, not permission to claim a passed clinical or paper gate.

The verifier's 0/2 limits constrain appended text. If filtering leaves no text, the original implementation returns a canonical fallback without applying that append limit. This behavior is retained and tested.

`models/scfr_controller_vectors_full_precision.json` contains the archived final-router vectors already provided with the manuscript. They are insufficient for end-to-end frozen inference without the corresponding fitted proposal/verifier models. Do not substitute new component fits and describe that as an exact frozen-model replay. The cross-generator OOF experiment refits component proposals on three folds, uses one fold for the deployment gate, tests on the remaining fold, and keeps the final `locked1000` router fixed. The portable `train` command instead fits a new complete bundle; these are different procedures.

## Reproduction

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) maps paper components and tables to the actual source entrypoints and required inputs. [CLI_INVENTORY.json](CLI_INVENTORY.json) lists parser options extracted from the shipped source. Every script also supports `--help` when its optional dependencies are installed.

No raw reports, images, private IDs, reader responses, signing keys, trained neural weights, or restricted dataset manifests are included. Real-data reproduction requires the relevant dataset authorization, original subject-disjoint splits, frozen generator revisions, protected arm outputs, and fitted component assets. The local tests do not verify GPU generation or reproduce the complete clinical-data experiments. See `verification/verification.json` for the exact checks performed and remaining boundaries.

## Data, models, and safety

Acquire credentialed data through its official access process and comply with its data-use agreement. Do not redistribute MIMIC-CXR or other restricted images, reports, per-case predictions, answer keys, credentials, or identifying split manifests. Keep local working outputs in access-controlled storage. Share only reviewed aggregates. The reader-analysis entrypoints verify local protocol artifacts; the release contains their source code, not historical keys or responses.

Source-valid arms use current images, true prior images, and/or a valid prior report. `matched_shuffled_prior` is a reference-conditioned, wrong-patient analysis-only intervention. Its construction requires the original explicit acknowledgement flag. Neither wrong-history arm is accepted as a deployable inference feature by the release wrapper.

Use frozen, locally authorized model snapshots and `--local_files_only`/`--local-files-only` where offered. The original generator adapters may use `trust_remote_code=True`; inspect the pinned upstream model code before execution. This release performs no automatic login, model download, remote submission, or clinical deployment. Adapters are retained for reproducibility, including their original fallback behavior and optional third-party requirements.

## License and citation

The [MIT license](LICENSE) is copied from the existing repository, including its original copyright attribution. It covers this repository's code only. Dataset, pretrained-model, and third-party-code licenses remain separate. No dataset, model, or third-party repository is relicensed here. `CITATION.cff` records confirmed paper metadata without inventing a DOI or public release date.
