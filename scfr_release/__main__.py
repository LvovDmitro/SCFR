"""Local-only SCFR fitting, inference, aggregate evaluation and synthetic demo."""
import argparse
import json
from pathlib import Path
import statistics

from . import pipeline


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Fit original proposal/verifier/router models; requires [verifier]")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--component-split", default="component_train")
    train.add_argument("--meta-split", default="meta_train")
    train.add_argument("--min-samples", type=int, default=20)
    infer = sub.add_parser("infer", help="Reference-free five-candidate construction and routing; requires [verifier]")
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="Evaluate keyed predictions against a separate local reference file")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--references", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    demo = sub.add_parser("demo", help="Invented-text software smoke test, not paper/clinical validation")
    demo.add_argument("--output-dir", type=Path, required=True)
    demo.add_argument("--full", action="store_true", help="Also fit real logistic verifier and five-candidate router")
    args = parser.parse_args()
    if args.command == "train":
        save(args.output, pipeline.train_bundle(load(args.input), min_samples=args.min_samples, component_split=args.component_split, meta_split=args.meta_split))
    elif args.command == "infer":
        save(args.output, pipeline.infer(load(args.input), load(args.model)))
    elif args.command == "evaluate":
        from scripts.analyze_source_causal_arms import row_scores
        refs = load(args.references)
        predictions = load(args.predictions)
        reference_index = {str(row["id"]): row["reference"] for row in refs}
        prediction_ids = [str(row["id"]) for row in predictions]
        if len(reference_index) != len(refs) or len(set(prediction_ids)) != len(predictions) or set(prediction_ids) != set(reference_index):
            raise ValueError("evaluation requires unique identical prediction/reference ID sets")
        scores = [row_scores({"prediction": row["prediction"], "reference": reference_index[str(row["id"])]}) for row in predictions]
        if not scores:
            raise ValueError("evaluation requires nonempty predictions")
        save(args.output, {"rows": len(scores), "metric_type": "rule_based_11_finding_proxy_not_clinical_ground_truth", "metrics": {key: statistics.mean(row[key] for row in scores) for key in scores[0]}})
    elif args.command == "demo":
        rows = pipeline.synthetic_rows()
        out = args.output_dir
        save(out / "invented_fixture.json", rows)
        if args.full:
            bundle = pipeline.train_bundle(rows, min_samples=1, diagnostic=True)
            save(out / "synthetic_model.json", bundle)
            inputs = [{k: v for k, v in row.items() if k in pipeline.INFERENCE_FIELDS} for row in rows if row["split"] == "test"]
            save(out / "inference_input.json", inputs)
            predictions = pipeline.infer(inputs, bundle)
            save(out / "predictions.json", predictions)
            receipt = {"fixture": "wholly_invented_nonclinical_software_test", "full_original_proposals_and_router_exercised": True, "predictions": len(predictions), "patient_data_used": False, "paper_metrics_reproduced": False, "diagnostic_grid": "C=.1,class_weight=balanced,threshold=.5", "real_training_epochs": 50, "reference_available_at_inference": False}
        else:
            report, _ = pipeline.ranker.build_report(rows, candidate_arms=pipeline.BASE_ARMS, split_field="split", tune_split="train", eval_splits=("validate", "test"), fp_weight=.2, fn_weight=.1, min_samples=1, epochs=50, learning_rate=.05, average_weights=True, allow_leaky_eval=False)
            save(out / "synthetic_base_ranker.json", report)
            receipt = {"fixture": "wholly_invented_nonclinical_software_test", "full_original_proposals_and_router_exercised": False, "base_ranker_exercised": True, "patient_data_used": False, "paper_metrics_reproduced": False}
        save(out / "receipt.json", receipt)
        print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
