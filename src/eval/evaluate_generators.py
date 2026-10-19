import argparse
import json
import os
import time
import traceback
import inspect
from pathlib import Path
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# LLM-RG4 needs its repo for evalcap, models
sys.path.insert(0, str(PROJECT_ROOT / "external" / "LLM-RG4"))
# LLaVA-Rad (Microsoft) needs its repo
sys.path.insert(0, str(PROJECT_ROOT / "external" / "llava-rad"))

from src.generators import get_generator


def build_result_row(
    item,
    prediction,
    generator=None,
    preserve_input_metadata=False,
    record_generation_provenance=False,
):
    """Build an eval row while keeping the legacy schema as the default."""
    if preserve_input_metadata:
        row = dict(item)
        row["prediction"] = prediction
    else:
        row = {
            "id": item["id"],
            "image_paths": item["image_paths"],
            "prior_image_paths": item.get("prior_image_paths", []),
            "context": item.get("context", ""),
            "prior_report": item.get("prior_report", ""),
            "prediction": prediction,
            "reference": item["reference"],
        }

    if record_generation_provenance:
        raw_prediction = prediction
        raw_getter = getattr(generator, "get_last_raw_prediction", None)
        if callable(raw_getter):
            raw_prediction = raw_getter() or prediction
        row["raw_prediction"] = raw_prediction

        metadata_getter = getattr(
            generator,
            "get_last_generation_metadata",
            None,
        )
        if callable(metadata_getter):
            row["generation_metadata"] = metadata_getter()

    return row


def _unwrap_wrapped_path(p: str) -> str:
    """Handle directory-wrapping: a file like 'foo.jpg' may be stored as
    a directory 'foo.jpg/' containing the actual file 'foo.jpg/foo.jpg'.
    Returns the unwrapped path if it exists, otherwise the original path."""
    import os
    if os.path.isdir(p):
        basename = os.path.basename(p)
        inner = os.path.join(p, basename)
        if os.path.isfile(inner):
            return inner
    return p


def _rewrite_path(p: str, base: Path) -> str:
    """Resolve an archived absolute path against an available dataset base path."""
    p_path = Path(p.replace("\\", "/"))
    parts = p_path.parts
    # If the path is absolute on linux, the first part is '/', so strip it so it doesn't override base
    if parts and parts[0] in ('/', '\\'):
        parts = parts[1:]

    # Try checking suffixes of the original path, from longest to shortest
    for i in range(len(parts)):
        suffix_path = Path(*parts[i:])
        full_test = base / suffix_path
        if full_test.exists():
            resolved = str(full_test)
            # Handle directory-wrapping (file stored as dir/file)
            return _unwrap_wrapped_path(resolved)

    return p


def load_dataset(
    json_path: Path,
    dataset_base: Path = None,
    max_samples: int = 0,
    require_all_images: bool = False,
):
    if not json_path.exists():
        print(f"Dataset not found at {json_path}")
        return []

    # Handle archives where the path is a directory wrapping the actual JSON file.
    # e.g. .../eval_manifest.json/eval_manifest.json
    if json_path.is_dir():
        inner = json_path / json_path.name
        if inner.is_file():
            print(f"Resolved directory-wrapped JSON: {inner}")
            json_path = inner
        else:
            print(f"ERROR: {json_path} is a directory, not a file, and no inner JSON found")
            return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    base = dataset_base or PROJECT_ROOT
        
    # Verify images exist, rewriting stale archived paths when a base is supplied.
    valid_data = []
    dropped = 0
    for item in data:
        paths = list(item["image_paths"])
        paths = [_rewrite_path(p, base) for p in paths]
        prior_paths = [_rewrite_path(p, base) for p in item.get("prior_image_paths", [])]
        if require_all_images:
            invalid_paths = [
                p
                for p in paths + prior_paths
                if not Path(p).is_file() or Path(p).is_symlink()
            ]
            if not paths or invalid_paths:
                raise FileNotFoundError(
                    "Strict image-input validation failed before generation "
                    f"for sample {item.get('id', '?')}: {invalid_paths}"
                )
            valid_paths = paths
            valid_prior_paths = prior_paths
        else:
            valid_paths = [p for p in paths if os.path.exists(p)]
            valid_prior_paths = [p for p in prior_paths if os.path.exists(p)]
        if valid_paths:
            item["image_paths"] = valid_paths
            item["prior_image_paths"] = valid_prior_paths
            valid_data.append(item)
        else:
            dropped += 1
            if dropped <= 3:
                print(f"WARNING: No images found for sample {item.get('id', '?')}, tried: {paths}")

    if dropped > 0:
        print(f"WARNING: {dropped}/{len(data)} samples dropped (images not found)")
    if dropped > len(data) * 0.5:
        print(f"ERROR: More than 50% of samples have missing images! Check dataset_base path.")

    if max_samples and max_samples > 0 and len(valid_data) > max_samples:
        print(f"Limiting dataset to first {max_samples}/{len(valid_data)} valid samples")
        valid_data = valid_data[:max_samples]

    return valid_data

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generator model.")
    parser.add_argument("--generator", type=str, required=True,
                        help="Generator name (e.g. r2gen, r2gencmn, medgemma, llava_med, llava_rad, maira2, llm_rg4, libra, etc.)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to evaluation JSON.")
    parser.add_argument("--output_suffix", type=str, default=None,
                        help="Suffix for result filename. Auto-derived from dataset path if not set.")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--model-revision",
        type=str,
        default="",
        help="Exact Hugging Face revision forwarded to the generator.",
    )
    parser.add_argument("--dataset_base", type=str, default=None,
                        help="Base path used to resolve archived absolute image paths.")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional cap applied after path validation.")
    parser.add_argument(
        "--require-all-images",
        action="store_true",
        help=(
            "Fail before generation if any current/prior manifest image is "
            "missing, not a regular file, or a symlink."
        ),
    )
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-generator timeout in seconds (default: 1800 = 30 min).")
    parser.add_argument(
        "--maira2-prompt-style",
        type=str,
        default="default",
        choices=["default", "scfr_safety_v1"],
        help=(
            "MAIRA-2 reporting prompt style. The default reproduces the "
            "existing adapter; scfr_safety_v1 enables the frozen "
            "training-free safety instruction."
        ),
    )
    parser.add_argument(
        "--preserve-input-metadata",
        action="store_true",
        help="Copy every JSON manifest field into each output row.",
    )
    parser.add_argument(
        "--record-generation-provenance",
        action="store_true",
        help=(
            "Record raw decoded prediction and generator-provided prompt "
            "metadata in each output row."
        ),
    )
    args = parser.parse_args()

    if args.generator != "maira2" and args.maira2_prompt_style != "default":
        parser.error("--maira2-prompt-style is only valid with --generator maira2")

    print(f"Initializing generator: {args.generator}", flush=True)
    try:
        generator_kwargs = {
            "local_files_only": args.local_files_only,
            "dataset_name": args.output_suffix,
        }
        if args.model_revision:
            generator_kwargs["revision"] = args.model_revision
        if args.generator == "maira2":
            generator_kwargs["prompt_style"] = args.maira2_prompt_style
        gen = get_generator(args.generator, **generator_kwargs)
    except Exception as e:
        print(f"ERROR: Failed to load generator {args.generator}: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    dataset_base = Path(args.dataset_base) if args.dataset_base else None
    dataset = load_dataset(
        Path(args.dataset),
        dataset_base=dataset_base,
        max_samples=args.max_samples,
        require_all_images=args.require_all_images,
    )
    if not dataset:
        print(f"ERROR: No evaluation dataset found or no images exist for {args.generator}. Exiting with failure.", flush=True)
        sys.exit(1)

    print(f"Loaded dataset with {len(dataset)} samples. Starting generation...")
    
    results = []
    error_count = 0
    start_time = time.time()
    timeout_reached = False
    for item in tqdm(dataset):
        # Check per-generator timeout
        elapsed = time.time() - start_time
        if elapsed > args.timeout:
            print(f"TIMEOUT: Generator {args.generator} exceeded {args.timeout}s limit "
                  f"after {len(results)}/{len(dataset)} samples. Saving partial results.", flush=True)
            timeout_reached = True
            break

        try:
            generate_kwargs = {
                "image_paths": item["image_paths"],
                "prior_image_paths": item.get("prior_image_paths"),
                "context": item.get("context"),
            }
            if "prior_report" in inspect.signature(gen.generate).parameters:
                generate_kwargs["prior_report"] = item.get("prior_report")
            pred = gen.generate(**generate_kwargs)

            # Guard against None/non-string returns
            if pred is None:
                pred = ""
            elif not isinstance(pred, str):
                pred = str(pred)

            results.append(
                build_result_row(
                    item,
                    pred,
                    generator=gen,
                    preserve_input_metadata=args.preserve_input_metadata,
                    record_generation_provenance=args.record_generation_provenance,
                )
            )
        except Exception as e:
            error_count += 1
            print(f"ERROR [{args.generator}] sample {item.get('id', '?')}: {e}", flush=True)
            if error_count <= 3:
                traceback.print_exc()
            # Append empty prediction instead of skipping
            results.append(
                build_result_row(
                    item,
                    "",
                    generator=gen,
                    preserve_input_metadata=args.preserve_input_metadata,
                    record_generation_provenance=args.record_generation_provenance,
                )
            )

    status = "TIMEOUT (partial)" if timeout_reached else "complete"
    print(f"Generation {status}: {len(results)}/{len(dataset)} samples processed, "
          f"{error_count} errors, {time.time() - start_time:.1f}s elapsed", flush=True)

    # Compute quick metrics if possible (Optional, or leave for dedicated metric script)
    out_dir = PROJECT_ROOT / "src" / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = args.output_suffix
    if suffix is None:
        suffix = Path(args.dataset).stem
    out_path = out_dir / f"results_{args.generator}_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"Finished evaluation. Results saved to {out_path}")
    
    # Print a quick sample
    if results:
        print("\n--- SAMPLE OUTPUT ---")
        print(f"REFERENCE: {results[0]['reference']}")
        print(f"PREDICTION: {results[0]['prediction']}")
        print("---------------------\n")

if __name__ == "__main__":
    main()

