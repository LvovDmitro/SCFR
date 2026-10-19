#!/usr/bin/env python3
"""
Multimodal MedGemma Corrector trained with Severity-Weighted DPO (SW-DPO).

Training formulation (doctor-with-everything):
  - Chosen:   preferred correction target (clinical reference for SFT-style rows,
              preference-chosen arm for source-causal DPO rows)
  - Rejected: R2Gen draft (what the generator produced)
  - Image:    X-ray image (visual ground truth)
  - Error type: from Verifier (informs severity weight)

SW-DPO loss (Rafailov et al. + clinical severity weighting):
  L = -w * log σ(β * [(log π_θ(chosen) - log π_ref(chosen))
                       - (log π_θ(rejected) - log π_ref(rejected))])

where π_ref is a frozen copy of the model at the start of DPO training.

The Corrector is trained as a FULL REPORT generator — it produces the complete
corrected report, not just an appended sentence. This is the "doctor with
everything at hand" formulation: given the image + erroneous draft, produce
the correct complete report.

Data: data/r2gen_drafts/{train,val,test}.json
  Each sample: {id, image_paths, draft (rejected), reference (chosen/preferred target)}
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.metrics_extended import compute_bleu_corpus, compute_rouge_l_corpus

# Severity weights by error type (from ReXVal)
# Clinically critical errors (false negation/prediction) get the highest weight.
SEVERITY_WEIGHTS = {
    "false prediction": 1.0,
    "false negation": 1.0,
    "change location": 0.7,
    "change severity": 0.6,
    "change measurement": 0.6,
    "add medical device": 0.5,
    "change position of device": 0.4,
    "change name of device": 0.3,
    "add typo": 0.2,
    "add repetition": 0.2,
    "change to homophone": 0.2,
    "not applicable": 0.1,
}


def get_severity_weight(error_type: str, weights: Dict = None) -> float:
    if weights is None:
        weights = SEVERITY_WEIGHTS
    return weights.get(error_type.lower().strip(), 0.5)


def _metric_scalar(metric, keys: Tuple[str, ...], default: float = 0.0) -> float:
    """Extract the scalar score used for checkpoint selection from metric output."""
    if isinstance(metric, Mapping):
        for key in keys:
            if key in metric and metric[key] is not None:
                return float(metric[key])
        return default
    if metric is None:
        return default
    return float(metric)


# ─────────────────── Dataset ───────────────────

class CorrectorDPODataset(Dataset):
    """
    DPO training dataset for the corrector.

    Each sample:
      - chosen:   preferred target; may be clinical ground truth or preference-chosen text
      - rejected: R2Gen draft (to be corrected)
      - image:    X-ray image paths
      - error_type: detected or estimated error type (for severity weight)

    If available, also uses synthetic error data to augment training:
      - chosen: original sentence
      - rejected: error-injected sentence
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        use_synthetic_augment: bool = True,
        drafts_dir: Optional[str] = None,
        image_root: Optional[str] = None,
    ):
        default_roots = [
            PROJECT_ROOT / "Datasets" / "IU_CXR",
            PROJECT_ROOT / "data" / "local_sources",
        ]
        env_root = os.environ.get("CORRECTOR_IMAGE_ROOT", "").strip()
        raw_roots = [image_root, env_root, *[str(path) for path in default_roots]]
        self.image_roots = []
        for raw_root in raw_roots:
            if not raw_root:
                continue
            root_path = Path(raw_root)
            if not root_path.is_absolute():
                root_path = PROJECT_ROOT / root_path
            if root_path not in self.image_roots:
                self.image_roots.append(root_path)
        self.image_dir = self.image_roots[0] if self.image_roots else PROJECT_ROOT
        self.samples = []

        # ── Primary: R2Gen draft pairs ──
        drafts_root = Path(drafts_dir) if drafts_dir else PROJECT_ROOT / "data" / "r2gen_drafts"
        drafts_path = drafts_root / f"r2gen_drafts_{split}.json"
        if drafts_path.exists():
            with open(drafts_path) as f:
                drafts = json.load(f)
            print(f"[CorrectorDPODataset:{split}] Loaded {len(drafts)} R2Gen draft pairs")
            for item in drafts:
                draft = item.get("draft", "").strip()
                reference = item.get("reference", "").strip()
                img_paths = item.get("image_paths", [])
                if isinstance(img_paths, str):
                    try:
                        import ast
                        img_paths = ast.literal_eval(img_paths)
                    except Exception:
                        img_paths = [img_paths]

                if len(draft) < 5 or len(reference) < 5:
                    continue
                # Skip trivially identical pairs
                if draft.strip().lower() == reference.strip().lower():
                    continue

                self.samples.append({
                    "chosen": reference,
                    "rejected": draft,
                    "image_paths": img_paths,
                    "error_type": item.get("error_type", "false prediction"),
                    "severity_weight": item.get("severity_weight"),
                    "source": item.get("source", "r2gen"),
                })
        else:
            print(f"[CorrectorDPODataset:{split}] No R2Gen drafts found at {drafts_path}")

        # ── Augment with synthetic report-level errors ──
        if use_synthetic_augment:
            for v in ["v2", ""]:
                syn_dir = PROJECT_ROOT / "Datasets" / f"synthetic_iuxray_errors{'_'+v if v else ''}"
                syn_path = syn_dir / f"synthetic-report-level_{split}.csv"
                if syn_path.exists():
                    df = pd.read_csv(syn_path)
                    df = df[
                        df["original_report"].notna()
                        & df["error_report"].notna()
                        & (df["original_report"] != df["error_report"])
                    ]
                    print(f"  Augmenting with {len(df)} synthetic pairs from {syn_path}")
                    for _, row in df.iterrows():
                        import ast
                        try:
                            img_paths = ast.literal_eval(row["image_path"]) if pd.notna(row["image_path"]) else []
                        except Exception:
                            img_paths = []
                        self.samples.append({
                            "chosen": str(row["original_report"]),
                            "rejected": str(row["error_report"]),
                            "image_paths": img_paths,
                            "error_type": str(row.get("error_type", "not applicable")),
                            "source": "synthetic",
                        })
                    break

        if max_samples and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        print(f"  [{split}] Total DPO pairs: {len(self.samples)}")
        sources = {}
        for s in self.samples:
            sources[s["source"]] = sources.get(s["source"], 0) + 1
        print(f"  Sources: {sources}")

    def _find_image(self, paths: List) -> Optional[str]:
        for p in paths:
            if not isinstance(p, str):
                continue
            raw_path = Path(p)
            candidates = [raw_path]
            for root in self.image_roots:
                candidates.append(root / p)
                candidates.append(root / raw_path.name)
            for candidate in candidates:
                if candidate.exists():
                    return str(candidate)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self._find_image(s["image_paths"])
        severity = s.get("severity_weight")
        try:
            severity_weight = float(severity)
        except (TypeError, ValueError):
            severity_weight = get_severity_weight(s["error_type"])
        return {
            "chosen": s["chosen"],
            "rejected": s["rejected"],
            "error_type": s["error_type"],
            "severity_weight": severity_weight,
            "image_path": img_path,
        }


def corrector_collate(batch):
    return {
        "chosen": [b["chosen"] for b in batch],
        "rejected": [b["rejected"] for b in batch],
        "error_types": [b["error_type"] for b in batch],
        "severity_weights": torch.tensor([b["severity_weight"] for b in batch], dtype=torch.float32),
        "image_paths": [b["image_path"] for b in batch],
    }


# ─────────────────── Model ───────────────────

class MultimodalCorrector(nn.Module):
    """
    MedGemma 4B Corrector fine-tuned with SW-DPO.

    At inference: given (image, erroneous_draft, error_type_from_verifier)
                  → generate complete corrected report.

    The prompt is always: "Given this X-ray and the following report that may
    contain errors, write a corrected radiology report."
    This is the doctor formulation: the model has all the information needed.
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        use_qlora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        adapter_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        kw = {"trust_remote_code": True}
        if local_files_only:
            kw["local_files_only"] = True

        print(f"[MultimodalCorrector] Loading {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name, **kw)

        if use_qlora:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, quantization_config=bnb,
                device_map="auto", dtype=torch.bfloat16, **kw,
            )
            self.model = prepare_model_for_kbit_training(self.model)
            if adapter_dir:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_dir,
                    is_trainable=False,
                )
            else:
                lora_cfg = LoraConfig(
                    r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    bias="none", task_type="CAUSAL_LM",
                )
                self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name, dtype=torch.bfloat16, device_map="auto", **kw,
            )
            if adapter_dir:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_dir,
                    is_trainable=False,
                )

    def _build_prompt(
        self,
        draft: str,
        error_type: Optional[str] = None,
    ) -> str:
        """Build the correction prompt. Always asks for the full corrected report."""
        if error_type and error_type.lower() not in ("not applicable", ""):
            type_hint = f" (suspected error type: {error_type})"
        else:
            type_hint = ""

        return (
            f"The following radiology report may contain clinical errors{type_hint}. "
            f"Based on the X-ray image and your clinical knowledge, write the complete "
            f"corrected radiology report. Be thorough and accurate.\n\n"
            f"Draft report: {draft}\n\n"
            f"Corrected report:"
        )

    def _build_inputs(
        self,
        texts: List[str],
        image_paths: List[Optional[str]],
    ) -> Dict:
        """Build processor inputs for (text, image) pairs."""
        images = []
        formatted_texts = []
        for text, img_path in zip(texts, image_paths):
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
            else:
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            images.append(img)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }]
            formatted_texts.append(self.processor.apply_chat_template(messages, tokenize=False))

        inputs = self.processor(
            text=formatted_texts,
            images=[[img] for img in images],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return inputs

    def compute_logprobs(
        self,
        texts: List[str],
        image_paths: List[Optional[str]],
    ) -> torch.Tensor:
        """Compute average per-token log-probability for each text."""
        inputs = self._build_inputs(texts, image_paths)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        mask = inputs["attention_mask"][:, 1:].float()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        token_lp = logprobs.gather(2, labels.unsqueeze(-1)).squeeze(-1) * mask
        return token_lp.sum(-1) / mask.sum(-1).clamp(min=1)

    @torch.no_grad()
    def correct(
        self,
        draft: str,
        image_path: Optional[str],
        error_type: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate a complete corrected report given a draft and image."""
        self.eval()
        prompt = self._build_prompt(draft, error_type)
        inputs = self._build_inputs([prompt], [image_path])
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        gen_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


# ─────────────────── SW-DPO Loss ───────────────────

class SeverityWeightedDPOLoss(nn.Module):
    """
    Proper SW-DPO loss with frozen reference model.

    L = -w * log σ(β * [(log π_θ(chosen) - log π_ref(chosen))
                         - (log π_θ(rejected) - log π_ref(rejected))])

    The reference model prevents reward hacking: without it, the policy
    trivially wins by assigning extreme probabilities to short degenerate outputs.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        chosen_lp: torch.Tensor,
        rejected_lp: torch.Tensor,
        severity_weights: torch.Tensor,
        ref_chosen_lp: Optional[torch.Tensor] = None,
        ref_rejected_lp: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if ref_chosen_lp is not None and ref_rejected_lp is not None:
            rewards_diff = self.beta * (
                (chosen_lp - ref_chosen_lp) - (rejected_lp - ref_rejected_lp)
            )
        else:
            rewards_diff = self.beta * (chosen_lp - rejected_lp)

        per_sample = -F.logsigmoid(rewards_diff)
        loss = (per_sample * severity_weights).mean()

        return loss, {
            "dpo_loss": loss.item(),
            "reward_margin": rewards_diff.mean().item(),
            "chosen_lp": chosen_lp.mean().item(),
            "rejected_lp": rejected_lp.mean().item(),
            "avg_severity": severity_weights.mean().item(),
        }


# ─────────────────── Supervised Fine-Tuning Loss (Phase 1) ───────────────────

def compute_sft_loss(
    model: MultimodalCorrector,
    chosen_texts: List[str],
    drafts: List[str],
    image_paths: List[Optional[str]],
    device: torch.device,
) -> torch.Tensor:
    """
    Supervised fine-tuning: train model to generate chosen given draft as context.
    Uses causal LM cross-entropy, masking prompt tokens (only loss on output).
    """
    prompts = [model._build_prompt(d) for d in drafts]
    full_texts = [p + " " + c for p, c in zip(prompts, chosen_texts)]
    inputs_full = model._build_inputs(full_texts, image_paths)
    inputs_prompt = model._build_inputs(prompts, image_paths)

    device_m = next(model.model.parameters()).device
    inputs_full = {k: v.to(device_m) for k, v in inputs_full.items()}

    prompt_lens = inputs_prompt["input_ids"].shape[1]

    outputs = model.model(**inputs_full)
    logits = outputs.logits[:, :-1, :]
    labels = inputs_full["input_ids"][:, 1:].clone()
    mask = inputs_full["attention_mask"][:, 1:].float()

    # Only compute loss on the response part (not the prompt)
    labels[:, :prompt_lens] = -100
    mask[:, :prompt_lens] = 0

    valid = mask > 0
    if not valid.any():
        return torch.tensor(0.0, device=device_m, requires_grad=True)

    loss_flat = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    )
    loss = (loss_flat.reshape(labels.shape) * mask).sum() / mask.sum().clamp(min=1)
    return loss


# ─────────────────── Training Loops ───────────────────

def train_epoch_sft(
    model: MultimodalCorrector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
) -> float:
    """Phase 1: SFT — teach the model the report correction format."""
    model.model.train()
    total_loss = 0.0
    n_steps = 0
    accum_loss = 0.0
    accum_count = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        image_paths = batch["image_paths"]

        try:
            loss = compute_sft_loss(model, chosen, rejected, image_paths, device)
            loss_s = loss / grad_accum_steps
            loss_s.backward()
            accum_loss += loss.item()
            accum_count += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accum_count = 0
                accum_loss = 0.0
                continue
            raise

        if accum_count >= grad_accum_steps or batch_idx == len(loader) - 1:
            if accum_count > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
                total_loss += accum_loss
                n_steps += 1
                accum_loss = 0.0
                accum_count = 0

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            avg = total_loss / max(n_steps, 1)
            print(f"  [SFT] E{epoch} B{batch_idx+1}/{len(loader)} | loss: {avg:.4f}")

    return total_loss / max(n_steps, 1)


def train_epoch_dpo(
    model: MultimodalCorrector,
    ref_model: Optional[MultimodalCorrector],
    loader: DataLoader,
    loss_fn: SeverityWeightedDPOLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
) -> float:
    """Phase 2: SW-DPO — preference training with frozen reference model."""
    model.model.train()
    total_loss = 0.0
    n_steps = 0
    accum_loss = 0.0
    accum_count = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        image_paths = batch["image_paths"]
        sw = batch["severity_weights"].to(device)

        try:
            # Policy log-probs
            chosen_lp = model.compute_logprobs(chosen, image_paths)
            rejected_lp = model.compute_logprobs(rejected, image_paths)

            # Reference log-probs (frozen model)
            ref_chosen_lp = ref_rejected_lp = None
            if ref_model is not None:
                with torch.no_grad():
                    ref_chosen_lp = ref_model.compute_logprobs(chosen, image_paths)
                    ref_rejected_lp = ref_model.compute_logprobs(rejected, image_paths)

            loss, metrics = loss_fn(
                chosen_lp, rejected_lp, sw,
                ref_chosen_lp=ref_chosen_lp,
                ref_rejected_lp=ref_rejected_lp,
            )

            loss_s = loss / grad_accum_steps
            loss_s.backward()
            accum_loss += loss.item()
            accum_count += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                accum_count = 0
                accum_loss = 0.0
                continue
            raise

        if accum_count >= grad_accum_steps or batch_idx == len(loader) - 1:
            if accum_count > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
                total_loss += accum_loss
                n_steps += 1
                accum_loss = 0.0
                accum_count = 0

        if (batch_idx + 1) % 50 == 0 or batch_idx == 0:
            avg = total_loss / max(n_steps, 1)
            print(f"  [DPO] E{epoch} B{batch_idx+1}/{len(loader)} | "
                  f"loss: {avg:.4f} | margin: {metrics.get('reward_margin', 0):.4f}")

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def evaluate_corrector(
    model: MultimodalCorrector,
    val_data: List[Dict],
    max_eval: int = 100,
) -> Dict:
    """Evaluate corrector on a subset of val data using BLEU and ROUGE-L."""
    model.model.eval()
    random.seed(42)
    subset = random.sample(val_data, min(max_eval, len(val_data)))

    hyps = []
    refs = []
    for item in subset:
        try:
            corrected = model.correct(
                draft=item["rejected"],
                image_path=item.get("image_path"),
                error_type=item.get("error_type"),
                max_new_tokens=200,
            )
            hyps.append(corrected)
            refs.append(item["chosen"])
        except Exception as e:
            hyps.append(item["rejected"])  # fallback: no correction
            refs.append(item["chosen"])

    refs_wrapped = [[ref] for ref in refs]
    bleu = _metric_scalar(
        compute_bleu_corpus(hyps, refs_wrapped),
        ("BLEU_4_corpus", "BLEU_4", "bleu4", "bleu_4"),
    )
    rouge = _metric_scalar(
        compute_rouge_l_corpus(hyps, refs_wrapped),
        ("ROUGE_L_f1", "rouge_l", "rougeL"),
    )

    # Also compute improvement over draft
    draft_bleu = _metric_scalar(
        compute_bleu_corpus([item["rejected"] for item in subset], refs_wrapped),
        ("BLEU_4_corpus", "BLEU_4", "bleu4", "bleu_4"),
    )

    return {
        "bleu4": bleu,
        "rouge_l": rouge,
        "draft_bleu4": draft_bleu,
        "improvement": bleu - draft_bleu,
        "n_evaluated": len(subset),
    }


def main():
    parser = argparse.ArgumentParser(description="Train MedGemma Corrector with SW-DPO")
    parser.add_argument("--sft-epochs", type=int, default=2,
                        help="Supervised fine-tuning epochs (Phase 1)")
    parser.add_argument("--dpo-epochs", type=int, default=3,
                        help="SW-DPO training epochs (Phase 2)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dpo-lr", type=float, default=5e-6)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-val", type=int, default=500)
    parser.add_argument("--val-eval-max", type=int, default=50)
    parser.add_argument("--model", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--drafts-dir", type=str, default=None,
                        help="Directory containing r2gen_drafts_{train,val,test}.json")
    parser.add_argument("--image-root", type=str, default=None,
                        help="Root for relative image paths in draft files.")
    parser.add_argument("--skip-sft", action="store_true",
                        help="Skip SFT phase and go directly to DPO")
    parser.add_argument("--no-augment", action="store_true",
                        help="Do not augment with synthetic errors")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n=== Loading DPO Training Data ===")
    train_ds = CorrectorDPODataset("train", max_samples=args.max_train,
                                   use_synthetic_augment=not args.no_augment,
                                   drafts_dir=args.drafts_dir,
                                   image_root=args.image_root)
    val_ds = CorrectorDPODataset("val", max_samples=args.max_val,
                                 use_synthetic_augment=False,
                                 drafts_dir=args.drafts_dir,
                                 image_root=args.image_root)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=corrector_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=corrector_collate, num_workers=0)

    # Build model
    print("\n=== Building Corrector Model ===")
    model = MultimodalCorrector(
        model_name=args.model,
        use_qlora=not args.no_qlora,
        local_files_only=args.local_files_only,
    )

    ckpt_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "checkpoints" / "corrector_medgemma_swdpo"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    history = []
    best_bleu = 0.0

    # ── Phase 1: SFT ──
    if not args.skip_sft and args.sft_epochs > 0:
        print("\n=== Phase 1: Supervised Fine-Tuning ===")
        sft_params = [p for p in model.parameters() if p.requires_grad]
        sft_optimizer = torch.optim.AdamW(sft_params, lr=args.lr, weight_decay=0.01)

        for epoch in range(1, args.sft_epochs + 1):
            t0 = time.time()
            sft_loss = train_epoch_sft(
                model, train_loader, sft_optimizer, device, epoch,
                grad_accum_steps=args.grad_accum,
            )
            elapsed = time.time() - t0
            print(f"  SFT Epoch {epoch}/{args.sft_epochs} ({elapsed:.1f}s) | loss: {sft_loss:.4f}")
            history.append({"phase": "sft", "epoch": epoch, "train_loss": sft_loss})
            gc.collect()
            torch.cuda.empty_cache()

        # Save SFT checkpoint
        if hasattr(model.model, "save_pretrained"):
            model.model.save_pretrained(str(ckpt_dir / "lora_sft"))
        print("  SFT phase complete.")

    # ── Phase 2: SW-DPO ──
    if args.dpo_epochs > 0:
        print("\n=== Phase 2: Severity-Weighted DPO ===")
        print("  Creating frozen reference model...")
        try:
            ref_model = copy.deepcopy(model)
            for p in ref_model.parameters():
                p.requires_grad_(False)
            ref_model.model.eval()
            print("  Reference model created (frozen, same as post-SFT weights).")
        except Exception as e:
            print(f"  WARNING: Could not create reference model ({e}). Using simplified DPO.")
            ref_model = None

        sw_dpo = SeverityWeightedDPOLoss(beta=args.dpo_beta)
        dpo_params = [p for p in model.parameters() if p.requires_grad]
        dpo_optimizer = torch.optim.AdamW(dpo_params, lr=args.dpo_lr, weight_decay=0.01)

        # Val data as list for generation-based eval
        val_items = [val_ds[i] for i in range(min(200, len(val_ds)))]

        for epoch in range(1, args.dpo_epochs + 1):
            t0 = time.time()
            dpo_loss = train_epoch_dpo(
                model, ref_model, train_loader, sw_dpo, dpo_optimizer,
                device, epoch, grad_accum_steps=args.grad_accum,
            )
            elapsed = time.time() - t0
            print(f"\n  DPO Epoch {epoch}/{args.dpo_epochs} ({elapsed:.1f}s) | loss: {dpo_loss:.4f}")

            # Generation-based evaluation every epoch
            print("  Evaluating generation quality...")
            val_metrics = evaluate_corrector(model, val_items, max_eval=args.val_eval_max)
            print(f"  Val BLEU-4: {val_metrics['bleu4']:.4f} (draft: {val_metrics['draft_bleu4']:.4f}, "
                  f"improvement: {val_metrics['improvement']:+.4f})")

            history.append({
                "phase": "dpo", "epoch": epoch, "train_loss": dpo_loss,
                "val_bleu4": val_metrics["bleu4"],
                "val_rouge_l": val_metrics["rouge_l"],
                "improvement": val_metrics["improvement"],
            })

            if val_metrics["bleu4"] > best_bleu:
                best_bleu = val_metrics["bleu4"]
                if hasattr(model.model, "save_pretrained"):
                    model.model.save_pretrained(str(ckpt_dir / "lora_best"))
                    model.processor.save_pretrained(str(ckpt_dir / "processor"))
                    print(f"  -> Saved best model (BLEU-4={best_bleu:.4f})")

            gc.collect()
            torch.cuda.empty_cache()

    # Final save
    if hasattr(model.model, "save_pretrained"):
        model.model.save_pretrained(str(ckpt_dir / "lora_final"))
        model.processor.save_pretrained(str(ckpt_dir / "processor"))

    results = {
        "config": vars(args),
        "best_bleu4": best_bleu,
        "history": history,
    }
    rpath = results_dir / "corrector_swdpo_results.json"
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Corrector SW-DPO Training Complete!")
    print(f"  Best val BLEU-4: {best_bleu:.4f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Results: {rpath}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

