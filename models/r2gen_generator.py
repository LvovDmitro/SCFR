#!/usr/bin/env python3
"""
R2Gen Generator wrapper for the report generation pipeline.
Loads R2Gen model and provides generate(image_paths) -> report.

Key fix: Uses pre-saved vocabulary files (vocab_iu_xray.json, vocab_mimic_cxr.json)
to ensure the tokenizer matches the checkpoint's expected vocabulary.
The original code rebuilds vocab from annotation data, which can produce mismatched
token<->id mappings if a different annotation file is used than during training.
"""
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
R2GEN_ROOT = PROJECT_ROOT / "external" / "R2Gen"
sys.path.insert(0, str(R2GEN_ROOT))


def _find_vocab_path(dataset_name: str, model_name: str = "r2gen") -> Optional[str]:
    """Find pre-saved vocabulary JSON for checkpoint compatibility."""
    if dataset_name in ("mimic", "mimic_cxr"):
        vocab_name = "vocab_mimic_cxr.json"
    else:
        vocab_name = "vocab_iu_xray.json"

    candidates = [
        PROJECT_ROOT / "checkpoints" / model_name / vocab_name,
        PROJECT_ROOT / "checkpoints" / "r2gen" / vocab_name,
        PROJECT_ROOT / "data" / "r2gen_vocab" / vocab_name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _find_annotation_path(dataset_name: str) -> str:
    """Find annotation JSON for tokenizer vocab building (fallback if no saved vocab)."""
    if dataset_name in ("mimic", "mimic_cxr"):
        candidates = [
            R2GEN_ROOT / "data" / "mimic_cxr" / "annotation.json",
            PROJECT_ROOT / "Datasets" / "annotation.json",
            PROJECT_ROOT / "data" / "mimic_annotation.json",
        ]
    else:
        # Use annotation_official_raw.json for IU-Xray (matches official R2Gen checkpoint)
        candidates = [
            PROJECT_ROOT / "Datasets" / "annotation_official_raw.json",
            R2GEN_ROOT / "data" / "iu_xray" / "annotation.json",
            PROJECT_ROOT / "Datasets" / "annotation.json",
        ]

    for p in candidates:
        if p.exists():
            return str(p)

    return str(R2GEN_ROOT / "data" / "iu_xray" / "annotation.json")


def load_r2gen_generator(
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda",
    dataset_name: str = "mimic",
) -> callable:
    """
    Load R2Gen model and return a generator function.
    checkpoint_path: path to model checkpoint
    dataset_name: "mimic" or "iuxray" — determines tokenizer vocab and model forward
    """
    from modules.tokenizers import Tokenizer
    from models.r2gen import R2GenModel

    import argparse

    is_mimic = dataset_name in ("mimic", "mimic_cxr")
    # Map to R2Gen's expected dataset_name values
    r2gen_dataset_name = "mimic_cxr" if is_mimic else "iu_xray"

    ann_path = _find_annotation_path(dataset_name)
    vocab_path = _find_vocab_path(dataset_name, "r2gen")

    print(f"[R2Gen] dataset_name={dataset_name} -> r2gen_dataset_name={r2gen_dataset_name}")
    print(f"[R2Gen] Using annotation: {ann_path}")
    if vocab_path:
        print(f"[R2Gen] Using pre-saved vocab: {vocab_path}")

    args = argparse.Namespace(
        image_dir=str(R2GEN_ROOT / "data" / "iu_xray" / "images"),
        ann_path=ann_path,
        vocab_path=vocab_path,
        dataset_name=r2gen_dataset_name,
        max_seq_length=100 if is_mimic else 60,
        threshold=10 if is_mimic else 3,
        visual_extractor="resnet101",
        visual_extractor_pretrained=True,
        d_model=512,
        d_ff=512,
        d_vf=2048,
        num_heads=8,
        num_layers=3,
        dropout=0.1,
        logit_layers=1,
        bos_idx=0,
        eos_idx=0,
        pad_idx=0,
        use_bn=0,
        drop_prob_lm=0.5,
        rm_num_slots=3,
        rm_num_heads=8,
        rm_d_model=512,
        sample_method="beam_search",
        beam_size=3,
        temperature=1.0,
        sample_n=1,
        group_size=1,
        output_logsoftmax=1,
        decoding_constraint=0,
        block_trigrams=1,
    )

    tokenizer = Tokenizer(args)
    vocab_size = len(tokenizer.token2idx)
    print(f"[R2Gen] Tokenizer vocab size: {vocab_size}")

    # Build model with correct vocab size
    model = R2GenModel(args, tokenizer)

    ckpt_path = checkpoint_path or (PROJECT_ROOT / "checkpoints" / "r2gen" / "model_best.pth")
    if ckpt_path.exists():
        import os
        print(f"[R2Gen] Found checkpoint at {ckpt_path}, size: {os.path.getsize(ckpt_path)} bytes")
        try:
            try:
                state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            except TypeError:
                # Older PyTorch versions don't support weights_only
                state = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            # Check vocab compatibility
            if "encoder_decoder.logit.weight" in state:
                ckpt_vocab = state["encoder_decoder.logit.weight"].shape[0]
                model_vocab = vocab_size + 1  # tgt_vocab = vocab_size + 1 in EncoderDecoder
                if ckpt_vocab != model_vocab:
                    raise RuntimeError(
                        f"Vocab mismatch: checkpoint expects tgt_vocab={ckpt_vocab} "
                        f"but tokenizer built tgt_vocab={model_vocab}. "
                        f"Need correct vocab JSON or annotation.json matching the training data. "
                        f"(dataset={dataset_name}, ann={ann_path})"
                    )
                else:
                    print(f"[R2Gen] Vocab OK: ckpt tgt_vocab={ckpt_vocab} == model tgt_vocab={model_vocab}")
            model.load_state_dict(state, strict=False)
            print(f"[R2Gen] Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"[R2Gen] Failed to load checkpoint: {e}")
            print("[R2Gen] WARNING: using random init instead")
    else:
        print(f"[R2Gen] WARNING: No checkpoint at {ckpt_path}, using random init")

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    def generate_fixed(image_input):
        if isinstance(image_input, (str, Path)):
            paths = [str(image_input)]
        else:
            paths = [str(p) for p in image_input[:2]]

        images_list = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
                images_list.append(transform(img))
            except Exception:
                pass

        if not images_list:
            return "No findings."

        if is_mimic:
            # MIMIC: single image -> [1, C, H, W]
            images = images_list[0].unsqueeze(0).to(device)
        else:
            # IU-Xray: 2 views -> [1, 2, C, H, W]
            while len(images_list) < 2:
                images_list.append(images_list[0].clone())
            images = torch.stack(images_list).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(images, mode="sample")

        return _decode_r2gen_output(output, tokenizer)

    return generate_fixed


def _decode_r2gen_output(output, tokenizer) -> str:
    """Decode R2Gen model output (word ids) to text.

    The model outputs a tuple (seq, seqLogprobs) from _sample or _sample_beam.
    seq has shape [B, max_seq_length] with token IDs. ID 0 = EOS/PAD.
    """
    if isinstance(output, (tuple, list)):
        seq = output[0]  # [B, max_seq_length]
    else:
        seq = output

    if hasattr(seq, "cpu"):
        seq = seq.cpu()
    if isinstance(seq, torch.Tensor):
        ids = seq[0].tolist() if seq.dim() > 1 else seq.tolist()
    else:
        ids = seq[0] if isinstance(seq, (list, tuple)) else seq

    # Use the tokenizer's own decode method for consistency
    report = tokenizer.decode(ids)
    if not report.strip():
        return "No findings."

    # Clean up: deduplicate sentences
    text = report.replace(" . ", ". ").strip()
    sents = [s.strip() for s in text.split(".") if len(s.strip()) > 2]
    seen, unique = set(), []
    for s in sents:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    text = ". ".join(unique)
    if text and not text.endswith("."):
        text += "."
    return text

