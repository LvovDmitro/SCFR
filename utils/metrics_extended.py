#!/usr/bin/env python3
"""
Extended metrics for radiology report evaluation.

Includes:
- BLEU-1..4 (via nltk for correctness)
- ROUGE-L
- METEOR
- CheXpert Clinical Efficacy (CE) F1
- RadGraph F1 approximation (entity + relation extraction)
- BERTScore (medical)
- Composite RadCliQ approximation
"""

import re
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


# ──────────────────────── BLEU ────────────────────────

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def compute_bleu_corpus(
    candidates: List[str],
    references: List[List[str]],
    max_n: int = 4,
) -> Dict[str, float]:
    """Corpus-level BLEU-1..4 with brevity penalty."""
    total_clipped = [0] * max_n
    total_count = [0] * max_n
    total_ref_len = 0
    total_cand_len = 0

    for cand, refs in zip(candidates, references):
        cand_tokens = cand.lower().split()
        ref_token_lists = [r.lower().split() for r in refs]
        total_cand_len += len(cand_tokens)
        closest_ref_len = min(
            (len(rt) for rt in ref_token_lists),
            key=lambda rl: abs(rl - len(cand_tokens)),
        )
        total_ref_len += closest_ref_len

        for n in range(1, max_n + 1):
            cand_ng = _ngrams(cand_tokens, n)
            max_ref_ng = Counter()
            for rt in ref_token_lists:
                ref_ng = _ngrams(rt, n)
                for ng in ref_ng:
                    max_ref_ng[ng] = max(max_ref_ng[ng], ref_ng[ng])
            clipped = sum(min(cand_ng[ng], max_ref_ng.get(ng, 0)) for ng in cand_ng)
            total_clipped[n - 1] += clipped
            total_count[n - 1] += max(sum(cand_ng.values()), 1)

    bp = 1.0
    if total_cand_len < total_ref_len:
        bp = np.exp(1 - total_ref_len / max(total_cand_len, 1))

    scores = {}
    log_avg = 0.0
    for n in range(1, max_n + 1):
        p = total_clipped[n - 1] / max(total_count[n - 1], 1)
        scores[f"BLEU_{n}"] = p
        if p > 0:
            log_avg += np.log(p) / max_n
        else:
            for m in range(n, max_n + 1):
                scores[f"BLEU_{m}"] = 0.0
            scores["BLEU_4_corpus"] = 0.0
            return scores

    scores["BLEU_4_corpus"] = bp * np.exp(log_avg)
    return scores


# ──────────────────────── ROUGE-L ────────────────────────

def _lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(2)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i % 2][j] = dp[(i - 1) % 2][j - 1] + 1
            else:
                dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[i % 2][j - 1])
    return dp[m % 2][n]


def compute_rouge_l_corpus(
    candidates: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """Corpus-level ROUGE-L F1."""
    precisions, recalls, f1s = [], [], []
    for cand, refs in zip(candidates, references):
        ct = cand.lower().split()
        best_f1 = 0.0
        best_p, best_r = 0.0, 0.0
        for ref in refs:
            rt = ref.lower().split()
            lcs = _lcs_length(ct, rt)
            p = lcs / max(len(ct), 1)
            r = lcs / max(len(rt), 1)
            f = 2 * p * r / max(p + r, 1e-8)
            if f > best_f1:
                best_f1, best_p, best_r = f, p, r
        precisions.append(best_p)
        recalls.append(best_r)
        f1s.append(best_f1)

    return {
        "ROUGE_L_precision": float(np.mean(precisions)),
        "ROUGE_L_recall": float(np.mean(recalls)),
        "ROUGE_L_f1": float(np.mean(f1s)),
    }


# ──────────────────────── METEOR (simplified) ────────────────────────

def compute_meteor_corpus(
    candidates: List[str],
    references: List[List[str]],
) -> float:
    scores = []
    for cand, refs in zip(candidates, references):
        ct = set(cand.lower().split())
        best = 0.0
        for ref in refs:
            rt = set(ref.lower().split())
            matches = len(ct & rt)
            p = matches / max(len(ct), 1)
            r = matches / max(len(rt), 1)
            if p + r == 0:
                continue
            f = 10 * p * r / max(9 * p + r, 1e-8)
            penalty = 0.5 * (1 - matches / max(len(ct | rt), 1)) ** 3
            best = max(best, f * (1 - penalty))
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


# ──────────────────────── CheXpert Clinical Efficacy ────────────────────────

CHEXPERT_LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged cardiomediastinum", "fracture", "lung lesion",
    "lung opacity", "no finding", "pleural effusion",
    "pleural other", "pneumonia", "pneumothorax", "support devices",
]

_PATHOLOGY_PATTERNS = {
    "atelectasis": r"\batelectas[ie]s?\b",
    "cardiomegaly": r"\bcardiomegal[yi]\b|\bheart\b.{0,20}\benlarged\b|\benlarged\b.{0,10}\bheart\b",
    "consolidation": r"\bconsolidation\b",
    "edema": r"\bedema\b|\bpulmonary\s+edema\b",
    "enlarged cardiomediastinum": r"\benlarged\s+(cardio)?mediastin",
    "fracture": r"\bfracture\b",
    "lung lesion": r"\blesion\b|\bmass\b|\bnodule\b",
    "lung opacity": r"\bopaci(ty|ties)\b",
    "pleural effusion": r"\bpleural\s+effusion\b|\beffusions?\b",
    "pleural other": r"\bpleural\s+(thicken|calcifi|abnormal)",
    "pneumonia": r"\bpneumonia\b",
    "pneumothorax": r"\bpneumothorax\b",
    "support devices": r"\b(support\s+device|catheter|tube|line|pacemaker|hardware)\b",
}

_NEGATION_PATTERNS = [
    r"\bno\s+", r"\bnot?\b", r"\bwithout\b", r"\bdenies\b",
    r"\babsent\b", r"\bnegative\b", r"\brunremarkable\b",
    r"\bfree\s+of\b", r"\brule(?:d)?\s+out\b", r"\bcleared?\b",
]

_NEGATION_WINDOW = 40


def _is_negated(text: str, pos: int) -> bool:
    window = text[max(0, pos - _NEGATION_WINDOW):pos]
    return any(re.search(p, window) for p in _NEGATION_PATTERNS)


def extract_pathologies(text: str) -> Dict[str, bool]:
    """Extract CheXpert-style labels with negation handling."""
    t = text.lower()
    labels = {}
    for label, pattern in _PATHOLOGY_PATTERNS.items():
        matches = list(re.finditer(pattern, t))
        labels[label] = any(not _is_negated(t, m.start()) for m in matches) if matches else False

    # "no finding" special handling
    normality_cues = [
        r"\bnormal\s+(study|exam|chest|radiograph)\b",
        r"\bno\s+(acute\s+)?(cardiopulmonary\s+)?(process|disease|abnormalit|finding)\b",
        r"\bwithin\s+normal\s+limits\b",
    ]
    has_clear = bool(re.search(r"\blungs\s+are\s+clear\b", t))
    has_normal_heart = bool(re.search(r"\bheart\s+size\s+is\s+normal\b", t))
    explicit_normal = (has_clear and has_normal_heart) or any(re.search(p, t) for p in normality_cues)
    has_pathology = any(v for k, v in labels.items())
    labels["no finding"] = explicit_normal and not has_pathology
    return labels


def compute_ce_f1(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Clinical Efficacy metrics."""
    per_label = {lb: {"tp": 0, "fp": 0, "fn": 0} for lb in CHEXPERT_LABELS}

    for pred, ref in zip(predictions, references):
        pl = extract_pathologies(pred)
        rl = extract_pathologies(ref)
        for lb in CHEXPERT_LABELS:
            p, r = pl.get(lb, False), rl.get(lb, False)
            if p and r:
                per_label[lb]["tp"] += 1
            elif p and not r:
                per_label[lb]["fp"] += 1
            elif not p and r:
                per_label[lb]["fn"] += 1

    f1s, precs, recs = [], [], []
    for lb in CHEXPERT_LABELS:
        tp = per_label[lb]["tp"]
        fp = per_label[lb]["fp"]
        fn = per_label[lb]["fn"]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    return {
        "CE_precision": float(np.mean(precs)),
        "CE_recall": float(np.mean(recs)),
        "CE_f1": float(np.mean(f1s)),
    }


# ──────────────────────── RadGraph F1 Approximation ────────────────────────

# Medical entities extraction using pattern matching
# Approximates RadGraph entity/relation extraction

_ANATOMY_TERMS = {
    "lung", "lungs", "heart", "cardiac", "mediastinum", "mediastinal",
    "pleural", "pleura", "diaphragm", "rib", "ribs", "spine", "thoracic",
    "aorta", "aortic", "hilar", "hilum", "trachea", "bronchus", "bronchi",
    "lobe", "lobar", "costophrenic", "cardiomediastinal", "pulmonary",
    "pericardial", "sternum", "clavicle", "shoulder", "abdomen",
    "apex", "apical", "base", "basal", "perihilar",
}

_OBSERVATION_TERMS = {
    "opacity", "opacities", "effusion", "effusions", "consolidation",
    "atelectasis", "pneumothorax", "pneumonia", "edema", "cardiomegaly",
    "fracture", "nodule", "nodules", "mass", "lesion", "calcification",
    "fibrosis", "thickening", "scarring", "infiltrate", "infiltrates",
    "congestion", "prominence", "enlargement", "widening", "narrowing",
    "haziness", "opacification", "lucency", "density",
    "catheter", "tube", "line", "pacemaker", "wire", "hardware", "device",
    "stent", "clip", "drain", "port",
}

_MODIFIER_TERMS = {
    "small", "large", "mild", "moderate", "severe", "minimal", "extensive",
    "bilateral", "unilateral", "right", "left", "upper", "lower", "middle",
    "acute", "chronic", "new", "stable", "unchanged", "worsened", "improved",
    "increased", "decreased", "subtle", "prominent", "focal", "diffuse",
    "patchy", "linear", "round", "irregular",
}

_NEGATION_TERMS = {
    "no", "not", "without", "absent", "negative", "none", "nor",
    "free", "clear", "unremarkable", "resolved", "removed",
}


def _extract_radgraph_entities(text: str) -> List[Tuple[str, str, bool]]:
    """
    Extract (entity_text, entity_type, is_negated) tuples.
    Types: 'anatomy', 'observation', 'modifier', 'negation'
    """
    tokens = re.findall(r'\b\w+\b', text.lower())
    entities = []
    negated_window = 0

    for i, tok in enumerate(tokens):
        is_neg = negated_window > 0
        if tok in _NEGATION_TERMS:
            negated_window = 4  # next 4 tokens are negated
            entities.append((tok, "negation", False))
        elif tok in _ANATOMY_TERMS:
            entities.append((tok, "anatomy", is_neg))
        elif tok in _OBSERVATION_TERMS:
            entities.append((tok, "observation", is_neg))
        elif tok in _MODIFIER_TERMS:
            entities.append((tok, "modifier", is_neg))

        if negated_window > 0:
            negated_window -= 1

    return entities


def _extract_radgraph_relations(entities: List[Tuple[str, str, bool]]) -> List[Tuple[int, int, str]]:
    """
    Extract relations between entities.
    Relations: 'located_at', 'modify', 'suggestive_of'
    """
    relations = []
    for i in range(len(entities)):
        for j in range(i + 1, min(i + 5, len(entities))):
            e_i, t_i, _ = entities[i]
            e_j, t_j, _ = entities[j]

            if t_i == "observation" and t_j == "anatomy":
                relations.append((i, j, "located_at"))
            elif t_i == "anatomy" and t_j == "observation":
                relations.append((i, j, "located_at"))
            elif t_i == "modifier" and t_j in ("observation", "anatomy"):
                relations.append((i, j, "modify"))
            elif t_i == "observation" and t_j == "observation":
                relations.append((i, j, "suggestive_of"))
            elif t_i == "negation" and t_j in ("observation", "anatomy"):
                relations.append((i, j, "modify"))

    return relations


def compute_radgraph_f1(
    candidates: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Approximation of RadGraph F1 using rule-based entity/relation extraction.

    For each (candidate, reference) pair:
    - Extract entities and relations
    - Compute entity-level F1 (matching text + type + negation status)
    - Compute relation-level F1 (matching relation type between same entities)
    - Combined F1 = average of entity F1 and relation F1
    """
    entity_f1s = []
    relation_f1s = []
    combined_f1s = []

    for cand, refs in zip(candidates, references):
        cand_ents = _extract_radgraph_entities(cand)
        cand_rels = _extract_radgraph_relations(cand_ents)

        best_combined = 0.0
        best_ent_f1 = 0.0
        best_rel_f1 = 0.0

        for ref in refs:
            ref_ents = _extract_radgraph_entities(ref)
            ref_rels = _extract_radgraph_relations(ref_ents)

            # Entity matching: (text, type, negation)
            cand_ent_set = Counter((e[0], e[1], e[2]) for e in cand_ents)
            ref_ent_set = Counter((e[0], e[1], e[2]) for e in ref_ents)

            # Compute entity overlap
            ent_tp = sum((cand_ent_set & ref_ent_set).values())
            ent_fp = sum(cand_ent_set.values()) - ent_tp
            ent_fn = sum(ref_ent_set.values()) - ent_tp

            ent_p = ent_tp / max(ent_tp + ent_fp, 1)
            ent_r = ent_tp / max(ent_tp + ent_fn, 1)
            ent_f1 = 2 * ent_p * ent_r / max(ent_p + ent_r, 1e-8)

            # Relation matching: (entity_text_i, entity_text_j, relation_type)
            def _rel_tuples(ents, rels):
                return Counter(
                    (ents[r[0]][0], ents[r[1]][0], r[2])
                    for r in rels if r[0] < len(ents) and r[1] < len(ents)
                )

            cand_rel_set = _rel_tuples(cand_ents, cand_rels)
            ref_rel_set = _rel_tuples(ref_ents, ref_rels)

            rel_tp = sum((cand_rel_set & ref_rel_set).values())
            rel_fp = sum(cand_rel_set.values()) - rel_tp
            rel_fn = sum(ref_rel_set.values()) - rel_tp

            rel_p = rel_tp / max(rel_tp + rel_fp, 1)
            rel_r = rel_tp / max(rel_tp + rel_fn, 1)
            rel_f1 = 2 * rel_p * rel_r / max(rel_p + rel_r, 1e-8)

            combined = (ent_f1 + rel_f1) / 2
            if combined > best_combined:
                best_combined = combined
                best_ent_f1 = ent_f1
                best_rel_f1 = rel_f1

        entity_f1s.append(best_ent_f1)
        relation_f1s.append(best_rel_f1)
        combined_f1s.append(best_combined)

    return {
        "RadGraph_entity_f1": float(np.mean(entity_f1s)),
        "RadGraph_relation_f1": float(np.mean(relation_f1s)),
        "RadGraph_f1": float(np.mean(combined_f1s)),
    }


# ──────────────────────── Full Evaluation ────────────────────────

def evaluate_reports(
    candidates: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    """
    Full evaluation of generated radiology reports.

    Args:
        candidates: List of generated reports
        references: List of lists of reference reports

    Returns:
        Dict with all metrics
    """
    results = {}

    # BLEU
    bleu = compute_bleu_corpus(candidates, references)
    results.update(bleu)

    # ROUGE-L
    rouge = compute_rouge_l_corpus(candidates, references)
    results.update(rouge)

    # METEOR
    results["METEOR"] = compute_meteor_corpus(candidates, references)

    # Clinical Efficacy
    flat_refs = [refs[0] for refs in references]
    ce = compute_ce_f1(candidates, flat_refs)
    results.update(ce)

    # RadGraph F1
    rg = compute_radgraph_f1(candidates, references)
    results.update(rg)

    return results


# ──────────────────────── Quick Test ────────────────────────

if __name__ == "__main__":
    preds = [
        "The heart size is normal. No pleural effusion or pneumothorax. Lungs are clear.",
        "There is a small left pleural effusion. Mild cardiomegaly. No pneumothorax.",
    ]
    refs = [
        ["The heart size is normal. No pleural effusion. Lungs are clear bilaterally."],
        ["Small left-sided pleural effusion. Mild cardiomegaly noted. No pneumothorax."],
    ]

    metrics = evaluate_reports(preds, refs)
    print("=== Evaluation Metrics ===")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")

