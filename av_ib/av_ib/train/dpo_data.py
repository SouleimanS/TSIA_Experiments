"""Preference-pair dataset for FINER-inspired DPO abstention training.

Consumes the empirical abstain-label miner output (runs/abstain_labels.json,
produced by `explain.py alabels`) and turns it into (prompt, corruption,
chosen, rejected) preference pairs. No schema change to the miner is required.

Pair construction (see docs/abstention_dpo_plan.md):

    abstain pair      id_correct & vid-flip & not aud-flip
                      corruption=vid_zero/mean  chosen="I can't tell"  rejected=gold
    capability anchor id_correct
                      corruption=identity       chosen=gold            rejected="I can't tell"
    negative control  id_correct & aud-flip
                      corruption=aud_zero/mean  chosen=gold            rejected="I can't tell"

The abstain pairs teach refusal when the essential (video) evidence is removed;
the anchor and negative-control pairs stop the model from collapsing into
"I can't tell" everywhere (preserve coverage), exactly the capability-preservation
that FINER-Tuning reports for DPO.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from torch.utils.data import Dataset


ABSTAIN_STR = "I can't tell"
ABSTAIN_PROMPT_SUFFIX = (
    " Answer with one word from the allowed vocabulary, "
    "or say \"I can't tell\" if the audio or video does not support an answer."
)


def _gold_of(rec: dict) -> str:
    """Raw MUSIC-AVQA record -> gold answer string. Note: the raw schema spells
    the answer field 'anser'."""
    for k in ("anser", "answer", "gold"):
        if k in rec and rec[k] is not None:
            return str(rec[k]).strip()
    raise KeyError(f"no answer field in record keys={list(rec.keys())}")


def _prompt_of(rec: dict) -> str:
    """Raw record -> rendered question prompt, matching MusicAVQADataset."""
    if "prompt" in rec and rec["prompt"]:
        return rec["prompt"]
    from av_ib.data.musicavqa import render_question
    return render_question(rec["question_content"], rec["templ_values"])


def _video_path_of(rec: dict, video_root: str | Path) -> str:
    """Raw record -> absolute video path, matching MusicAVQADataset."""
    if rec.get("video_path"):
        return rec["video_path"]
    return str(Path(video_root) / f"{rec['video_id']}.mp4")


def build_pairs(labels_json: str | Path,
                video_root: str | Path,
                *,
                include_anchor: bool = True,
                include_negctrl: bool = True,
                anchor_ratio: float = 1.0,
                with_abstain_suffix: bool = True) -> List[dict]:
    """Return a flat list of preference-pair dicts.

    Each pair: {video_path, audio_path, prompt, corr, chosen, rejected, kind}
    `corr` is a corruption mode understood by explain.make_ablation
    ("identity", "vid_zero", "vid_mean", "aud_zero", "aud_mean").
    """
    recs = json.loads(Path(labels_json).read_text())
    pairs: List[dict] = []
    n_abstain = 0

    suffix = ABSTAIN_PROMPT_SUFFIX if with_abstain_suffix else ""

    for rec in recs:
        if not rec.get("identity_correct", False):
            continue  # only items the model got right under clean evidence
        try:
            gold = _gold_of(rec)
            prompt = _prompt_of(rec) + suffix
            vp = _video_path_of(rec, video_root)
        except Exception:
            continue  # skip malformed records (e.g. bad templ_values)
        ap = vp  # MUSIC-AVQA: audio extracted from the same mp4

        vid_flip_zero = rec.get("flips_vid_zero", False)
        vid_flip_mean = rec.get("flips_vid_mean", False)
        aud_flip = rec.get("flips_aud_zero", False) or rec.get("flips_aud_mean", False)
        vid_flip = vid_flip_zero or vid_flip_mean

        # ── abstain pairs: video removed -> prefer refusal over the gold ──
        # Only clean sufficiency signal: video flips but audio does not.
        if vid_flip and not aud_flip:
            for mode, flag in (("vid_zero", vid_flip_zero), ("vid_mean", vid_flip_mean)):
                if flag:
                    pairs.append({
                        "video_path": vp, "audio_path": ap, "prompt": prompt,
                        "corr": mode, "chosen": ABSTAIN_STR, "rejected": gold,
                        "kind": "abstain",
                    })
                    n_abstain += 1

        # ── capability anchor: intact evidence -> prefer the gold over refusal ──
        if include_anchor:
            pairs.append({
                "video_path": vp, "audio_path": ap, "prompt": prompt,
                "corr": "identity", "chosen": gold, "rejected": ABSTAIN_STR,
                "kind": "anchor",
            })

        # ── negative control: only audio degraded -> still prefer answering ──
        if include_negctrl and aud_flip:
            mode = "aud_zero" if rec.get("flips_aud_zero", False) else "aud_mean"
            pairs.append({
                "video_path": vp, "audio_path": ap, "prompt": prompt,
                "corr": mode, "chosen": gold, "rejected": ABSTAIN_STR,
                "kind": "negctrl",
            })

    # Optionally subsample anchors to keep the abstain signal from being drowned.
    if include_anchor and anchor_ratio < 1.0 and n_abstain > 0:
        import random
        anchors = [p for p in pairs if p["kind"] == "anchor"]
        others = [p for p in pairs if p["kind"] != "anchor"]
        keep = max(1, int(n_abstain * anchor_ratio))
        random.Random(0).shuffle(anchors)
        pairs = others + anchors[:keep]

    return pairs


class PreferencePairDataset(Dataset):
    def __init__(self, labels_json: str | Path, video_root: str | Path, **kw):
        self.pairs = build_pairs(labels_json, video_root, **kw)
        if not self.pairs:
            raise RuntimeError(
                f"no preference pairs built from {labels_json}; "
                "check identity_correct / flips_* fields exist")
        kinds = {}
        for p in self.pairs:
            kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
        self.kind_counts = kinds

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def _collate(batch):
    # batch_size is 1 in this single-GPU regime (Qwen processor not fork-safe)
    return batch[0]
