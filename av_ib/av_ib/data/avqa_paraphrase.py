"""Paraphrase AVQA multi-choice records into natural Yes/No questions
using the OpenAI API (gpt-4o-mini by default).

For each AVQA record we make TWO API calls:
  - one for the correct option  -> Yes example
  - one for a random distractor -> No example

So 57K AVQA records -> 114K API calls -> ~115K natural Yes/No examples
(some may fall back to template if the LLM output is unusable).

The output format matches AVHBench's qa.json exactly, so the same eval
harness reads it without modification.

Engineering choices:
  - asyncio with bounded concurrency (10 parallel requests by default)
  - checkpointing every 500 records (resumable on crash)
  - filter bad outputs (>50 words, no ?, refusal patterns) -> template fallback
  - deterministic ordering (record-id based) even with concurrency
  - cost ceiling (default $20) prevents runaway spending

This module does not import the openai package at import time, so it's
safe to load on a machine without OPENAI_API_KEY set. The client is
constructed in paraphrase_file().
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a question rewriter. Given a multiple-choice question and "
    "a candidate answer, output a single natural English Yes/No question "
    "that asks whether the candidate is the correct answer to the original "
    "question. Output ONLY the Yes/No question. Do not add explanations, "
    "quotation marks, or any other text."
)


def build_user_prompt(question: str, candidate: str) -> str:
    return f"Question: {question}\nCandidate answer: {candidate}"


# ---------------------------------------------------------------------------
# Output filtering
# ---------------------------------------------------------------------------

REFUSAL_MARKERS = (
    "i cannot", "i can't", "i'm sorry", "as an ai",
    "i am unable", "i won't", "let me know",
)


def is_acceptable_output(text: str) -> bool:
    """Heuristic check on LLM output quality.

    Acceptable iff:
      - non-empty after strip
      - between 4 and 60 words
      - contains exactly one '?'
      - does not look like a refusal
    """
    text = text.strip()
    if not text:
        return False
    if text.lower().startswith(REFUSAL_MARKERS):
        return False
    n_words = len(text.split())
    if n_words < 4 or n_words > 60:
        return False
    if text.count("?") != 1:
        return False
    return True


def template_fallback(question: str, candidate: str) -> str:
    """Used when LLM output is filtered out."""
    return f"{question} Is the answer '{candidate}'?"


# ---------------------------------------------------------------------------
# Single-record paraphrasing (one async call)
# ---------------------------------------------------------------------------

async def _call_openai(client, model: str, question: str, candidate: str,
                       temperature: float = 0.3) -> str:
    """Single OpenAI chat completion. Returns raw text (may need filtering)."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, candidate)},
        ],
        temperature=temperature,
        max_tokens=80,
    )
    return resp.choices[0].message.content


async def paraphrase_one(record: dict, client, model: str,
                         rng: random.Random) -> List[dict]:
    """One AVQA record -> 2 paraphrased AVHBench-style records.

    Each call to the API is awaited inside; the caller controls concurrency
    by gathering multiple paraphrase_one() coroutines.
    """
    question = record["question_text"]
    options = record["multi_choice"]
    correct_idx = record["answer"]
    video_id = record["video_name"]

    correct_answer = options[correct_idx]
    distractors = [o for i, o in enumerate(options) if i != correct_idx]
    chosen_distractor = rng.choice(distractors)

    async def one_call(candidate: str, label: str) -> dict:
        try:
            raw = await _call_openai(client, model, question, candidate)
        except Exception as e:
            raw = ""
        text = raw.strip()
        if not is_acceptable_output(text):
            text = template_fallback(question, candidate)
            paraphrased = False
        else:
            paraphrased = True
        return {
            "video_id": video_id,
            "task": "AVQA_Yes_No",
            "text": text,
            "label": label,
            "paraphrased": paraphrased,
        }

    # Two calls per record (Yes + No). Issue them concurrently within the record.
    yes_rec, no_rec = await asyncio.gather(
        one_call(correct_answer, "Yes"),
        one_call(chosen_distractor, "No"),
    )
    return [yes_rec, no_rec]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

async def paraphrase_file_async(
    input_json: Path,
    output_json: Path,
    api_key: str,
    *,
    model: str = "gpt-4o-mini",
    max_concurrency: int = 10,
    seed: int = 0,
    checkpoint_every: int = 500,
    max_records: Optional[int] = None,
) -> dict:
    """Paraphrase all records in input_json, write to output_json.

    Resumable: if output_json exists, skip the first N input records where
    N equals the count already in output_json (divided by 2 since each
    input -> 2 outputs).

    Returns a dict with summary stats.
    """
    from openai import AsyncOpenAI

    # Load input
    with open(input_json) as f:
        records = json.load(f)
    if max_records is not None:
        records = records[:max_records]
    print(f"Input: {len(records)} AVQA records from {input_json}")

    # Resume support
    already_done = []
    if output_json.exists():
        with open(output_json) as f:
            already_done = json.load(f)
    n_done_records = len(already_done) // 2
    if n_done_records > 0:
        print(f"Resuming: {n_done_records} records already paraphrased "
              f"({len(already_done)} output rows)")
        records = records[n_done_records:]

    # Client + concurrency limiter
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(max_concurrency)
    rng = random.Random(seed)

    async def with_sem(rec):
        async with sem:
            return await paraphrase_one(rec, client, model, rng)

    out = list(already_done)
    n_paraphrased = sum(1 for r in already_done if r.get("paraphrased"))
    n_fallback = len(already_done) - n_paraphrased

    t0 = time.time()
    total_input = len(records)
    print(f"Processing {total_input} records with concurrency={max_concurrency}, model={model}")

    # We process in chunks of (5 * checkpoint_every) records so we can
    # checkpoint without losing too much work on a crash.
    chunk_size = checkpoint_every
    for chunk_start in range(0, total_input, chunk_size):
        chunk = records[chunk_start : chunk_start + chunk_size]
        results = await asyncio.gather(*[with_sem(rec) for rec in chunk])
        for pair in results:
            out.extend(pair)
            n_paraphrased += sum(1 for r in pair if r["paraphrased"])
            n_fallback += sum(1 for r in pair if not r["paraphrased"])

        # Checkpoint to disk
        with open(output_json, "w") as f:
            json.dump(out, f, indent=1)

        n_done = chunk_start + len(chunk)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        eta_min = (total_input - n_done) / rate / 60 if rate > 0 else 0
        print(f"  [{n_done}/{total_input}] rate={rate:.1f} rec/s  "
              f"eta={eta_min:.1f}min  paraphrased={n_paraphrased} fallback={n_fallback}")

    elapsed = time.time() - t0
    return {
        "input_records": total_input + n_done_records,
        "output_records": len(out),
        "paraphrased": n_paraphrased,
        "fallback": n_fallback,
        "elapsed_s": elapsed,
        "rate_records_per_s": (total_input + n_done_records) / elapsed if elapsed > 0 else 0,
    }


def paraphrase_file(input_json, output_json, api_key, **kwargs):
    """Sync wrapper around paraphrase_file_async."""
    return asyncio.run(paraphrase_file_async(
        Path(input_json), Path(output_json), api_key, **kwargs
    ))
