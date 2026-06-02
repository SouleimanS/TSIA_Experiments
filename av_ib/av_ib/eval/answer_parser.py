"""Shared answer parsing for MUSIC-AVQA eval.

Vocab-constrained matching: model output is normalized, scanned for any
canonical vocab token, returns the canonical token if found; "??" otherwise.

Pre-normalization mappings (digit->word, location-noun->indoor/outdoor)
are defined a priori from English semantics, NOT fitted to gold labels.
"""
from __future__ import annotations

import re

ANSWER_VOCAB = {
    "yes", "no",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "more than ten",
    "left", "right", "middle",
    "indoor", "outdoor",
    "simultaneously",
    "violin", "cello", "piano", "flute", "guitar", "acoustic_guitar", "electric_bass",
    "clarinet", "saxophone", "accordion", "trumpet", "tuba", "trombone", "horn", "ukulele",
    "banjo", "pipa", "guzheng", "erhu", "suona", "xylophone",
    "drum", "congas", "bassoon", "bagpipe",   # observed in test set
}

DIGIT_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}

# Enclosed structures -> "indoor"
INDOOR_WORDS = {
    "room", "livingroom", "bedroom", "kitchen", "bathroom",
    "studio", "classroom", "auditorium", "hall", "church",
    "temple", "mosque", "synagogue", "cathedral",
    "stage", "theater", "theatre", "concert", "opera",
    "house", "home", "apartment", "building", "office",
    "gym", "gymnasium", "mall", "bar", "club", "lounge",
    "library", "museum", "gallery", "hallway", "corridor",
    "indoors", "inside",
}

# Open-air locations -> "outdoor"
OUTDOOR_WORDS = {
    "street", "road", "highway", "sidewalk", "pavement",
    "park", "garden", "yard", "lawn", "field", "meadow",
    "forest", "woods", "jungle", "desert", "mountain",
    "beach", "shore", "coast", "ocean", "lake", "river",
    "courtyard", "porch", "patio", "deck", "rooftop", "roof",
    "outdoors", "outside",
}


def parse_answer(text: str, vocab: set = ANSWER_VOCAB) -> str:
    """Match model output against canonical vocab. Returns "??" on no match.

    Pre-normalizes: strip punctuation, digits->words, location-nouns->binary.
    Then word-boundary scan for any vocab token (longest-first), with
    underscore-to-space variant for multi-word tokens like 'electric_bass'.
    """
    if not text:
        return "??"

    # Strip punctuation, lowercase
    head = text.strip().lower()
    head = re.sub(r"[.,!?;:\"'()]", " ", head)

    # Digit -> word
    head = re.sub(r"\b(\d+)\b",
                  lambda m: DIGIT_TO_WORD.get(m.group(1), m.group(1)),
                  head)

    # Location-noun -> indoor/outdoor
    # Replace whole-word matches so vocab scan below picks them up
    def _loc_sub(m):
        w = m.group(0)
        if w in INDOOR_WORDS:
            return "indoor"
        if w in OUTDOOR_WORDS:
            return "outdoor"
        return w
    head = re.sub(r"\b\w+\b", _loc_sub, head)

    # Vocab scan, longest-first to handle "more than ten" before "ten"
    for tok in sorted(vocab, key=lambda s: -len(s)):
        candidates = [tok]
        if "_" in tok:
            candidates.append(tok.replace("_", " "))
        for c in candidates:
            if re.search(r"\b" + re.escape(c) + r"\b", head):
                return tok

    return "??"
