"""Tests for render_question in musicavqa.py.

Run with: python test_render_question.py
Should print 'All tests passed' and exit 0.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Allow importing musicavqa from sibling location
sys.path.insert(0, str(Path(__file__).resolve().parent))
from musicavqa import render_question


CASES_OK = [
    # (question_content, templ_values_str, expected)
    ("How many instruments are sounding in the video?", "[]",
     "How many instruments are sounding in the video?"),
    ("What kind of instrument is the <LRer> instrument? ", '["leftest"]',
     "What kind of instrument is the leftest instrument? "),
    ("Is the instrument on the <LR> louder than the instrument on the <LR>?",
     '["left", "right"]',
     "Is the instrument on the left louder than the instrument on the right?"),
    ("What is the instrument on the <LR> of <Object>?", '["left", "guzheng"]',
     "What is the instrument on the left of guzheng?"),
    # Chinese full-width question mark — renderer should pass it through unchanged
    ("How many sounding <Object> in the video？", '["clarinet"]',
     "How many sounding clarinet in the video？"),
    ("Is the <Object> in the video always playing?", '["accordion"]',
     "Is the accordion in the video always playing?"),
    # Repeated same value used twice — make sure left-to-right consumes correctly
    ("Are <Object> and <Object> playing together?", '["violin", "violin"]',
     "Are violin and violin playing together?"),
]


CASES_ERR = [
    # (question_content, templ_values_str, expected_error_substring)
    # Too few values
    ("Is the <LR> louder than the <LR>?", '["left"]', "Placeholder count mismatch"),
    # Too many values
    ("How many <Object>?", '["a", "b"]', "Placeholder count mismatch"),
    # Not JSON
    ("How many?", "not a json string", "not valid JSON"),
    # Not a list
    ("How many?", '"just a string"', "must be a list"),
]


def main() -> int:
    n_ok = n_fail = 0
    for q, tv, expected in CASES_OK:
        got = render_question(q, tv)
        if got == expected:
            n_ok += 1
        else:
            n_fail += 1
            print(f"FAIL OK-case:")
            print(f"  q       : {q!r}")
            print(f"  tv      : {tv!r}")
            print(f"  expected: {expected!r}")
            print(f"  got     : {got!r}")

    for q, tv, err_substr in CASES_ERR:
        try:
            got = render_question(q, tv)
            n_fail += 1
            print(f"FAIL ERR-case: should have raised but returned {got!r}")
            print(f"  q : {q!r}")
            print(f"  tv: {tv!r}")
        except ValueError as e:
            if err_substr in str(e):
                n_ok += 1
            else:
                n_fail += 1
                print(f"FAIL ERR-case: raised wrong error:")
                print(f"  q       : {q!r}")
                print(f"  tv      : {tv!r}")
                print(f"  expected substr: {err_substr!r}")
                print(f"  got error      : {e!r}")

    total = n_ok + n_fail
    if n_fail == 0:
        print(f"All {total} tests passed")
        return 0
    print(f"{n_fail}/{total} tests FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
