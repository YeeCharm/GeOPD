"""Scoring rules for CHOICE, GEOBench-VLM, OmniEarth, HRVQA, and LHRS-Bench."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

_NUMBER_WORD_RE = re.compile(r"\b(?:" + "|".join(_NUMBER_WORDS) + r")\b")
_TERMINAL_INTRO_RE = re.compile(
    r"\b(?:the\s+)?(?:final\s+answer|correct\s+(?:answer|option)|answer)\s*(?:is\b|[:=])\s*",
    re.IGNORECASE,
)


def clean_answer(text: str | None) -> str:
    """Strip reasoning tags and common answer prefixes."""
    value = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE).strip()
    match = re.search(r"<answer>(.*?)</answer>", value, flags=re.DOTALL | re.IGNORECASE)
    if match:
        value = match.group(1)
    return re.sub(
        r"^\s*(?:final answer|answer|the answer is)\s*[:：]?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()


def normalize_text(text: Any) -> str:
    """Normalize short-answer strings for open-ended evaluation."""
    value = unicodedata.normalize("NFKC", clean_answer(str(text))).lower().strip()
    value = re.sub(r"\bgrey\b", "gray", value)
    value = _NUMBER_WORD_RE.sub(lambda m: _NUMBER_WORDS[m.group()], value)
    value = re.sub(r"^(?:a|an|the)\s+", "", value)
    return value.strip(" .!\n\t\"'")


def parse_choices(record: dict[str, Any]) -> dict[str, str]:
    """Parse lettered options (A-F) from the record's options or question field."""
    source = record.get("options") or record.get("question", "")
    matches = list(re.finditer(r"(?:^|[\s\n])\(?([A-F])\)?\s*[.:：]\s*", source))
    choices: dict[str, str] = {}
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(source)
        choices[match.group(1)] = source[match.end() : end].strip()
    return choices


def extract_option_letters(prediction: str, record: dict[str, Any]) -> set[str]:
    """Extract option letters from a model response without misreading English articles."""
    cleaned = clean_answer(prediction).strip("*` ")
    opts = parse_choices(record)

    if re.fullmatch(r"\(?[A-Fa-f]\)?(?:\s*[,;/& ]\s*\(?[A-Fa-f]\)?)*[.!]?", cleaned):
        found = set(re.findall(r"[A-Fa-f]", cleaned.upper()))
    else:
        found = {
            letter.upper()
            for letter in re.findall(
                r"(?:^|\boption\s+|\bchoice\s+)\(?([A-Fa-f])\)?(?=[.:：)]|\s*$)",
                cleaned,
                flags=re.IGNORECASE,
            )
        }

    if found and (not opts or found <= set(opts)):
        return found

    normalized_pred = normalize_text(cleaned)
    return {key for key, val in opts.items() if normalized_pred == normalize_text(val)}


def recover_terminal_option(record: dict[str, Any], prediction: str) -> str:
    """Fallback parser for explicit terminal declarations such as 'Therefore, the answer is B.'."""
    fmt = record.get("format")
    if fmt not in {"mcq", "multi"} or extract_option_letters(prediction, record):
        return prediction

    text = re.sub(r"<think>.*?</think>", "", prediction or "", flags=re.IGNORECASE | re.DOTALL).strip()
    if any(marker in text.lower() for marker in ("<think>", "</think>", "```")):
        return prediction

    declarations = list(_TERMINAL_INTRO_RE.finditer(text))
    if not declarations:
        return prediction

    tail = re.sub(r"\band\b", ",", text[declarations[-1].end() :].strip(), flags=re.IGNORECASE)
    if not tail or not re.fullmatch(r"[A-Fa-f\s,;/&().!*`]+", tail):
        return prediction

    labels = set(re.findall(r"[A-Fa-f]", tail.upper()))
    if not labels or (fmt == "mcq" and len(labels) != 1):
        return prediction

    candidate = ",".join(sorted(labels))
    if extract_option_letters(candidate, record) != labels:
        return prediction

    for declaration in declarations[:-1]:
        earlier = text[declaration.end() :]
        match = re.match(r"[*` (]*([A-Fa-f])(?:[)*`]*)(?=[.,;:!)]|\s*$)", earlier)
        if match and match.group(1).upper() not in labels:
            return prediction

    return candidate


def score_prediction(record: dict[str, Any], prediction: str) -> dict[str, Any]:
    """Score a single prediction against a manifest record.

    - ``mcq``: exact single-option accuracy (1.0 or 0.0).
    - ``multi``: 1.0 for exact option set, 0.5 for a nonempty strict subset, else 0.0.
    - ``open``: normalized exact string / exact numeric match.
    """
    fmt = record.get("format", "mcq")
    if fmt in {"mcq", "multi"}:
        recovered = recover_terminal_option(record, prediction)
        got = extract_option_letters(recovered, record)
        raw_gt = record["answer"]
        gt = set(raw_gt) if isinstance(raw_gt, list) else set(re.findall(r"[A-F]", str(raw_gt)))

        if fmt == "mcq":
            value = float(len(got) == 1 and got == gt)
        else:
            value = 1.0 if got == gt else 0.5 if (got and got < gt) else 0.0

        return {
            "score": value,
            "extracted": sorted(got),
            "reference": sorted(gt),
            "needs_review": False,
        }

    ref = normalize_text(record["answer"])
    pred = normalize_text(prediction)
    if pred == ref:
        return {"score": 1.0, "extracted": pred, "reference": ref, "needs_review": False}

    if re.fullmatch(r"-?\d+(?:\.\d+)?", ref):
        nums = re.findall(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)", pred)
        if len(nums) == 1:
            return {
                "score": float(float(nums[0]) == float(ref)),
                "extracted": nums[0],
                "reference": ref,
                "needs_review": False,
            }

    return {
        "score": 0.0,
        "extracted": pred,
        "reference": ref,
        "needs_review": bool(pred),
    }
