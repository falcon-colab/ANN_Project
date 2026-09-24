"""
src/s1_physics/llm_experiment.py

The generation half of the S1 LLM reporting experiment (protocol 10.1).

Question: can a language model turn a verified structured experimental record
into a short scientific summary without inventing, altering or misreporting
numbers, and does an automatic verifier catch it when it does?

Four arms. The first three are the ones section 10.1 requires; the fourth
measures the verifier itself.

  direct           the raw results JSON and a plain request for a summary
  constrained      the same JSON with explicit instructions to use only
                   values present in the source and to make no comparison the
                   record does not support
  generate_verify  constrained generation, then the deterministic verifier
                   runs, and if it finds anything the model is shown its own
                   errors and asked to correct them. Both the first-pass and
                   post-correction rates are reported, because the difference
                   is the value the verification loop actually adds
  injection        the verifier's own detection rate. A summary that is
                   faithful by construction is corrupted with known errors of
                   each type section 10.1 names, and we count how many the
                   verifier catches. Without this the other three arms only
                   tell you what the models did, not whether the checker
                   works

Providers: gemini and groq free tiers, a local ollama, and `mock`, which
needs no network and emits summaries with deliberately seeded error types so
the whole pipeline can be exercised and demonstrated offline.

Model identifiers change often, so nothing is hardcoded. Use --list-models to
ask your key what it can reach, then pass --model.

    set GEMINI_API_KEY=...          (PowerShell: $env:GEMINI_API_KEY="...")
    python src/s1_physics/llm_experiment.py --provider gemini --list-models
    python src/s1_physics/llm_experiment.py --provider gemini --model <name>
    python src/s1_physics/llm_experiment.py --provider mock --repeats 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "common"))
sys.path.append(str(Path(__file__).resolve().parent))
from config import ARTIFACT_DIR, SEED
from llm_verify import NUMBER_RE, build_record, flatten, verify

TIMEOUT = 120


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

TASK = (
    "Write a short scientific results summary, at most 150 words, for the "
    "results below. They come from a four-class micro-Doppler radar target "
    "classification study using nested five-by-five measurement-grouped "
    "cross-validation."
)

DIRECT_PROMPT = TASK + "\n\nResults:\n{record}\n"

CONSTRAINED_PROMPT = (
    TASK
    + "\n\nStrict rules:\n"
      "1. Use ONLY numbers that appear in the JSON below. Do not compute, "
      "estimate, round beyond two decimal places, or introduce any other "
      "figure.\n"
      "2. Do not state any comparison or ranking that the JSON does not "
      "directly support.\n"
      "3. Do not mention anything the JSON does not contain: no convergence "
      "behaviour, no hardware, no dataset details beyond those given.\n"
      "4. If you are unsure whether a number is in the source, leave it out.\n"
      "\nResults:\n{record}\n"
)

CORRECTION_PROMPT = (
    "Your summary was checked automatically against the source record and the "
    "following problems were found.\n\n{issues}\n\nRewrite the summary so that "
    "every number appears in the source record and every comparison matches "
    "it. Keep it under 150 words. Output only the corrected summary.\n\n"
    "Source record:\n{record}\n\nYour previous summary:\n{previous}\n"
)


def issues_text(report) -> str:
    lines = []
    for c in report.numeric:
        if c.verdict in ("ALTERED", "UNSUPPORTED"):
            lines.append(f"- the figure {c.text.strip()} is {c.verdict.lower()}"
                         + (f" ({c.matched})" if c.matched else "")
                         + f'; it appears in: "{c.sentence.strip()}"')
    for s in report.statements:
        if s.verdict == "INCORRECT":
            lines.append(f"- {s.detail}")
    return "\n".join(lines) if lines else "- none"


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

MAX_RETRIES = 5
# Per-day quota exhaustion is not worth retrying: no amount of backoff will
# clear it before the quota window resets. Per-minute limits are.
DAILY_QUOTA_MARKERS = ("perday", "per day", "requests_per_day", "quota_limit_value",
                       "GenerateRequestsPerDayPerProject")


def _post(url: str, payload: dict, headers: dict) -> dict:
    """
    POST with exponential backoff on rate limiting.

    Free tiers are tight: Gemini allows roughly 15 requests per minute, and a
    full run is three arms times the repeat count plus correction calls, so
    hitting 429 is the normal case rather than an exceptional one.
    """
    body = json.dumps(payload).encode()
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()
            if e.code == 429 and any(m.lower() in detail.lower()
                                     for m in DAILY_QUOTA_MARKERS):
                raise SystemExit(
                    "Daily quota exhausted for this model. Retrying will not "
                    "help until the window resets.\n  Options: wait, use a "
                    "different model with --model, or switch to a local "
                    "provider with --provider ollama, which has no quota.\n  "
                    + detail[:400])
            if e.code not in (429, 500, 502, 503, 504) or attempt == MAX_RETRIES - 1:
                raise urllib.error.URLError(
                    f"HTTP {e.code}: {detail[:400]}") from e
            wait = 2 ** attempt * 5
            print(f"    HTTP {e.code}, backing off {wait}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
    raise urllib.error.URLError("retries exhausted")


def _get(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        hint = {401: "the key is missing or malformed",
                403: "the key is rejected, restricted, or the service is not "
                     "available from this region",
                404: "wrong endpoint or model name"}.get(e.code, "")
        raise SystemExit(f"HTTP {e.code} from {url}\n  {hint}\n  {detail}")


class Provider:
    name = "base"

    def generate(self, prompt: str, temperature: float) -> str:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        return []


class Gemini(Provider):
    name = "gemini"
    BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str | None):
        self.key = os.getenv("GEMINI_API_KEY")
        if not self.key:
            raise SystemExit("set GEMINI_API_KEY")
        self.model = model or "gemini-2.5-flash"

    def list_models(self) -> list[str]:
        d = _get(f"{self.BASE}/models", {"x-goog-api-key": self.key})
        return sorted(m["name"].split("/")[-1] for m in d.get("models", [])
                      if "generateContent" in m.get("supportedGenerationMethods", []))

    def generate(self, prompt: str, temperature: float) -> str:
        d = _post(f"{self.BASE}/models/{self.model}:generateContent",
                  {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": temperature}},
                  {"x-goog-api-key": self.key})
        parts = d["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)


class Groq(Provider):
    name = "groq"
    BASE = "https://api.groq.com/openai/v1"

    def __init__(self, model: str | None):
        self.key = os.getenv("GROQ_API_KEY")
        if not self.key:
            raise SystemExit("set GROQ_API_KEY")
        self.model = model or "llama-3.3-70b-versatile"

    def list_models(self) -> list[str]:
        d = _get(f"{self.BASE}/models", {"Authorization": f"Bearer {self.key}"})
        return sorted(m["id"] for m in d.get("data", []))

    def generate(self, prompt: str, temperature: float) -> str:
        d = _post(f"{self.BASE}/chat/completions",
                  {"model": self.model, "temperature": temperature,
                   "messages": [{"role": "user", "content": prompt}]},
                  {"Authorization": f"Bearer {self.key}"})
        return d["choices"][0]["message"]["content"]


class Ollama(Provider):
    name = "ollama"
    BASE = "http://localhost:11434"

    def __init__(self, model: str | None):
        self.model = model or "llama3.1"

    def list_models(self) -> list[str]:
        return sorted(m["name"] for m in _get(f"{self.BASE}/api/tags", {})
                      .get("models", []))

    def generate(self, prompt: str, temperature: float) -> str:
        d = _post(f"{self.BASE}/api/generate",
                  {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": temperature}}, {})
        return d["response"]


class Mock(Provider):
    """
    Offline stand-in. Produces a faithful summary most of the time and a
    seeded error the rest, so the pipeline, the metrics and the correction
    loop can all be exercised and demonstrated without a network or a key.
    """

    name = "mock"

    def __init__(self, model: str | None, record: dict, error_rate: float = 0.6):
        self.record = record
        self.error_rate = error_rate
        self.rng = random.Random(SEED)
        self.model = model or "mock-v1"

    def generate(self, prompt: str, temperature: float) -> str:
        constrained = "Strict rules" in prompt or "Rewrite the summary" in prompt
        text = faithful_summary(self.record)
        rate = self.error_rate * (0.25 if constrained else 1.0)
        if self.rng.random() < rate:
            text = inject(text, self.record,
                          self.rng.choice(list(INJECTIONS)), self.rng)[0]
        return text


def make_provider(name: str, model: str | None, record: dict) -> Provider:
    return {"gemini": lambda: Gemini(model), "groq": lambda: Groq(model),
            "ollama": lambda: Ollama(model),
            "mock": lambda: Mock(model, record)}[name]()


# --------------------------------------------------------------------------
# Reference summary and controlled injections
# --------------------------------------------------------------------------

def faithful_summary(record: dict) -> str:
    """Built from the record, so correct by construction."""
    ms = record["models"]
    order = sorted(ms, key=lambda k: -ms[k]["mean_macro_f1"])
    best, second = ms[order[0]], ms[order[1]]
    worst = ms[order[-1]]
    ece_best = min(ms, key=lambda k: ms[k]["ece"])
    return (
        f"Across {record['n_outer_folds']} outer folds on "
        f"{record['n_segments']} segments from {record['n_measurements']} "
        f"measurements, {best['model_name']} obtained the highest mean "
        f"macro-F1 of {best['mean_macro_f1']} with a standard deviation of "
        f"{best['std_macro_f1']}. {second['model_name']} followed at "
        f"{second['mean_macro_f1']}, and {worst['model_name']} was lowest at "
        f"{worst['mean_macro_f1']}. All models exceeded the majority-class "
        f"baseline of {record['majority_baseline_macro_f1']}. "
        f"{ms[ece_best]['model_name']} recorded the lowest expected "
        f"calibration error at {ms[ece_best]['ece']}."
    )


INJECTIONS = {
    "altered_value": "a stored figure changed by one digit",
    "invented_value": "a number with no counterpart in the record",
    "wrong_superlative": "the wrong model named as best",
    "reversed_comparison": "a pairwise comparison stated backwards",
    "wrong_percentage": "an accuracy quoted as an incorrect percentage",
}


def inject(text: str, record: dict, kind: str, rng: random.Random) -> tuple[str, str]:
    ms = record["models"]
    order = sorted(ms, key=lambda k: -ms[k]["mean_macro_f1"])
    best, worst = ms[order[0]], ms[order[-1]]

    if kind == "altered_value":
        old = str(best["mean_macro_f1"])
        digits = list(old)
        for i in range(len(digits) - 1, 1, -1):
            if digits[i].isdigit():
                digits[i] = str((int(digits[i]) + 5) % 10)
                break
        return text.replace(old, "".join(digits), 1), old

    if kind == "invented_value":
        return (text + " Convergence was reached after 27 epochs.",
                "27 epochs")

    if kind == "wrong_superlative":
        return (text.replace(
            f"{best['model_name']} obtained the highest mean macro-F1",
            f"{worst['model_name']} obtained the highest mean macro-F1", 1),
            worst["model_name"])

    if kind == "reversed_comparison":
        return (text + f" {worst['model_name']} outperforms "
                       f"{best['model_name']} on macro-F1.", "reversed")

    if kind == "wrong_percentage":
        return (text + f" {best['model_name']} reached 97.5% accuracy.",
                "97.5%")

    raise ValueError(kind)


def injection_test(record: dict) -> dict:
    """How many seeded errors of each type does the verifier actually catch?"""
    rng = random.Random(SEED)
    base = faithful_summary(record)
    base_report = verify(base, record)
    out = {"reference_summary_clean": base_report.passed(),
           "reference_issues": issues_text(base_report), "by_type": {}}
    for kind in INJECTIONS:
        corrupted, _ = inject(base, record, kind, rng)
        rep = verify(corrupted, record)
        out["by_type"][kind] = {
            "description": INJECTIONS[kind],
            "detected": not rep.passed(),
            "counts": rep.counts(),
        }
    detected = sum(v["detected"] for v in out["by_type"].values())
    out["detection_rate"] = detected / len(INJECTIONS)
    return out


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------

@dataclass
class Run:
    arm: str
    index: int
    text: str
    n_numbers: int
    factual_accuracy: float
    unsupported_rate: float
    counts: dict
    passed: bool
    word_count: int = 0
    record_coverage: float = 0.0
    corrected_text: str = ""
    corrected_passed: bool | None = None
    corrected_accuracy: float | None = None
    seconds: float = 0.0
    prompt: str = ""
    error: str = ""


def record_coverage(text: str, record: dict) -> float:
    """
    Fraction of the record's numeric facts that appear verbatim in the text.

    A "summary" that reproduces every stored value is a transcription. It is
    trivially faithful and tells you nothing about whether the model can
    select and condense, so faithfulness alone is not a sufficient measure.
    """
    facts = set(flatten(record).values())
    if not facts:
        return 0.0
    found = {float(m.group(1).replace(",", ""))
             for m in NUMBER_RE.finditer(text)}
    hit = sum(any(abs(f - g) < 1e-9 or round(f, len(str(g).split(".")[-1])) == g
                  for g in found) for f in facts)
    return round(hit / len(facts), 4)


def run_arm(provider: Provider, arm: str, record: dict, repeats: int,
            temperature: float, delay: float = 0.0) -> list[Run]:
    rec_json = json.dumps(record, indent=2)
    template = DIRECT_PROMPT if arm == "direct" else CONSTRAINED_PROMPT
    runs = []
    for i in range(repeats):
        if delay and i:
            time.sleep(delay)
        t0 = time.perf_counter()
        try:
            text = provider.generate(template.format(record=rec_json),
                                     temperature).strip()
        except (urllib.error.URLError, KeyError, IndexError) as e:
            runs.append(Run(arm, i, "", 0, float("nan"), float("nan"), {},
                            False, error=f"{type(e).__name__}: {e}"))
            continue
        prompt = template.format(record=rec_json)
        rep = verify(text, record, prompt=prompt)
        run = Run(arm, i, text, rep.n_numbers, rep.factual_accuracy,
                  rep.unsupported_rate, rep.counts(), rep.passed(),
                  seconds=round(time.perf_counter() - t0, 2), prompt=prompt,
                  word_count=len(text.split()),
                  record_coverage=record_coverage(text, record))

        if arm == "generate_verify" and not rep.passed():
            try:
                fixed = provider.generate(
                    CORRECTION_PROMPT.format(issues=issues_text(rep),
                                             record=rec_json, previous=text),
                    temperature).strip()
                rep2 = verify(fixed, record, prompt=prompt)
                run.corrected_text = fixed
                run.corrected_passed = rep2.passed()
                run.corrected_accuracy = rep2.factual_accuracy
            except Exception as e:                       # noqa: BLE001
                run.error = f"correction failed: {e}"
        elif arm == "generate_verify":
            run.corrected_passed = True
            run.corrected_accuracy = rep.factual_accuracy
        runs.append(run)
        print(f"  {arm} run {i}: {'PASS' if run.passed else 'FAIL'} "
              f"| numbers {run.n_numbers} "
              f"| accuracy {run.factual_accuracy:.3f}"
              + (f" | after correction "
                 f"{'PASS' if run.corrected_passed else 'FAIL'}"
                 if run.corrected_passed is not None and not run.passed else ""))
    return runs


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    95% Wilson score interval for a pass rate.

    With five repeats a rate of 0.60 has an interval of roughly [0.23, 0.88],
    which cannot distinguish it from 0.30 or 0.85. Reporting a bare fraction
    from n=5 as though it were an estimate would not survive review, so the
    interval is printed alongside it.
    """
    import math
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def summarise(runs: list[Run]) -> dict:
    ok = [r for r in runs if not r.error]
    if not ok:
        return {"n": 0, "errors": len(runs)}
    acc = [r.factual_accuracy for r in ok if r.n_numbers]
    n_pass = sum(r.passed for r in ok)
    lo, hi = wilson(n_pass, len(ok))
    out = {"n": len(ok), "errors": len(runs) - len(ok),
           "n_pass": n_pass,
           "pass_rate": round(n_pass / len(ok), 4),
           "pass_rate_ci95": [round(lo, 4), round(hi, 4)],
           "mean_factual_accuracy": round(sum(acc) / len(acc), 4) if acc else None,
           "mean_unsupported_rate": round(
               sum(r.unsupported_rate for r in ok) / len(ok), 4),
           "mean_numbers_per_summary": round(
               sum(r.n_numbers for r in ok) / len(ok), 2),
           "mean_word_count": round(sum(r.word_count for r in ok) / len(ok), 1),
           "mean_record_coverage": round(
               sum(r.record_coverage for r in ok) / len(ok), 4)}
    corr = [r for r in ok if r.corrected_passed is not None]
    if corr:
        out["pass_rate_after_correction"] = round(
            sum(bool(r.corrected_passed) for r in corr) / len(corr), 4)
    return out


def _save(path: Path, results: dict, all_runs: dict, prior: dict) -> None:
    """Write the results file, attaching one correct and one incorrect
    example per arm as section 10.1 requires."""
    for arm, runs in all_runs.items():
        good = next((r for r in runs if r.passed and r.text), None)
        bad = next((r for r in runs if not r.passed and r.text), None)
        if good:
            results["arms"][arm]["example_correct"] = good.text
        if bad:
            results["arms"][arm]["example_incorrect"] = bad.text
    path.write_text(json.dumps(results, indent=2, default=str))


def inspect(path: Path, record: dict) -> None:
    """
    Print the reasoning behind every failure.

    A verifier that reports a high unsupported-claim rate is only useful if
    those reports are correct. Compound sentences that mention two models or
    two metrics are the obvious place for a false alarm, so every flagged
    claim is shown with the sentence it came from and the rule that fired.
    """
    if not path.exists():
        path = ARTIFACT_DIR / path.name
    data = json.loads(path.read_text())
    # Verify against the record stored WITH the run, not the current one.
    # Results files change as models are re-run, and re-checking old summaries
    # against new numbers would report faithful text as fabricated.
    stored = data.get("record")
    if stored is None:
        print("  note: this results file predates record storage, so it is "
              "being checked against the CURRENT record. Any mismatch may be "
              "an artefact of results having changed since the run.\n")
    else:
        record = stored
    print(f"provider {data['provider']} | model {data['model']} | "
          f"temperature {data['temperature']}\n")

    for arm, d in data["arms"].items():
        failed = [r for r in d["runs"] if not r["passed"] and r["text"]]
        print("=" * 78)
        print(f"ARM {arm}: {len(failed)} of {len(d['runs'])} runs flagged")
        print("=" * 78)
        for r in failed:
            rep = verify(r["text"], record, prompt=r.get("prompt"))
            print(f"\n--- run {r['index']} "
                  f"(numbers {rep.n_numbers}, accuracy "
                  f"{rep.factual_accuracy:.3f}) ---")
            echo = rep.counts()["PROMPT_ECHO"]
            if echo:
                print(f"  ({echo} number(s) echoed from the prompt, "
                      f"not counted)")
            for c in rep.numeric:
                if c.verdict in ("ALTERED", "UNSUPPORTED"):
                    print(f"  NUMBER [{c.verdict}] {c.text.strip()!r}"
                          + (f"  closest: {c.matched}" if c.matched else ""))
                    print(f"    in: {c.sentence.strip()}")
            for st in rep.statements:
                if st.verdict == "INCORRECT":
                    print(f"  CLAIM  [{st.kind}] {st.detail}")
                    print(f"    in: {st.sentence.strip()}")
            print(f"\n  full text:\n    "
                  + r["text"].replace("\n", "\n    "))
        if not failed:
            ex = d.get("example_correct")
            if ex:
                print("\n  no failures. one accepted summary:\n    "
                      + ex.replace("\n", "\n    "))
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock",
                    choices=["gemini", "groq", "ollama", "mock"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--inspect", metavar="RESULTS_JSON", default=None,
                    help="reload a results file and print every flagged claim "
                         "with the verifier's reason, so a failure can be "
                         "confirmed as a real error rather than a false alarm")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--delay", type=float, default=0.0,
                    help="seconds to wait between calls. Use about 5 on the "
                         "Gemini free tier (roughly 15 requests per minute).")
    ap.add_argument("--resume", action="store_true",
                    help="keep runs that already succeeded, so a quota "
                         "failure does not discard completed work")
    ap.add_argument("--arms", nargs="+",
                    default=["direct", "constrained", "generate_verify"])
    args = ap.parse_args()

    record = build_record()
    if not record["models"]:
        raise SystemExit("no results found; run the trainers first")

    if args.inspect:
        inspect(Path(args.inspect), record)
        return

    provider = make_provider(args.provider, args.model, record)
    if args.list_models:
        for m in provider.list_models():
            print(m)
        return

    print(f"provider {provider.name} | model {getattr(provider, 'model', '')} "
          f"| repeats {args.repeats} | temperature {args.temperature}\n")

    print("=== Arm 4: verifier detection rate on seeded errors ===")
    inj = injection_test(record)
    print(f"  reference summary is clean: {inj['reference_summary_clean']}")
    for kind, v in inj["by_type"].items():
        print(f"  {kind:22} {'DETECTED' if v['detected'] else 'MISSED':9} "
              f"({v['description']})")
    print(f"  detection rate {inj['detection_rate']:.2f}\n")

    results = {"generated": datetime.now(timezone.utc).isoformat(),
               "provider": provider.name,
               "model": getattr(provider, "model", ""),
               "temperature": args.temperature, "repeats": args.repeats,
               "record": record,
               "injection_test": inj, "arms": {}}

    # The model goes in the filename. Two Ollama runs with different models
    # both wrote results_llm_ollama.json and the second silently destroyed the
    # first.
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-",
                  getattr(provider, "model", "") or "default")
    out = ARTIFACT_DIR / f"results_llm_{provider.name}_{slug}.json"

    prior = {}
    if args.resume and out.exists():
        old = json.loads(out.read_text())
        for arm, d in old.get("arms", {}).items():
            keep = [r for r in d["runs"] if not r["error"] and r["text"]]
            if keep:
                prior[arm] = keep
                print(f"resuming: {len(keep)} usable {arm} runs already stored")
        print()

    all_runs = {}
    for arm in args.arms:
        have = prior.get(arm, [])
        need = max(0, args.repeats - len(have))
        print(f"=== Arm: {arm} ==="
              + (f" ({len(have)} kept, {need} to run)" if have else ""))
        runs = []
        if need:
            try:
                runs = run_arm(provider, arm, record, need, args.temperature,
                               args.delay)
            except SystemExit as e:
                print(f"  stopped: {e}")
                _save(out, results, all_runs, prior)
                raise
        merged = [Run(**{k: v for k, v in r.items()
                         if k in Run.__dataclass_fields__}) for r in have] + runs
        all_runs[arm] = merged
        results["arms"][arm] = {"summary": summarise(merged),
                                "runs": [asdict(r) for r in merged]}
        # Checkpoint after every arm. A quota failure three arms in should not
        # discard the two that completed.
        _save(out, results, all_runs, prior)
        print()

    print(f"{'arm':<18}{'pass':>8}{'95% CI':>16}{'accuracy':>10}"
          f"{'unsupp':>9}{'numbers':>9}{'words':>7}{'coverage':>10}"
          f"{'after fix':>11}")
    for arm, d in results["arms"].items():
        s = d["summary"]
        if not s.get("n"):
            print(f"{arm:<18}{'all failed':>8}")
            continue
        ci = s.get("pass_rate_ci95", [float("nan")] * 2)
        print(f"{arm:<18}{s['pass_rate']:>8.2f}"
              f"{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>16}"
              f"{(s['mean_factual_accuracy'] or 0):>10.3f}"
              f"{s['mean_unsupported_rate']:>9.3f}"
              f"{s['mean_numbers_per_summary']:>9.1f}"
              f"{s['mean_word_count']:>7.0f}"
              f"{s['mean_record_coverage']:>10.2f}"
              f"{s.get('pass_rate_after_correction', float('nan')):>11.2f}")

    _save(out, results, all_runs, prior)
    print(f"\nWrote {out.name}")


if __name__ == "__main__":
    main()
