# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "huggingface_hub>=0.30"]
# ///
"""Download the public benchmarks and normalise them, at the exact data revisions the published results used.

    uv run fetch.py <benchmark|all> <out-dir>

Writes <out-dir>/<benchmark>/<set>.jsonl, one item per line:
    {"id": str, "text": str, "label": 0|1, "pair": str?, "lang": str?, "source": str?, "meta": {...}?}
`label` is 1 for an attack or harmful item and 0 for a benign one. `pair` is a model's reply to `text`; when
present, the guard judges that reply. Nothing here needs an account: every source is an ungated Hugging Face
dataset or a public GitHub file. Sampling follows each benchmark's own published build scripts, with the same
seeds. Nothing is redistributed: every item is downloaded from its source.
"""

import gzip
import json
import os
import sys
import urllib.request

import pandas as pd
from huggingface_hub import hf_hub_download

# The data revisions behind the published results (fetched 2026-10-06). Change one and the items change.
REVISIONS = {
    "hf:Horizon-Labs/prompt-injection-eval-suite": "27c7bcd6f46e842f4c4d8b114b30eb6b1d892965",
    "hf:ToxicityPrompts/PolyGuardPrompts": "c5b466a95b64ff121db4398246b6abb7672696ec",
    "hf:PKU-Alignment/BeaverTails": "8401fe609d288129cc684a9b3be6a93e41cfe678",
    "hf:lmsys/toxic-chat": "29df8e4dba60e1f4af4b4075c0705c5b313548a8",
    "hf:mmathys/openai-moderation-api-evaluation": "84e5cf3bcd6acb3dfc70b6760451645872218a3e",
    "hf:Paul/XSTest": "f600c994b256f12867dfa5b3eb3d545a3e62f8b5",
    "hf:Bertievidgen/SimpleSafetyTests": "98223c5d8c4059c8f4d8fe2fec8720ee8a20d3c5",
    "hf:CohereLabs/aya_redteaming": "5a16fee03c19d681b4f3527b6a750e1ed4d434c7",
    "hf:textdetox/multilingual_toxicity_dataset": "01907546324b0330d2d8b7669648cc18823323e5",
    "hf:JailbreakBench/JBB-Behaviors": "886acc352a31533ffbcf4ef22c744658688086fc",
    "github:JailbreakBench/artifacts": "909e68c01d94222b8ad2e397a017e2e12e2adb73",
    "hf:furonghuang-lab/PHTest": "edb80210812ecc7dab2219aa2e70921710c01888",
}
MANIFEST: dict = {"sources": []}


def hf(repo: str, filename: str) -> str:
    rev = REVISIONS[f"hf:{repo}"]
    MANIFEST["sources"].append({"repo": f"hf:{repo}", "file": filename, "revision": rev})
    return hf_hub_download(repo, filename, repo_type="dataset", revision=rev)


def github(repo: str, path: str) -> str:
    sha = REVISIONS[f"github:{repo}"]
    MANIFEST["sources"].append({"repo": f"github:{repo}", "file": path, "revision": sha})
    return urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}").read().decode()


def save(out: str, name: str, rows: list[dict]) -> None:
    with open(os.path.join(out, f"{name}.jsonl"), "w") as f:
        for i, row in enumerate(rows):
            row.setdefault("id", f"{name}-{i}")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pos = sum(r["label"] for r in rows)
    print(f"{name}: {len(rows)} items, {pos} positive", flush=True)


def sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=0)


# ── Horizon Labs prompt-injection eval suite ─────────────────────────────────
# https://huggingface.co/datasets/Horizon-Labs/prompt-injection-eval-suite
HORIZON_PI_CONFIGS = [
    "notinject", "xstest", "orbench_hard", "deepset_test", "jackhhao_test", "simsonsun_jailbreaks",
    "boundary_pairs_test", "agentic5k_test", "neuralchemy_test", "slabs_test", "bipia", "piarena",
    "llmail_phase2", "synthetic_docs_test", "synthetic_direct_test", "synthetic_v2_test",
]


def horizon_pi(out: str) -> None:
    for config in HORIZON_PI_CONFIGS:
        df = pd.read_parquet(hf("Horizon-Labs/prompt-injection-eval-suite", f"data/{config}.parquet"))
        save(out, config, [
            {"text": r.text, "label": int(r.label), "lang": r.lang, "source": r.source, "meta": {"kind": r.kind}}
            for r in df.itertuples()
        ])


# ── Horizon Labs content-safety suite ────────────────────────────────────────
# Rebuilt exactly as code/safety/build_safety_evals.py in Horizon-Labs/content-safety-guard-base does it
# (same sources, same sampling, same seeds). "beavertails_unseen" needs Horizon's private training split,
# so we score the full BeaverTails test set instead.
PGL = dict(English="en", Hindi="hi", French="fr", Italian="it", German="de", Portuguese="pt", Thai="th",
           Spanish="es", Czech="cs", Swedish="sv", Chinese="zh", Arabic="ar", Dutch="nl", Korean="ko",
           Polish="pl", Russian="ru", Japanese="ja")


def content_safety(out: str) -> None:
    import random
    rng = random.Random(0)
    p = pd.read_parquet(hf("ToxicityPrompts/PolyGuardPrompts", "data/test-00000-of-00001.parquet"))
    rows = []
    for lang, g in p[p.prompt_harm_label.isin(["harmful", "unharmful"])].groupby("language"):
        rows += [{"text": r.prompt, "label": int(r.prompt_harm_label == "harmful"), "lang": PGL[lang]}
                 for r in sample(g, 400).itertuples()]
    save(out, "polyguard_prompt", rows)
    rows = []
    q = p[p.response_harm_label.isin(["harmful", "unharmful"]) & p.response.fillna("").str.strip().astype(bool)]
    for lang, g in q.groupby("language"):
        rows += [{"text": r.prompt, "pair": r.response, "label": int(r.response_harm_label == "harmful"),
                  "lang": PGL[lang]} for r in sample(g, 300).itertuples()]
    save(out, "polyguard_response", rows)

    b = [json.loads(x) for x in gzip.open(hf("PKU-Alignment/BeaverTails", "round0/30k/test.jsonl.gz"), "rt")]
    save(out, "beavertails_response",
         [{"text": r["prompt"], "pair": r["response"], "label": int(not r["is_safe"]), "lang": "en"} for r in b])

    t = pd.read_csv(hf("lmsys/toxic-chat", "data/0124/toxic-chat_annotation_test.csv"))
    save(out, "toxicchat", [{"text": r.user_input, "label": int(r.toxicity), "lang": "en"} for r in t.itertuples()])

    o = [json.loads(x) for x in gzip.open(hf("mmathys/openai-moderation-api-evaluation", "samples-1680.jsonl.gz"), "rt")]
    save(out, "openai_mod", [
        {"text": r["prompt"], "label": int(any(r.get(k) == 1 for k in ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"])),
         "lang": "en"} for r in o])

    x = pd.read_csv(hf("Paul/XSTest", "xstest_prompts.csv"))
    save(out, "xstest", [{"text": r.prompt, "label": int(r.label == "unsafe"), "lang": "en",
                          "meta": {"type": r.type}} for r in x.itertuples()])

    s = pd.read_csv(hf("Bertievidgen/SimpleSafetyTests", "sst_test_cases.csv"))
    save(out, "simplesafety", [{"text": r.prompt, "label": 1, "lang": "en"} for r in s.itertuples()])

    rows = []
    for code, lang in [("arb", "ar"), ("eng", "en"), ("fra", "fr"), ("hin", "hi"), ("rus", "ru"), ("spa", "es"),
                       ("srp", "sr"), ("tgl", "tl")]:
        a = [json.loads(z) for z in open(hf("CohereLabs/aya_redteaming", f"aya_{code}.jsonl"))]
        rng.shuffle(a)
        rows += [{"text": r["prompt"], "label": 1, "lang": lang} for r in a[:300]]
    save(out, "aya_redteaming", rows)

    rows = []
    for lang in ["am", "ar", "de", "en", "es", "fr", "he", "hi", "it", "ja", "ru", "tt", "uk", "zh"]:
        d = pd.read_parquet(hf("textdetox/multilingual_toxicity_dataset", f"data/{lang}-00000-of-00001.parquet"))
        d = pd.concat([sample(d[d.toxic == 1], 200), sample(d[d.toxic == 0], 200)])
        rows += [{"text": r.text, "label": int(r.toxic), "lang": lang} for r in d.itertuples()]
    save(out, "textdetox", rows)


# ── JailbreakBench ───────────────────────────────────────────────────────────
# Behaviours: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors (100 harmful, 100 benign).
# Artifacts: https://github.com/JailbreakBench/artifacts, every attack prompt and the target model's reply.
JBB_ARTIFACTS = [
    ("PAIR", "black_box"), ("prompt_with_random_search", "black_box"), ("GCG", "transfer"), ("GCG", "white_box"),
    ("JBC", "manual"), ("DSN", "white_box"),
]


def jailbreakbench(out: str) -> None:
    harmful = pd.read_csv(hf("JailbreakBench/JBB-Behaviors", "data/harmful-behaviors.csv"))
    benign = pd.read_csv(hf("JailbreakBench/JBB-Behaviors", "data/benign-behaviors.csv"))
    save(out, "behaviors", [
        *[{"id": f"harmful-{r.Index}", "text": r.Goal, "label": 1, "meta": {"category": r.Category}}
          for r in harmful.itertuples()],
        *[{"id": f"benign-{r.Index}", "text": r.Goal, "label": 0, "meta": {"category": r.Category}}
          for r in benign.itertuples()],
    ])
    tree = json.load(urllib.request.urlopen(
        f"https://api.github.com/repos/JailbreakBench/artifacts/git/trees/{REVISIONS['github:JailbreakBench/artifacts']}?recursive=1"))["tree"]
    for method, kind in JBB_ARTIFACTS:
        prefix = f"attack-artifacts/{method}/{kind}/"
        for path in sorted(t["path"] for t in tree if t["path"].startswith(prefix) and t["path"].endswith(".json")):
            model = path[len(prefix):-5]
            art = json.loads(github("JailbreakBench/artifacts", path))
            rows = [
                {"id": f"{method}-{model}-{j['index']}", "text": j["prompt"] or "", "pair": j["response"] or "",
                 "label": int(str(j["jailbroken"]) == "True"),
                 "meta": {"behavior": j["behavior"], "category": j["category"], "submitted": bool(j["prompt"])}}
                for j in art["jailbreaks"]
            ]
            save(out, f"artifact__{method}__{kind}__{model}", rows)


# ── PHTest (pseudo-harmful prompts, false refusals) ──────────────────────────
def phtest(out: str) -> None:
    df = pd.read_csv(hf("furonghuang-lab/PHTest", "PHTest.csv"))
    save(out, "harmless", [{"id": f"ph-{r.ID}", "text": r.Request, "label": 0} for r in df.itertuples()
                           if r.Harmfulness == "harmless"])
    save(out, "controversial", [{"id": f"ph-{r.ID}", "text": r.Request, "label": 0} for r in df.itertuples()
                                if r.Harmfulness == "controversial"])


# ── Artificial Analysis guardrail benchmark, rerun ───────────────────────────
# https://artificialanalysis.ai/articles/guardrail-safety-benchmark (11 Jun 2026): 1,699 WildGuardTest prompt-harm
# items, the 5,083 ToxicChat 0124 test prompts and XSTest's 450. WildGuardTest itself is a gated download, but the
# English split of PolyGuardPrompts (CC-BY-4.0, ungated) is the same 1,699 prompts with the same labels.
def aa_guardrail(out: str) -> None:
    p = pd.read_parquet(hf("ToxicityPrompts/PolyGuardPrompts", "data/test-00000-of-00001.parquet"))
    e = p[(p.language == "English") & p.prompt_harm_label.isin(["harmful", "unharmful"])]
    save(out, "wildguardtest_prompt_harm", [
        {"id": f"wgt-{r.id}", "text": r.prompt, "label": int(r.prompt_harm_label == "harmful"), "lang": "en",
         "meta": {"adversarial": bool(r.adversarial)}} for r in e.itertuples()])
    t = pd.read_csv(hf("lmsys/toxic-chat", "data/0124/toxic-chat_annotation_test.csv"))
    save(out, "toxicchat", [{"id": f"tc-{r.conv_id}", "text": r.user_input, "label": int(r.toxicity), "lang": "en"}
                            for r in t.itertuples()])
    x = pd.read_csv(hf("Paul/XSTest", "xstest_prompts.csv"))
    save(out, "xstest", [{"id": f"xs-{r.id}", "text": r.prompt, "label": int(r.label == "unsafe"), "lang": "en",
                          "meta": {"type": r.type}} for r in x.itertuples()])


BENCHMARKS = {
    "aa-guardrail": aa_guardrail,
    "content-safety": content_safety,
    "horizon-pi": horizon_pi,
    "jailbreakbench": jailbreakbench,
    "phtest": phtest,
}


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in [*BENCHMARKS, "all"]:
        sys.exit(f"usage: fetch.py <{'|'.join(BENCHMARKS)}|all> <out-dir>")
    names = list(BENCHMARKS) if sys.argv[1] == "all" else [sys.argv[1]]
    for name in names:
        out = os.path.join(sys.argv[2], name)
        os.makedirs(out, exist_ok=True)
        MANIFEST["sources"] = []
        BENCHMARKS[name](out)
        with open(os.path.join(out, "manifest.json"), "w") as f:
            json.dump({"benchmark": name, **MANIFEST}, f, indent=1)


if __name__ == "__main__":
    main()
