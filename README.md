# Spellguard open models

Open-weights guard models from [Spellguard](https://spellguard.ai) for AI agents and chat applications. They flag harmful prompts, harmful model replies, and prompt-injection and jailbreak attempts.

[Website](https://spellguard.ai) · [Documentation](https://docs.spellguard.ai) · [Spellguard-1.0-8B on Hugging Face](https://huggingface.co/Spellguard/Spellguard-1.0-8B) · [GGUF](https://huggingface.co/Spellguard/Spellguard-1.0-8B-GGUF)

The weights are on Hugging Face. This repository holds usage examples, the results in machine-readable form, and the scripts that reproduce the published open-weights scores.

## Models

| Version | Weights | Base model | Results |
|---|---|---|---|
| **1.0** | [Spellguard-1.0-8B](https://huggingface.co/Spellguard/Spellguard-1.0-8B) (BF16) · [GGUF](https://huggingface.co/Spellguard/Spellguard-1.0-8B-GGUF) (Q4_K_M) | [Qwen3Guard-Gen-8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B) | [`results/spellguard-1.0-8b.json`](results/spellguard-1.0-8b.json) |

Names follow `Spellguard-<major.minor>-<size>`. Each version has its own Hugging Face repository, and published weights are never replaced.

## Results

Results on four public benchmarks, Spellguard-1.0-8B against its base model. Both were scored with [`eval/run.sh`](eval/run.sh) as Q4_K_M GGUFs in llama.cpp, flagging an input when P(Unsafe) + P(Controversial) ≥ 0.5:

| Benchmark | Metric | Qwen3Guard-Gen-8B | Spellguard-1.0-8B |
|---|---|---:|---:|
| Artificial Analysis guardrail | mean macro F1, 3 sets | 86.94 | **91.01** |
| Horizon content safety | mean unsafe-class F1, 7 sets | 78.64 | **80.55** |
| Horizon prompt injection | mean of 10 sets | 55.44 | **75.64** |
| PHTest | harmless accuracy | 80.69 | **94.56** |

These are our own runs, not leaderboard submissions. [`eval/README.md`](eval/README.md) lists where we depart from each leaderboard's setup. Per-set results are in [`results/spellguard-1.0-8b.json`](results/spellguard-1.0-8b.json).

### In the Spellguard platform

Spellguard-1.0-8B is one of the base products that protect agents in [Spellguard](https://spellguard.ai). Used within the platform, together with Spellguard's other security systems, it scores higher on prompt injection and content safety: 82.29 on Horizon prompt injection and 81.99 on Horizon content safety, against 75.64 and 80.55 for the model alone. These figures were calculated from per-item scores with the platform's guard (`spellguard-1.0` on Spellguard's Guard API) as configured for this release, and [`eval/`](eval) does not reproduce them. To learn more and try it yourself, go to [spellguard.ai](https://spellguard.ai).

## Use it

- [`examples/transformers_example.py`](examples/transformers_example.py): the text answer and the score with Hugging Face Transformers.
- [`examples/llama_server_example.py`](examples/llama_server_example.py): the same through llama.cpp's `llama-server`.

The model keeps Qwen3Guard-Gen-8B's interface: the same chat template, the same `Safety:` and `Categories:` answer and the same categories. Two details matter:

- **Score from the label token, not the text.** The published open-weights numbers read the probabilities of the first token after `Safety:`, renormalised over the three labels, and flag at P(Unsafe) + P(Controversial) ≥ 0.5. Both examples show how.
- **Judging a reply with `llama-server`:** `/v1/chat/completions` treats a trailing assistant message as the start of its own answer. Render the chat template yourself and use `/completion`, as the llama.cpp example does.

## Reproduce the benchmarks

[`eval/`](eval) rebuilds every test set from its public source, scores each item with the model as published on Hugging Face, and computes the metrics for both models. Our score for every item is in [`results/scores/`](results/scores), so a rerun can be compared item by item. [`eval/README.md`](eval/README.md) has the steps. If you get different numbers, please open an issue with your `metrics-*.json`.

The training data and training code are not published.

## License

Code in this repository is Apache 2.0 ([LICENSE](LICENSE)). Each model's licence and attribution notice are on its Hugging Face page; Spellguard-1.0-8B is Apache 2.0. Benchmark data stays under its own licences and is downloaded from its source, never redistributed here.

## Citation

```bibtex
@misc{spellguard2026spellguard10,
  title  = {Spellguard-1.0-8B},
  author = {Spellguard},
  year   = {2026},
  url    = {https://huggingface.co/Spellguard/Spellguard-1.0-8B}
}
```

## About Spellguard

[Spellguard](https://spellguard.ai) is a control plane that secures how AI agents communicate and act. It verifies each agent's identity, checks every action against your policies before it happens, and keeps one audit trail of every interaction. [Documentation](https://docs.spellguard.ai) · [Hugging Face](https://huggingface.co/Spellguard)
