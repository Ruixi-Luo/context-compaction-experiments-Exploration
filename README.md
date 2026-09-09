# Context Compaction Theory (Empirical Study) [Forked by Ruixi]

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## This fork: Kimi (Moonshot) client-side compaction arm

Based on Tirmazi et al., *Context Compaction Theory* (arXiv:2608.01326);
original code and claude/openai records unchanged (MIT).

Added `kimi_experiment.py`: a client-side LLM-as-compactor arm (GEN class)
against Moonshot's Kimi-K2.7-Code via SiliconFlow (n=6000; summary budget
matched at ~0.95 bits/item; same prompts and scoring as the original arms).

Results across two batches (6 runs total, seeds 42-44 each):
- 5/6 collapse to random-guess level (error ~0.5; all-NO strategy with
  explicit "cannot guarantee exact membership" disclaimers in the summary)
- 1/6 reached error 0.11, below the Bloom-filter frontier at matched
  bits/item (batch 1, seed 44; archived in results-kimi-batch1-20260909/)
Compaction quality under identical settings is high-variance — a
"strategy lottery": whether the condenser encodes discriminative patterns
or gives up is not stable across runs.

Reproduce: `$env:SILICONFLOW_API_KEY=...; python kimi_experiment.py run --seed 42`
(plus 43, 44), then `python kimi_experiment.py plot`.

#This is again from the original branch

To run the experiments, you will need a key to OpenAI and a key to Anthropic. 

```sh
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

You can run the experiments using the following script:

```
run_all.sh
```

Take a look at its commands for more information.

To generate figures using the data from the experiments, you can use

```sh
uv run experiment.py plot
```

To run tests, you can use 

```sh
uv run --with pytest pytest -q
```

Finally, to run a lint check, just use

```
uvx ruff check .
```
