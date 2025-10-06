# TODO.md — ai-ped-red-team (PyPI / MIT)

> **Goal**: Ship a vendor-agnostic, template-driven red-teaming toolkit for pedagogy bias studies (EHCP example first), with state-chart validation hooks, reproducible runs, and automated analysis.
> **Outputs**: Python package (`ai-ped-red-team`), CLI (`aprt`), docs site, demo datasets, and CI/CD for PyPI.

## 2) Detailed checklists for implementation (for Codex)

### A) Bootstrap & metadata
- [x] Create repo from scaffold above.
- [x] Add `LICENSE` (MIT), `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`.
- [x] Write `pyproject.toml` (see template below): name, version, description, authors, license, classifiers, dependencies, optional extras (`dev`, `docs`).
- [x] Configure `pre-commit` with ruff+black and `pre-commit install`. citeturn1search7
- [x] Add GitHub Actions: `ci.yml` (lint/test), `release.yml` (build+publish to PyPI).
- [x] Add `.env.example` with needed keys and comments.

### B) Configuration & secrets
- [x] Implement `Settings` in `config.py` using `pydantic-settings` for: API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`), default model names for each role (generator/tester/analyst), paths, timeouts. citeturn0search2
- [x] Provide `load_settings()` factory reading env + `.env` (optional) with precedence.
- [x] Never log secrets; redact in JSONL/CSV exports.

### C) Schemas (Pydantic) in `models/schema.py`
- [x] `QuestionnaireTemplate` {id, hotness, placeholders[], template_text, metadata}.
- [x] `PromptVariant` {variant_id, variant_prompt, tokens_hint?}.
- [x] `RunConfig` {model, temperature, seed, n_variants, counterbalance}.
- [x] `RunResult` {variant_id, prompt, response, latency_ms, model_info, timestamps}.
- [x] `AnalysisRecord` {labels, counts, sentiment, directives_ratio, readability, effect_sizes}.

### D) Templates layer
- [x] Implement `templates/loader.py` to load JSON, validate via pydantic/jsonschema; helpful errors. citeturn1search3
- [x] Include `templates/examples/questionnaire/q_ehcp_gender.json` and EHCP pair texts.

### E) Model gateway (`models/gateway.py`)
- [x] Integrate **LiteLLM** with a single function: `llm_complete(model:str, prompt:str, temperature:float, seed:int|None) -> str & meta`.
- [x] Add timeouts, retries, backoff; record provider/model/version from response. citeturn0search8
- [x] Support local or alt providers via same interface (e.g., OpenRouter, Azure OpenAI).

### F) Variant generator (`generate/variants.py`)
- [x] Implement generator prompt template: take `QuestionnaireTemplate` + `hotness` (cold/hot).
- [x] Call generator model to produce N `PromptVariant` items (JSON parse with robust fallback).
- [x] Determinism controls: temperature per hotness (`cold` low, `hot` higher) + seed.

### G) Runner (`run/runner.py`)
- [x] Counterbalance Anthony↔Sarah and randomize order across repeats.
- [x] For each `PromptVariant`, substitute EHCP excerpt, call TEST model via gateway.
- [x] Stream results into JSONL and CSV; include latency, token counts (if available), and full prompts.
- [x] Respect rate limits; throttle and exponential backoff.

### H) Normalizer (`normalize/textnorm.py`)
- [x] PII redaction hooks; casing normalization; sentence splitting; lemmatization using spaCy. citeturn0search11
- [x] Map responses to advice categories via rule-set + optional ML classifier (later).

### I) Metrics (`analyze/metrics.py`)
- [x] Compute: word/lemma counts per category; directive modality ratio (`must/should/may`); sentiment/polarity (rule-based first pass); readability (textstat). citeturn0search20
- [x] Export tidy DataFrames (pandas). citeturn1search0

### J) Stats (`analyze/stats.py`)
- [x] Tests: Mann–Whitney U / t-test; χ² on categorical; BH correction; effect sizes (Cliff’s δ/Cohen’s d). Implement via `scipy/statsmodels/sklearn` as needed. citeturn0search5turn0search6
- [x] Return a compact `StatsSummary` with significance and thresholds.

### K) Reports (`analyze/report.py`)
- [x] Jinja2 templates to render Markdown and HTML summaries with tables and exemplar excerpts. citeturn1search12
- [x] Save under `./reports/{timestamp}/` with assets (CSV, JSONL, MD, HTML).

### L) State-chart validation (`models/statecheck.py`)
- [x] Provide `validate_chart(nodes, edges)` with checks: no orphans, single entry, at least one terminal, no illegal cycles (unless allowed), all transitions resolvable; build with **NetworkX**. citeturn1search10
- [x] Expose CLI: `aprt validate-chart --file scxml.json`.

### M) CLI (`src/ai_ped_red_team/cli.py` with Typer)
- [x] Commands:
  - `aprt gen-variants --template q_ehcp_gender.json --hotness cold --n 10`
  - `aprt run --template q_ehcp_gender.json --ehcp ./templates/examples/ehcp_pair/ --model gpt-4o-mini`
  - `aprt analyze --results ./reports/run_YYYYMMDD/results.jsonl`
  - `aprt report --summary ./reports/run_YYYYMMDD/summary.json`
  - `aprt validate-chart --file ./chart.json`
- [x] Add `--seed`, `--temperature`, `--counterbalance/--no-counterbalance`, `--csv` flags.
- [x] Rich output and progress bars. (Optional)
- [x] Record token usage totals and export JSON/CSV reports.

### N) Tests
- [x] Unit tests for gateway, loader, normalizer, metrics, stats.
- [x] Property-based tests for text normalization and metric invariants (Hypothesis). citeturn2search1
- [x] Golden-file tests for reports.
- [x] CI: run on 3.10–3.12; upload artifacts (report previews).

### O) Docs
- [x] MkDocs site: `docs/index.md`, `docs/getting-started.md`, `mkdocs.yml` (Material theme). citeturn2search2turn2search3
- [x] How-to: configure providers, run demo, interpret stats, export anonymized results.
- [ ] Add examples/gists; publish via GitHub Pages.

### P) Release (PyPI)
- [x] Ensure `pyproject.toml` has all metadata/classifiers and `console_scripts`.
- [x] Build with Hatchling; test install in a fresh venv; then publish (token via CI). citeturn2search4
- [ ] Tag release; update `CHANGELOG.md` (Keep a Changelog).

### Q) Ethics, privacy, compliance
- [x] Provide `ETHICS.md`: consent model for human participants; PII handling; “hot” prompt safeguards.
- [x] Default to “cold” in examples; require `--ack-hot` flag to generate “hot” variants.
- [x] Redact before sharing datasets; store secrets in env only.

---

## 9) Release checklist (PyPI, MIT)
- [x] Update version; changelog entry.
- [x] Build wheels/sdist with hatchling; verify install in fresh venv. citeturn2search4
- [ ] `twine upload` or GitHub Action token-based publish.
- [ ] Create GitHub release and docs site build.

---

## 10) Open questions / backlog
- [ ] Add sentiment/tonality via simple lexicons now; consider ML later.
- [ ] Optional Polars backend and/or Arrow I/O for large runs. citeturn1search9
- [ ] Add UI (Streamlit/Tk) wrapper if desired.
- [ ] Expand templates beyond EHCP; share a public gallery.
- [ ] Provide anonymization pipeline for community datasets.

---

### Research references (selected)
- LiteLLM: unified multi-provider LLM access. citeturn0search8turn0search16
- Typer CLI: type-hint based CLI. citeturn0search1turn0search9
- Pydantic Settings: env/secrets loading. citeturn0search2turn0search10
- HTTPX: async/sync HTTP client. citeturn0search7turn0search15
- spaCy: industrial NLP. citeturn0search11turn0search19
- textstat: readability metrics. citeturn0search4turn0search20
- pandas / polars: dataframes. citeturn1search0turn1search9
- statsmodels / scikit-learn: stats & ML. citeturn0search5turn0search6
- jsonschema: JSON Schema in Python. citeturn1search3
- NetworkX: graph validation. citeturn1search10
- Jinja2 templates. citeturn1search12
- Ruff/Black/Pre-commit. citeturn1search5turn1search6turn1search7
- pytest/Hypothesis. citeturn2search0turn2search1
- MkDocs + Material. citeturn2search2turn2search3
- Hatchling build backend. citeturn2search4
