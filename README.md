# ai-ped-red-team

Template-driven pedagogy red-team toolkit for LLM bias studies (EHCP demo).

## Features
- Vendor-agnostic model access (LiteLLM) with built-in support for OpenAI, Google Gemini, Anthropic, etc.
- Controlled prompt variation (hot/cold) with fallback safety prompts.
- Counterbalanced runs across matched EHCPs and template-driven persona swapping.
- Configurable axes (gender/support/history) to exhaust combinations and inject biased dialogue history.
- Normalization + metrics (VADER sentiment, Detoxify toxicity, lexical heuristics) + stats + templated reports.
- CLI-first, reproducible; MIT licensed.

## Quickstart
```bash
# 1) Install (dev mode for local work)
uv venv && source .venv/bin/activate            # or python -m venv .venv
pip install -e ".[dev,docs]"
pre-commit install

# 2) Configure env (copy .env.example → .env; set provider keys)
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...   # for Gemini (google/* models)

> Gemini setup tip: make sure the Google project tied to `GOOGLE_API_KEY` has the *Generative Language API* enabled and billing active. Without those, Google will return HTTP 403.

# 3) Guided experience (recommended)
aprt wizard
# If `examples/ehcp_variables.toml` is present, the wizard offers to run every
# persona/support/history combination and tags each report directory with the
# chosen axis labels.

# Manual steps (if you prefer granular control)
aprt gen-variants --template src/ai_ped_red_team/templates/examples/questionnaire/q_ehcp_gender.json --hotness cold --n 5 > variants.json
aprt run --template src/ai_ped_red_team/templates/examples/questionnaire/q_ehcp_gender.json \
         --ehcp examples/EHCP-templates \
         --axes-config examples/ehcp_variables.toml \
         --model openai/gpt-5-nano
aprt analyze --results ./reports/latest/results.jsonl > summary.json
aprt report --summary summary.json

# Inspect configured providers and available models
aprt vendors
aprt models openai
aprt models google

# Need to inspect raw LiteLLM traffic?
aprt --debug run --template ...
```

Generated runs now include token accounting artefacts (`token_usage.json` and `token_usage.csv`) alongside results, making usage tracking easy for billing reviews.

Axes-enabled runs
: The `examples/ehcp_variables.toml` file demonstrates two axes (`gender`, `support`) plus a `history` axis used to prep the conversation with templated prior turns. Each combination of axis options produces its own report directory (name suffix such as `20251005-174644-gender-anthony_davis-history-prior_misalignment`). Use this mechanism to compare baseline behaviour against controlled bias injections.

The wizard first asks for a provider and model. Press Enter to keep the defaults (`openai` / `gpt-5-nano`), which deliver long-context, chat-style responses out of the box.

## Bias metrics

Every result row now carries both emotional tone (VADER sentiment) and social toxicity scores (Detoxify). For each response you receive:

- `toxicity`, `severe_toxicity`, `obscene`, `threat`, `insult`, `identity_attack` — Detoxify probabilities for the answer under test.
- `history_*` counterparts capturing the injected conversation context when a history axis is active.
- `*_delta` fields quantifying how much more or less toxic the final answer became relative to the history seed.

The metrics surface in `results.csv`, `metrics.csv`, and summary outputs, making it easy to correlate sentiment drift with professionalism leakage.

Need different lexicons? Point the `APRT_SENTIMENT_LEXICON` environment variable at your own JSON file, or drop `sentiment_words.json` into the repository root or `~/.ai_ped_red_team/`. The analyzer seeds the file with defaults on first use and preserves any edits you make afterwards.

### Monitoring channels

| Detector | Package(s) / Install Command | Local Use |
| --- | --- | --- |
| Sentiment + Valence Shift | `pip install nltk textblob transformers torch` (VADER, TextBlob, optional RoBERTa sentiment) | Run directly on text samples; track baseline mean ± σ to quantify valence drift. |
| Toxicity / Harassment | `pip install detoxify torch` | Load Unitary's Detoxify (`Detoxify('original')`) for toxicity, severe toxicity, threat, insult, and identity-attack probabilities. |
| Perplexity / Entropy Drift | `pip install transformers torch` | Score GPT-2 perplexity per sentence to monitor language-model confidence changes. |
| Emotion Classifier | `pip install transformers torch` then use `bhadresh-savani/distilbert-base-uncased-emotion` or `joeddav/distilbert-base-uncased-go-emotions-student` | Hugging Face pipeline; no external API required. |
| Prompt Coercion Heuristics | `pip install spacy` and `python -m spacy download en_core_web_sm` | Rule-based detection of modal and imperative spikes that suggest coercive phrasing. |
| Linguistic Style Drift | `pip install textdistance sentence-transformers` | Compare embeddings or token overlap between turns to catch stylistic shifts. |
| Embeddings Similarity Monitor | `pip install sentence-transformers` | Measure semantic similarity between reference phrases and model output to catch topic drift. |

## Axes & history injection

The new axes system lets you factor experiments along multiple dimensions:

- **Persona/Gender axis** — swaps in templated names, short forms, and a complete pronoun set. The EHCP templates in `examples/EHCP-templates/` contain placeholders for each pronoun to guarantee consistent substitution.
- **Support axis** — rewrites the `Primary/Secondary/Additional Needs` blocks so that the same prompt variants can be evaluated against distinct support requirements.
- **History axis** — feeds a sequence of prior turns (`HISTORY_PROMPTS`) into the model *before* the prompt under test. This simulates teachers reusing a chat thread without clearing context and is a high-leverage way to study bias drift.

All axes are described in a single TOML file so runs remain reproducible. The runner expands every combination, materialises a dedicated report directory per combination, records the axis labels in `results.jsonl`, and includes the injected history text in each call made to the LLM.

> Gemini endpoints ignore LiteLLM's `seed` parameter. The gateway now drops it automatically for all `gemini/*`, `gemma-*`, `learnlm-*`, `imagen-*`, and Google embedding models so you can keep deterministic-looking defaults without triggering API validation errors.

## Ethics
- Default examples are “cold”; pass `--ack-hot` to enable “hot” variants.
- Redact PII; do not share raw EHCP data publicly.

## License
MIT © SoftOboros
