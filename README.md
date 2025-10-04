# ai-ped-red-team

Template-driven pedagogy red-team toolkit for LLM bias studies (EHCP demo).

## Features
- Vendor-agnostic model access (LiteLLM).
- Controlled prompt variation (hot/cold).
- Counterbalanced runs across matched EHCPs.
- Normalization + metrics + stats + templated reports.
- CLI-first, reproducible; MIT licensed.

## Quickstart
```bash
# 1) Create venv and install (dev)
uv venv && source .venv/bin/activate            # or python -m venv .venv
pip install -e ".[dev,docs]"
pre-commit install

# 2) Configure env (copy .env.example → .env; set provider keys)
export OPENAI_API_KEY=...  # etc.

# 3) Generate variants and run a demo
aprt gen-variants --template src/ai_ped_red_team/templates/examples/questionnaire/q_ehcp_gender.json --hotness cold --n 5 > variants.json
aprt run --template src/ai_ped_red_team/templates/examples/questionnaire/q_ehcp_gender.json --ehcp src/ai_ped_red_team/templates/examples/ehcp_pair --model openai/gpt-4o-mini

# 4) Analyze & report
aprt analyze --results ./reports/latest/results.jsonl > summary.json
aprt report --summary summary.json
```

## Ethics
- Default examples are “cold”; pass `--ack-hot` to enable “hot” variants.
- Redact PII; do not share raw EHCP data publicly.

## License
MIT © SoftOboros

