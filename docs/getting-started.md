# Getting Started

## Prerequisites
- Python 3.10 or newer
- Virtual environment tool (`python -m venv` or `uv`)
- Provider API keys (OpenAI, Anthropic, etc.)

## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,docs]"
pre-commit install
```

## Running the demo
1. Copy `.env.example` to `.env` and fill in provider keys.
2. Generate prompt variants: `aprt gen-variants --template src/ai_ped_red_team/templates/examples/questionnaire/q_ehcp_gender.json`.
3. Run the tester against EHCP pair: `aprt run ...`.
4. Analyse results: `aprt analyze --results reports/latest/results.jsonl`.
5. Render reports: `aprt report --summary reports/latest/summary.json`.

Troubleshooting tips and advanced usage will be added as the implementation matures.
