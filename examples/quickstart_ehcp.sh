#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
  echo "Create a .env file from .env.example before running this script."
  exit 1
fi

: "${APRT_GENERATOR_MODEL:=openai/gpt-4o-mini}"

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install

# Generate prompt variants
aprt gen-variants \
  --template src/ai_ped_red_team/templates/examples/questionnaire/q_ehcp_gender.json \
  --hotness cold \
  --n 5 \
  --seed 0 \
  > variants.json

# Run and analyse
aprt run \
  --template src/ai_ped_red_team/templates/examples/questionnaire/q_ehcp_gender.json \
  --ehcp src/ai_ped_red_team/templates/examples/ehcp_pair \
  --model "${APRT_TESTER_MODEL:-openai/gpt-4o-mini}"

aprt analyze --results reports/latest/results.jsonl > summary.json
aprt report --summary summary.json
