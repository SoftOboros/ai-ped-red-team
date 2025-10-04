# Agents

Located in `src/ai_ped_red_team/agents/`.

- **generator.py** — wraps `generate_variants()`; ensures JSON-only outputs; logs provenance.
- **tester.py** — wraps `run_variants()`; enforces counterbalancing; rate-limit handling.
- **analyst.py** — composes `compute_metrics()` + `summarize_stats()` + `render_report()`.

All agents consume `Settings` and never print secrets; they return paths/artefacts for the UI.
