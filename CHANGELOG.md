# Changelog

All notable changes will be documented in this file following [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]
- Added CLI commands `aprt vendors` and `aprt models` to inspect configured providers and discover available models (OpenAI/Gemini).
- Added configurable axes runner with persona/support/history sweeps and templated EHCP inputs.
- Introduced VADER + Detoxify sentiment/toxicity metrics, including history deltas for injected context.
- Documented templated pipeline, history injection workflows, and token/axes usage in README and Getting Started guide.
- Wizard now offers to load an axes TOML and spins out a report per combination, injecting pre-scripted histories automatically.
- LiteLLM gateway drops unsupported parameters (e.g., `seed` for Gemini-family models) and surfaces the last provider error when retries are exhausted.
- Pinned NumPy to `<2` to avoid ABI mismatches with PyTorch/Detoxify wheels.
- Added monitoring-channel install matrix and sentiment lexicon configuration guidance to the docs.

## [0.1.0] - 2025-10-04
- Initial project scaffold.
- Added interactive CLI wizard with Rich progress and token usage reporting.
