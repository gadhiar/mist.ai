"""MIST model backend A/B evaluation harness.

Model-agnostic Python harness that drives llama-server via the OpenAI-compatible
chat completions API and scores candidate models against MIST's real workload:
ontology-constrained extraction, tool selection, personality adherence, RAG
integration, multi-turn coherence, and speed metrics.

Entry point: `python -m scripts.eval_harness.run`.
"""
