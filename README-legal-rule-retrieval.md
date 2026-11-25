# Legal Rule Retrieval Harness

The current benchmark uses the downloaded Hugging Face document corpus and
`input/rule_recall.jsonl`. See [README-legal-rule-retrieval-final.md](README-legal-rule-retrieval-final.md)
for setup, commands, tools, and scoring.

`run-dsh` is an alias for the direct Python `run-python` command. Its native
OpenAI tool schemas are generated from `TOOL_SPECS` in
`legal_rule_retrieval/harness.py`; the optional MCP server exposes the same
document tools. The benchmark uses document IDs for retrieval and article
citations for submission.
