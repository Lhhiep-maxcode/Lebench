"""Expose the prepared legal corpus as tools to a DeepSeek Harness MCP client."""

import argparse
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .harness import ToolSession
from .vectors import VectorRuleDatabase, make_embedder


def create_server(database, workspace_root, task_id, submission_path):
    tools = ToolSession(database, workspace_root, task_id)
    submission_path = Path(submission_path)
    server = FastMCP(
        'legal_rules',
        instructions=('Find Vietnamese legal provisions for the supplied facts and issues. '
                      'Search, inspect full rules, check validity when relevant, then call '
                      'submit_rules once. Use only exact rule IDs returned by the tools. '
                      'Stop after a successful submission.'),
    )

    @server.tool()
    def search_legal_rules(query: str, top_k: int = 5) -> dict:
        """Embedding search for relevant provisions; returns exact IDs, scores and previews."""
        return tools.dispatch('search_legal_rules', {'query': query, 'top_k': top_k})

    @server.tool()
    def get_legal_rule(rule_id: str) -> dict:
        """Retrieve the full text and source metadata of one exact provision ID."""
        return tools.dispatch('get_legal_rule', {'rule_id': rule_id})

    @server.tool()
    def check_validity(rule_id: str, start: str, end: str) -> dict:
        """Check document-level effective dates over an ISO date interval; unknown is possible."""
        return tools.dispatch('check_validity', {'rule_id': rule_id, 'start': start, 'end': end})

    @server.tool()
    def submit_rules(rule_ids: list[str]) -> dict:
        """Submit exact IDs returned by search/get. This is the final benchmark action."""
        if tools.submitted is not None:
            raise ValueError('Rules have already been submitted')
        result = tools.dispatch('submit_rules', {'rule_ids': rule_ids})
        temporary = submission_path.with_suffix(submission_path.suffix + '.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
        temporary.replace(submission_path)
        return result

    @server.tool()
    def create_file(content: str, path: str) -> dict:
        """Create a new UTF-8 file under this task's private workspace (maximum 1 MB)."""
        return tools.dispatch('create_file', {'content': content, 'path': path})

    @server.tool()
    def read_file(path: str) -> dict:
        """Read a UTF-8 file in this task's private workspace (maximum 1 MB)."""
        return tools.dispatch('read_file', {'path': path})

    @server.tool()
    def add(a: str | int | float, b: str | int | float) -> dict:
        """Add two decimal numbers; return a decimal string."""
        return tools.dispatch('add', {'a': a, 'b': b})

    @server.tool()
    def subtract(a: str | int | float, b: str | int | float) -> dict:
        """Subtract b from a using decimal arithmetic."""
        return tools.dispatch('subtract', {'a': a, 'b': b})

    @server.tool()
    def multiply(a: str | int | float, b: str | int | float) -> dict:
        """Multiply two decimal numbers."""
        return tools.dispatch('multiply', {'a': a, 'b': b})

    @server.tool()
    def divide(a: str | int | float, b: str | int | float) -> dict:
        """Divide a by b using decimal arithmetic."""
        return tools.dispatch('divide', {'a': a, 'b': b})

    return server


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--rules-dir', type=Path, required=True)
    cli.add_argument('--index-dir', type=Path, required=True)
    cli.add_argument('--workspace-root', type=Path, required=True)
    cli.add_argument('--task-id', required=True)
    cli.add_argument('--submission-path', type=Path, required=True)
    cli.add_argument('--embedding-base-url')
    cli.add_argument('--embedding-api-key-env', default='EMBEDDING_API_KEY')
    args = cli.parse_args(argv)
    manifest = json.loads((args.index_dir / 'manifest.json').read_text(encoding='utf-8'))
    import os
    embedder = make_embedder(manifest['backend'], manifest['embedding_model'],
                             args.embedding_base_url,
                             os.getenv(args.embedding_api_key_env) or 'dummy')
    with VectorRuleDatabase(args.rules_dir, args.index_dir, embedder) as database:
        create_server(database, args.workspace_root, args.task_id,
                      args.submission_path).run(transport='stdio')


if __name__ == '__main__':
    main()
