"""Prepare provisions, embed them, then run a legal-rule retrieval agent."""

import argparse
import json
import os
from pathlib import Path

from .corpus import prepare, rows
from .files import private_text_output
from .harness import DemoClient, OpenAICompatibleClient, aggregate, run_task
from .vectors import VectorRuleDatabase, build_vectors, make_embedder
from .final_data import prepare_final


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS = Path(__file__).resolve().parent / 'demo_tasks.jsonl'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare', help='Convert raw document/part JSONL into provisions')
    prep.add_argument('--documents', type=Path, required=True)
    prep.add_argument('--parts', type=Path, required=True)
    prep.add_argument('--output-dir', type=Path, default=ROOT / 'output' / 'legal_rule_corpus')

    final = commands.add_parser('prepare-final', help='Embed titles in the final single-file legal corpus')
    final.add_argument('--parts', type=Path, required=True)
    final.add_argument('--output-dir', type=Path, required=True)
    final.add_argument('--embedding-backend', choices=('api', 'local', 'hash'), default='api')
    final.add_argument('--embedding-model', help='Embedding model name or local model directory')
    final.add_argument('--embedding-base-url', help='OpenAI-compatible local embedding endpoint')
    final.add_argument('--embedding-api-key-env', default='EMBEDDING_API_KEY')
    final.add_argument('--batch-size', type=int, default=32)
    final.add_argument('--timeout', type=float, default=120)

    embed = commands.add_parser('embed', help='Embed prepared provision chunks and build FAISS index')
    embed.add_argument('--rules-dir', type=Path, default=ROOT / 'output' / 'legal_rule_corpus')
    embed.add_argument('--index-dir', type=Path, default=ROOT / 'output' / 'legal_rule_index')
    embed.add_argument('--embedding-backend', choices=('api', 'local', 'hash'), default='api')
    embed.add_argument('--embedding-model', help='Embedding model name or local sentence-transformers path')
    embed.add_argument('--embedding-base-url', help='OpenAI-compatible embedding base URL ending in /v1')
    embed.add_argument('--embedding-api-key-env', default='EMBEDDING_API_KEY')
    embed.add_argument('--batch-size', type=int, default=32)
    embed.add_argument('--chunk-chars', type=int, default=1800)
    embed.add_argument('--overlap', type=int, default=200)
    embed.add_argument('--passage-prefix', default='', help='Optional model-specific prefix for indexed passages')
    embed.add_argument('--query-prefix', default='', help='Optional model-specific prefix for search queries')
    embed.add_argument('--timeout', type=float, default=120)

    run = commands.add_parser('run', help='Run benchmark tasks against the prepared vector index')
    run.add_argument('--rules-dir', type=Path, default=ROOT / 'output' / 'legal_rule_corpus')
    run.add_argument('--index-dir', type=Path, default=ROOT / 'output' / 'legal_rule_index')
    run.add_argument('--tasks', type=Path, required=True)
    run.add_argument('--output', type=Path, default=ROOT / 'output' / 'rule_retrieval_results.jsonl')
    run.add_argument('--workspace-root', type=Path, default=ROOT / 'output' / 'rule_workspaces')
    run.add_argument('--base-url', required=True, help='Chat model base URL ending in /v1')
    run.add_argument('--model', required=True, help='Chat model name')
    run.add_argument('--api-key-env', default='VLLM_API_KEY')
    run.add_argument('--embedding-base-url', help='Embedding server base URL; required for API index')
    run.add_argument('--embedding-api-key-env', default='EMBEDDING_API_KEY')
    run.add_argument('--extra-body', default='{}', help='Additional JSON fields for chat requests')
    run.add_argument('--timeout', type=float, default=120)
    run.add_argument('--max-steps', type=int, default=8)

    dsh = commands.add_parser('run-python', aliases=['run-dsh'],
                              help='Run the legal tool loop directly in Python; run-dsh is an alias')
    dsh.add_argument('--rules-dir', type=Path, default=ROOT / 'output' / 'legal_rule_corpus')
    dsh.add_argument('--index-dir', type=Path, default=ROOT / 'output' / 'legal_rule_index')
    dsh.add_argument('--final-dir', type=Path, help='Single-file final corpus prepared by prepare-final')
    dsh.add_argument('--tasks', type=Path, required=True)
    dsh.add_argument('--output', type=Path, default=ROOT / 'output' / 'rule_retrieval_dsh.jsonl')
    dsh.add_argument('--conversation-output', type=Path,
                     help='Optional OpenAI chat-template JSONL with messages, tools, and reasoning_content')
    dsh.add_argument('--workspace-root', type=Path, default=ROOT / 'output' / 'rule_workspaces')
    dsh.add_argument('--base-url', required=True, help='DeepSeek-compatible chat API base URL')
    dsh.add_argument('--model', required=True)
    dsh.add_argument('--api-key-env', default='VLLM_API_KEY')
    dsh.add_argument('--embedding-base-url')
    dsh.add_argument('--embedding-api-key-env', default='EMBEDDING_API_KEY')
    dsh.add_argument('--max-steps', type=int, default=12)
    dsh.add_argument('--max-tokens', type=int, default=4096)
    dsh.add_argument('--context-window', type=int, default=131072)
    dsh.add_argument('--request-timeout', '--startup-timeout', dest='request_timeout',
                     type=int, default=600,
                     help='Seconds to wait for each chat/embedding HTTP response; --startup-timeout is an alias')
    dsh.add_argument('--debug', action='store_true',
                     help='Print exception tracebacks and at least verbosity level 1')
    dsh.add_argument('--verbosity', type=int, choices=(0, 1, 2), default=0,
                     help='0: task results; 1: live step/tool progress; 2: include LLM text, tool arguments and observations')

    demo = commands.add_parser('demo', help='Offline end-to-end demo with hash vectors (not real embeddings)')
    demo.add_argument('--documents', type=Path, default=ROOT / 'src_data' / 'neo4j_documents_sample.jsonl')
    demo.add_argument('--parts', type=Path, default=ROOT / 'src_data' / 'neo4j_parts_sample.jsonl')
    demo.add_argument('--tasks', type=Path, default=DEFAULT_TASKS)
    demo.add_argument('--output-dir', type=Path, default=ROOT / 'output' / 'legal_rule_demo')
    args = parser.parse_args(argv)

    if args.command == 'prepare':
        if not args.documents.is_file() or not args.parts.is_file():
            parser.error('Raw document and part JSONL files must exist')
        print(json.dumps(prepare(args.documents, args.parts, args.output_dir), ensure_ascii=False))
        print('Prepared:', args.output_dir / 'rules.jsonl')
        return 0

    if args.command == 'prepare-final':
        if not args.parts.is_file():
            parser.error('Final legal parts JSONL does not exist')
        try:
            embedder = make_embedder(args.embedding_backend, args.embedding_model,
                                     args.embedding_base_url,
                                     os.getenv(args.embedding_api_key_env, 'dummy'), args.timeout)
            manifest = prepare_final(args.parts, args.output_dir, embedder,
                                     args.embedding_backend, args.embedding_model,
                                     args.batch_size)
        except (ValueError, ImportError, RuntimeError) as exc:
            parser.error(str(exc))
        print('Prepared final corpus:', args.output_dir / 'rules.jsonl')
        print('Manifest:', json.dumps(manifest, ensure_ascii=False))
        return 0

    if args.command == 'embed':
        if not (args.rules_dir / 'rules.jsonl').is_file():
            parser.error('Prepared rules missing; run prepare first')
        if args.embedding_backend == 'hash':
            print('WARNING: hash vectors are only for offline plumbing tests.')
        try:
            embedder = make_embedder(args.embedding_backend, args.embedding_model,
                                     args.embedding_base_url,
                                     os.getenv(args.embedding_api_key_env, 'dummy'), args.timeout)
            manifest = build_vectors(args.rules_dir, args.index_dir, embedder,
                                     args.embedding_backend, args.embedding_model,
                                     args.batch_size, args.chunk_chars, args.overlap,
                                     args.passage_prefix, args.query_prefix)
        except (ValueError, ImportError, RuntimeError) as exc:
            parser.error(str(exc))
        print('Index:', json.dumps(manifest, ensure_ascii=False))
        return 0

    if args.command in ('run-python', 'run-dsh'):
        from .python_runner import run_benchmark
        if not args.final_dir:
            parser.error('--final-dir is required for the direct Python runner')
        if args.final_dir:
            args.rules_dir = args.index_dir = args.final_dir
        if not args.tasks.is_file() or not (args.rules_dir / 'rules.jsonl').is_file():
            parser.error('Tasks or prepared rules missing; run prepare first')
        if not (args.index_dir / 'manifest.json').is_file():
            parser.error('Vector index missing; run embed first')
        if args.max_steps < 1 or args.max_tokens < 1 or args.context_window < 4096:
            parser.error('--max-steps and --max-tokens must be positive; --context-window >= 4096')
        if args.max_tokens >= args.context_window:
            parser.error('--max-tokens must be smaller than --context-window')
        if not 1 <= args.request_timeout <= 3600:
            parser.error('--request-timeout must be between 1 and 3600 seconds')
        protected = {args.tasks.resolve(), (args.rules_dir / 'rules.jsonl').resolve(),
                     (args.index_dir / 'index.faiss').resolve(),
                     (args.index_dir / 'manifest.json').resolve()}
        if args.output.resolve() in protected:
            parser.error('Result output must differ from tasks, rules, and index')
        if args.conversation_output and args.conversation_output.resolve() in protected:
            parser.error('Conversation output must differ from tasks, rules, and index')
        if args.conversation_output and args.conversation_output.resolve() == args.output.resolve():
            parser.error('Conversation output must differ from result output')
        summary = run_benchmark(args.tasks, args.output,
                      conversation_output=args.conversation_output,
                      final_dir=args.final_dir, workspace_root=args.workspace_root,
                      base_url=args.base_url, model=args.model,
                      api_key=os.getenv(args.api_key_env, 'dummy'),
                      max_steps=args.max_steps, max_tokens=args.max_tokens,
                      embedding_base_url=args.embedding_base_url,
                      embedding_api_key=os.getenv(args.embedding_api_key_env, 'dummy'),
                      request_timeout=args.request_timeout, debug=args.debug,
                      verbosity=args.verbosity)
        return 1 if any(summary['statuses'].get(status, 0) for status in
                        ('llm_error', 'runtime_error', 'error', 'no_finish_reason')) else 0

    if args.command == 'demo':
        if not args.documents.is_file() or not args.parts.is_file() or not args.tasks.is_file():
            parser.error('Demo source JSONL or tasks file missing')
        rules_dir, index_dir = args.output_dir / 'prepared', args.output_dir / 'index'
        result_path, workspace = args.output_dir / 'results.jsonl', args.output_dir / 'workspaces'
        print('Prepared:', json.dumps(prepare(args.documents, args.parts, rules_dir), ensure_ascii=False))
        print('Index:', json.dumps(build_vectors(rules_dir, index_dir, make_embedder('hash'), 'hash'),
                                   ensure_ascii=False))
        client = DemoClient()
        embedder = make_embedder('hash')
    else:
        if not args.tasks.is_file() or not (args.rules_dir / 'rules.jsonl').is_file():
            parser.error('Tasks or prepared rules missing; run prepare first')
        if args.max_steps < 1:
            parser.error('--max-steps must be positive')
        try:
            extra = json.loads(args.extra_body)
        except json.JSONDecodeError as exc:
            parser.error(f'--extra-body must be JSON: {exc}')
        if not isinstance(extra, dict):
            parser.error('--extra-body must be a JSON object')
        manifest_path = args.index_dir / 'manifest.json'
        if not manifest_path.is_file():
            parser.error('Vector index missing; run embed first')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        try:
            embedder = make_embedder(manifest['backend'], manifest['embedding_model'],
                         args.embedding_base_url, os.getenv(args.embedding_api_key_env, 'dummy'), args.timeout)
        except (ValueError, ImportError) as exc:
            parser.error(str(exc))
        client = OpenAICompatibleClient(args.base_url, args.model,
                         api_key=os.getenv(args.api_key_env, 'dummy'), timeout=args.timeout,
                         extra_body=extra)
        rules_dir, index_dir = args.rules_dir, args.index_dir
        result_path, workspace = args.output, args.workspace_root

    protected = {args.tasks.resolve(), (rules_dir / 'rules.jsonl').resolve(),
                 (index_dir / 'index.faiss').resolve(), (index_dir / 'manifest.json').resolve()}
    if result_path.resolve() in protected:
        parser.error('Result output must differ from tasks, rules, and index')
    result_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with VectorRuleDatabase(rules_dir, index_dir, embedder) as database, private_text_output(result_path) as output:
        for task in rows(args.tasks):
            result = run_task(task, database, client,
                              max_steps=args.max_steps if args.command == 'run' else 8,
                              workspace_root=workspace)
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
            output.flush()
            results.append({'metrics': result['metrics']})
            print(f"{result['id']}: {result['status']}, F1={result['metrics'].get('f1', 'n/a')}, "
                  f"tools={result['metrics']['tool_calls']}")
    print('Aggregate:', json.dumps(aggregate(results), ensure_ascii=False))
    print('Trace:', result_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
