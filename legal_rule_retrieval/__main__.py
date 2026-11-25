"""Run document-level legal retrieval against a downloaded Hugging Face corpus."""

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run-python', aliases=['run-dsh'],
                              help='Run the document-level legal tool loop in Python')
    run.add_argument('--corpus-dir', type=Path,
                     default=ROOT / 'input' / 'vietnamese-legal-corpus')
    run.add_argument('--tasks', type=Path, default=ROOT / 'input' / 'rule_recall.jsonl')
    run.add_argument('--output', type=Path, default=ROOT / 'output' / 'rule_recall_results.jsonl')
    run.add_argument('--conversation-output', type=Path)
    run.add_argument('--workspace-root', type=Path, default=ROOT / 'output' / 'rule_workspaces')
    run.add_argument('--base-url', required=True, help='OpenAI-compatible chat API ending in /v1')
    run.add_argument('--model', required=True)
    run.add_argument('--api-key-env', default='VLLM_API_KEY')
    run.add_argument('--embedding-base-url', required=True,
                     help='OpenAI-compatible embedding API ending in /v1')
    run.add_argument('--embedding-api-key-env', default='EMBEDDING_API_KEY')
    run.add_argument('--search-threads', type=int, default=1)
    run.add_argument('--max-steps', type=int, default=24)
    run.add_argument('--max-tokens', type=int, default=4096)
    run.add_argument('--context-window', type=int, default=131072)
    run.add_argument('--request-timeout', '--startup-timeout', dest='request_timeout',
                     type=int, default=600)
    run.add_argument('--debug', action='store_true')
    run.add_argument('--verbosity', type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args(argv)

    if not 1 <= args.search_threads <= 8:
        parser.error('--search-threads must be between 1 and 8')
    if args.max_steps < 1 or args.max_tokens < 1 or args.context_window < 4096:
        parser.error('--max-steps and --max-tokens must be positive; --context-window >= 4096')
    if args.max_tokens >= args.context_window:
        parser.error('--max-tokens must be smaller than --context-window')
    if not 1 <= args.request_timeout <= 3600:
        parser.error('--request-timeout must be between 1 and 3600 seconds')
    if not args.tasks.is_file():
        parser.error(f'Tasks file missing: {args.tasks}')

    # Set native thread limits before importing FAISS/NumPy through the runner.
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = str(args.search_threads)

    from .hf_database import require_corpus
    from .python_runner import run_benchmark

    try:
        require_corpus(args.corpus_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    corpus = args.corpus_dir.resolve()
    protected = {args.tasks.resolve()}
    for name, path in (('Result output', args.output),
                       ('Conversation output', args.conversation_output),
                       ('Workspace root', args.workspace_root)):
        if path is not None and (path.resolve() in protected or
                                 path.resolve().is_relative_to(corpus)):
            parser.error(f'{name} must differ from the tasks and corpus paths')
    if (args.conversation_output is not None and
            args.conversation_output.resolve() == args.output.resolve()):
        parser.error('Conversation output must differ from result output')

    summary = run_benchmark(
        args.tasks, args.output, corpus_dir=args.corpus_dir,
        conversation_output=args.conversation_output,
        workspace_root=args.workspace_root, base_url=args.base_url, model=args.model,
        api_key=os.getenv(args.api_key_env, 'dummy'),
        embedding_base_url=args.embedding_base_url,
        embedding_api_key=os.getenv(args.embedding_api_key_env, 'dummy'),
        max_steps=args.max_steps, max_tokens=args.max_tokens,
        request_timeout=args.request_timeout, debug=args.debug,
        verbosity=args.verbosity, search_threads=args.search_threads)
    return 1 if any(summary['statuses'].get(status, 0) for status in
                    ('llm_error', 'runtime_error', 'error', 'no_finish_reason')) else 0


if __name__ == '__main__':
    raise SystemExit(main())
