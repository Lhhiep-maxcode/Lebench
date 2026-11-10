"""One writer, one stateless judge, bounded rewrites; judgments run concurrently."""

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

import httpx
from tqdm import tqdm

if __package__:
    from .situation_prompts import build_judge_messages, build_writer_messages
else:
    from situation_prompts import build_judge_messages, build_writer_messages


async def call_llm(client, args, model, messages, *, judge=False):
    # A NEW two-message request each time: no conversation, history or session ID.
    body = {**args.extra_body, 'model': model, 'temperature': 0, 'max_tokens': args.max_tokens,
            'messages': messages}
    if judge:
        body['response_format'] = {'type': 'json_object'}
    url = (args.judge_base_url or args.base_url) if judge else args.base_url
    response = await client.post(url.rstrip('/')+'/chat/completions', json=body)
    response.raise_for_status()
    choice = response.json()['choices'][0]
    if choice.get('finish_reason') not in (None, 'stop'):
        raise ValueError(f'Incomplete response: finish_reason={choice.get("finish_reason")}')
    text = choice['message'].get('content')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Model returned no final content')
    return text.strip()


def parse_verdict(text):
    verdict = json.loads(text)
    if not isinstance(verdict, dict) or type(verdict.get('passed')) is not bool:
        raise ValueError('Judge must return a boolean passed')
    feedback = verdict.get('feedback')
    if not isinstance(feedback, list) or any(not isinstance(item, str) or not item.strip() for item in feedback):
        raise ValueError('Judge feedback must be a list of nonempty strings')
    if verdict['passed'] != (not feedback):
        raise ValueError('Judge must provide feedback on rejection and no feedback on acceptance')
    return {'passed': verdict['passed'], 'feedback': feedback}


async def extract_one(row, client, args):
    situation, verdict, attempts = None, None, 0
    try:
        source = row.get('markdown')
        if not isinstance(source, str) or not source.strip():
            raise ValueError('markdown is missing or empty')
        for attempt in range(args.max_retries+1):
            writer_messages = build_writer_messages(source, situation, verdict['feedback'] if verdict else None)
            attempts = attempt+1
            situation = await call_llm(client, args, args.writer_model, writer_messages)
            # Judge receives source + CURRENT draft (+ optional static law hints), no history.
            verdict = None
            law_hints = (row.get('citations_law') or []) if getattr(args, 'use_law_hints', False) else None
            judge_messages = build_judge_messages(source, situation, law_hints)
            judge_text = await call_llm(client, args, args.judge_model or args.writer_model,
                                       judge_messages, judge=True)
            verdict = parse_verdict(judge_text)
            if verdict['passed']:
                break
        return {**row, 'situation': situation, 'situation_status': 'passed' if verdict['passed'] else 'needs_review',
                'situation_attempts': attempts, 'situation_judge': verdict, 'situation_error': None}
    except Exception as exc:
        return {**row, 'situation': situation, 'situation_status': 'error', 'situation_attempts': attempts,
                'situation_judge': verdict, 'situation_error': f'{type(exc).__name__}: {str(exc)[:1000]}'}


async def run(args, *, transport=None):
    if args.concurrency < 1 or args.max_retries < 0 or args.max_tokens < 1 or args.timeout <= 0:
        raise ValueError('Invalid concurrency/retry/token/timeout setting')
    if args.limit is not None and args.limit < 1:
        raise ValueError('limit must be positive')
    if not isinstance(args.extra_body, dict) or {'messages', 'stream', 'response_format'} & args.extra_body.keys():
        raise ValueError('extra_body must be an object without messages/stream/response_format')
    source, destination = Path(args.input).resolve(), Path(args.output).resolve()
    if source == destination:
        raise ValueError('Input and output must be different files')
    with source.open(encoding='utf-8-sig') as stream:
        total = sum(bool(line.strip()) for line in stream)
    if args.limit is not None:
        total = min(total, args.limit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    counts, pending = Counter(), set()
    async with httpx.AsyncClient(timeout=args.timeout, transport=transport,
            headers={'Authorization': 'Bearer '+args.api_key},
            limits=httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=args.concurrency)) as client:
        with source.open(encoding='utf-8-sig') as stream, destination.open('w', encoding='utf-8') as output, \
                tqdm(total=total, desc='Judgments', unit='doc', disable=args.no_progress) as progress:

            async def collect_finished():
                nonlocal pending
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    result = task.result()
                    output.write(json.dumps(result, ensure_ascii=False)+'\n')
                    output.flush()
                    counts[result['situation_status']] += 1
                    progress.set_postfix(dict(counts), refresh=False)
                    progress.update(1)

            selected = 0
            try:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    if args.limit is not None and selected >= args.limit:
                        break
                    selected += 1
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict) or not isinstance(row.get('doc_name'), str) or not row['doc_name']:
                            raise ValueError('Each row requires a nonempty string doc_name')
                    except ValueError as exc:
                        counts['invalid_input'] += 1
                        tqdm.write(f'Line {line_number}: {exc}', file=progress.fp)
                        progress.update(1)
                        continue
                    pending.add(asyncio.create_task(extract_one(row, client, args)))
                    # At most concurrency judgments exist in memory/in flight.
                    if len(pending) >= args.concurrency:
                        await collect_finished()
                while pending:
                    await collect_finished()
            finally:
                for task in pending:
                    task.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='src/server/sample.jsonl')
    parser.add_argument('--output', default='outputs/situations_simple.jsonl', help='Output JSONL file (overwritten on each run)')
    parser.add_argument('--base-url', default=os.getenv('VLLM_BASE_URL'))
    parser.add_argument('--writer-model', '--model', dest='writer_model', default=os.getenv('VLLM_MODEL'))
    parser.add_argument('--judge-model', default=None, help='Defaults to writer model, with a separate stateless request')
    parser.add_argument('--judge-base-url', default=None, help='Defaults to base-url')
    parser.add_argument('--api-key', default=os.getenv('VLLM_API_KEY', 'EMPTY'))
    parser.add_argument('--concurrency', type=int, default=8, help='Number of judgments processed at the same time')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum rewrites after the first draft; stop early on pass')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--extra-body', type=json.loads, default={})
    parser.add_argument('--use-law-hints', action='store_true', help='Give the judge unverified citations_law metadata as optional attention hints')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--no-progress', action='store_true')
    args = parser.parse_args()
    if not args.base_url or not args.writer_model:
        parser.error('Set --base-url/--writer-model or VLLM_BASE_URL/VLLM_MODEL')
    try:
        counts = asyncio.run(run(args))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return int(bool(counts.get('error') or counts.get('invalid_input')))


if __name__ == '__main__':
    raise SystemExit(main())
