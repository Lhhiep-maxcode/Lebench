"""Tool contract, dispatch, file workspace, and benchmark counters."""

import hashlib
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


# The native OpenAI schemas and MCP descriptions are generated from this list.
TOOL_SPECS = [
    {'name': 'search_legal_rules',
     'description': 'Embed a natural-language query and search law-document titles; return only each document id and title.',
     'arguments': {'query': 'string', 'top_k': 'integer 1..20'}},
    {'name': 'get_legal_rule',
     'description': 'Read the full_content of a law document by document id.',
     'arguments': {'rule_id': 'string'}},
    {'name': 'check_validity',
     'description': 'Check document validity over an ISO date interval using ngay_co_hieu_luc and ngay_het_hieu_luc.',
     'arguments': {'rule_id': 'string', 'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}},
    {'name': 'check_relationship',
     'description': 'Return the direct relationships of a law document by document id.',
     'arguments': {'rule_id': 'string'}},
    {'name': 'submit_rules',
     'description': 'Submit article citations and finish. Use Điều {article number} {legal document} {citation year}, e.g. Điều 114 Luật Tố tụng hành chính 2010.',
     'arguments': {'rule_ids': 'array of strings'}},
    {'name': 'create_file',
     'description': 'Create a new UTF-8 text file in the private task workspace.',
     'arguments': {'content': 'string', 'path': 'relative path'}},
    {'name': 'read_file',
     'description': 'Read a UTF-8 text file from the private task workspace.',
     'arguments': {'path': 'relative path'}},
    {'name': 'append_file',
     'description': 'Append UTF-8 text to an existing file in the private task workspace.',
     'arguments': {'content': 'string', 'path': 'relative path'}},
    *[{'name': name,
       'description': f'Compute {name}(a, b) using decimal arithmetic; return a decimal string.',
       'arguments': {'a': 'number or decimal string', 'b': 'number or decimal string'}}
      for name in ('add', 'subtract', 'multiply', 'divide')],
]


class ToolSession:
    def __init__(self, database, workspace_root, task_id):
        self.database = database
        self.submitted = None
        self.files_created = 0
        self.bytes_written = 0
        label = re.sub(r'[^a-zA-Z0-9_-]+', '_', str(task_id))[:40] or 'task'
        digest = hashlib.sha256(str(task_id).encode()).hexdigest()[:12]
        self.workspace = (Path(workspace_root) / f'{label}_{digest}').resolve()
        self.workspace.mkdir(mode=0o700, parents=True, exist_ok=True)

    def file_path(self, path):
        if not isinstance(path, str) or not path.strip():
            raise ValueError('path must be a nonempty relative path')
        normalized = path.replace('\\', '/')
        if (normalized.startswith('/') or ':' in normalized.split('/')[0] or
                '..' in normalized.split('/')):
            raise ValueError('path must stay inside the task workspace')
        resolved = (self.workspace / normalized).resolve()
        if not resolved.is_relative_to(self.workspace) or resolved == self.workspace:
            raise ValueError('path must stay inside the task workspace')
        return resolved

    @staticmethod
    def number(value):
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError('a and b must be finite decimal numbers')
        try:
            number = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError('a and b must be finite decimal numbers') from exc
        if not number.is_finite():
            raise ValueError('a and b must be finite decimal numbers')
        return number

    def dispatch(self, action, arguments):
        if action == 'search_legal_rules':
            if set(arguments) - {'query', 'top_k'}:
                raise ValueError('Unexpected search arguments')
            return {'results': self.database.search(arguments.get('query'),
                                                    arguments.get('top_k', 5))}
        if action == 'get_legal_rule':
            return self.database.get(arguments.get('rule_id'))
        if action == 'check_validity':
            return self.database.check_validity(arguments.get('rule_id'),
                                                arguments.get('start'), arguments.get('end'))
        if action == 'check_relationship':
            return self.database.check_relationship(arguments.get('rule_id'))
        if action == 'submit_rules':
            ids = arguments.get('rule_ids')
            if (not isinstance(ids, list) or len(ids) > 100 or
                    any(not isinstance(item, str) for item in ids)):
                raise ValueError('rule_ids must be at most 100 citation strings')
            self.submitted = list(dict.fromkeys(ids))
            return {'submitted_rule_ids': self.submitted}
        if action == 'create_file':
            path, content = self.file_path(arguments.get('path')), arguments.get('content')
            if not isinstance(content, str):
                raise ValueError('content must be UTF-8 text')
            byte_count = len(content.encode('utf-8'))
            if byte_count > 1_000_000:
                raise ValueError('content must be UTF-8 text of at most 1 MB')
            if self.files_created >= 10 or self.bytes_written + byte_count > 2_000_000:
                raise ValueError('Task file quota exceeded (10 files or 2 MB)')
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open('x', encoding='utf-8') as output:
                    output.write(content)
            except FileExistsError as exc:
                raise ValueError('File already exists') from exc
            self.files_created += 1
            self.bytes_written += byte_count
            return {'path': str(path.relative_to(self.workspace)),
                    'characters_written': len(content)}
        if action == 'read_file':
            path = self.file_path(arguments.get('path'))
            if not path.is_file() or path.stat().st_size > 1_000_000:
                raise ValueError('File does not exist or exceeds 1 MB')
            return {'path': str(path.relative_to(self.workspace)),
                    'content': path.read_text(encoding='utf-8')}
        if action == 'append_file':
            path, content = self.file_path(arguments.get('path')), arguments.get('content')
            if not isinstance(content, str):
                raise ValueError('content must be UTF-8 text')
            if not path.is_file():
                raise ValueError('File does not exist')
            byte_count = len(content.encode('utf-8'))
            if (path.stat().st_size + byte_count > 1_000_000 or
                    self.bytes_written + byte_count > 2_000_000):
                raise ValueError('Task file quota exceeded (1 MB per file or 2 MB total)')
            with path.open('a', encoding='utf-8') as output:
                output.write(content)
            self.bytes_written += byte_count
            return {'path': str(path.relative_to(self.workspace)),
                    'characters_written': len(content)}
        if action in ('add', 'subtract', 'multiply', 'divide'):
            a, b = self.number(arguments.get('a')), self.number(arguments.get('b'))
            if action == 'divide' and b == 0:
                raise ValueError('Division by zero')
            result = {'add': lambda: a + b, 'subtract': lambda: a - b,
                      'multiply': lambda: a * b, 'divide': lambda: a / b}[action]()
            return {'result': str(result)}
        raise ValueError(f'Unknown action: {action}')


def tool_metrics(trace):
    tool_calls = sum(1 for step in trace if step.get('tool_called'))
    work_calls = sum(1 for step in trace if step.get('tool_called')
                     and step.get('action') != 'submit_rules')
    retrieval_calls = sum(1 for step in trace if step.get('action') in
                          ('search_legal_rules', 'get_legal_rule') and step.get('tool_called'))
    failed_calls = sum(1 for step in trace if step.get('tool_called')
                       and 'error' in step.get('observation', {}))
    return {'tool_calls': tool_calls, 'work_calls': work_calls,
            'retrieval_calls': retrieval_calls, 'failed_tool_calls': failed_calls,
            'llm_steps': len(trace)}


def aggregate(results):
    scored = [item['metrics'] for item in results if 'f1' in item['metrics']]
    base = {'tasks': len(results), 'scored_tasks': len(scored),
            'tool_calls': sum(item['metrics']['tool_calls'] for item in results),
            'work_calls': sum(item['metrics']['work_calls'] for item in results),
            'retrieval_calls': sum(item['metrics']['retrieval_calls'] for item in results)}
    if not scored:
        return base
    tp = sum(m['true_positives'] for m in scored)
    predicted = sum(m['predicted_count'] for m in scored)
    gold = sum(m['gold_count'] for m in scored)
    micro_precision = tp / predicted if predicted else (1.0 if not gold else 0.0)
    micro_recall = tp / gold if gold else (1.0 if not predicted else 0.0)
    return {**base,
            'macro_precision': sum(m['precision'] for m in scored) / len(scored),
            'macro_recall': sum(m['recall'] for m in scored) / len(scored),
            'macro_f1': sum(m['f1'] for m in scored) / len(scored),
            'micro_precision': micro_precision, 'micro_recall': micro_recall,
            'micro_f1': (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                         if micro_precision + micro_recall else 0.0),
            'mean_tool_call_efficiency': sum(m['tool_call_efficiency'] for m in scored) / len(scored)}
