"""Convert Neo4j JSONL exports into a portable provision corpus."""

import json
import re
import sys
from datetime import date
from pathlib import Path

from tqdm import tqdm


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_number}: invalid JSON: {exc}') from exc
                if not isinstance(row, dict):
                    raise ValueError(f'{path}:{line_number}: expected an object')
                yield row


def document_for(part_id, documents):
    # A parent may be another part. The document is a suffix of the full identifier.
    for comma in re.finditer(r',\s*', part_id):
        document_id = part_id[comma.end():]
        if document_id in documents:
            return document_id
    return None


def prepare(documents_path, parts_path, output_dir):
    """Write rules.jsonl, preserving full provision text and source metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / 'rules.jsonl'
    if destination.resolve() in {Path(documents_path).resolve(), Path(parts_path).resolve()}:
        raise ValueError('Prepared output must differ from raw inputs')
    documents = {}
    for row in rows(documents_path):
        key = row.get('identifier')
        if not isinstance(key, str) or not key.strip():
            raise ValueError('Document missing identifier')
        if key in documents:
            raise ValueError(f'Duplicate document identifier: {key}')
        documents[key] = {field: row.get(source) for field, source in {
            'law_id': 'lawId', 'document_code': 'documentCode', 'title': 'title',
            'document_type': 'documentType', 'agency_issued': 'agencyIssued',
            'date_issued': 'dateIssued', 'effective_date': 'effectiveDate',
            'expiration_date': 'expirationDate', 'validity_status': 'validityStatus',
            'url': 'url', 'field': 'field'}.items()}
    seen = set()
    stats = {'documents': len(documents), 'rules': 0, 'rules_with_document': 0,
             'rules_without_document': 0, 'empty_parts_skipped': 0}
    temporary = output_dir / 'rules.jsonl.tmp'
    try:
        with temporary.open('w', encoding='utf-8') as output:
            for part in tqdm(rows(parts_path), desc='Preparing provisions', unit='part',
                             disable=not sys.stderr.isatty()):
                rule_id, content = part.get('identifier'), part.get('content')
                if not isinstance(rule_id, str) or not rule_id.strip():
                    raise ValueError('Part missing identifier')
                if rule_id in seen:
                    raise ValueError(f'Duplicate provision identifier: {rule_id}')
                seen.add(rule_id)
                if not isinstance(content, str) or not content.strip():
                    stats['empty_parts_skipped'] += 1
                    continue
                document_id = document_for(rule_id, documents)
                doc = documents.get(document_id, {})
                rule = {'rule_id': rule_id, 'parent_id': part.get('parent'),
                        'document_id': document_id, 'name': part.get('name'),
                        'part_type': part.get('partType'), 'part_id': part.get('partId'),
                        'content': content, **{field: doc.get(field) for field in (
                            'law_id', 'document_code', 'title', 'document_type',
                            'agency_issued', 'date_issued', 'effective_date',
                            'expiration_date', 'validity_status', 'url', 'field')}}
                output.write(json.dumps(rule, ensure_ascii=False) + '\n')
                stats['rules'] += 1
                stats['rules_with_document' if document_id else 'rules_without_document'] += 1
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return stats


class RuleStore:
    """Random-access JSONL store: only IDs and byte offsets are held in memory."""

    def __init__(self, rules_path):
        self.path = Path(rules_path)
        self.source = self.path.open('rb')
        self.offsets = []
        self.id_to_row = {}
        try:
            while True:
                offset = self.source.tell()
                line = self.source.readline()
                if not line:
                    break
                row = json.loads(line)
                rule_id = row['rule_id']
                if rule_id in self.id_to_row:
                    raise ValueError(f'Duplicate provision identifier: {rule_id}')
                self.id_to_row[rule_id] = len(self.offsets)
                self.offsets.append(offset)
        except Exception:
            self.source.close()
            raise

    def close(self):
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self):
        return len(self.offsets)

    def by_row(self, row_number):
        self.source.seek(self.offsets[row_number])
        return json.loads(self.source.readline())

    def get(self, rule_id):
        row_number = self.id_to_row.get(rule_id)
        return self.by_row(row_number) if row_number is not None else None

    def check_validity(self, rule_id, start, end):
        rule = self.get(rule_id)
        if rule is None:
            raise ValueError('Unknown rule_id')
        try:
            left, right = date.fromisoformat(start), date.fromisoformat(end)
        except (ValueError, TypeError) as exc:
            raise ValueError('start and end must be ISO dates YYYY-MM-DD') from exc
        if left > right:
            raise ValueError('start must be on or before end')
        try:
            effective = date.fromisoformat(rule['effective_date']) if rule['effective_date'] else None
            expiration = date.fromisoformat(rule['expiration_date']) if rule['expiration_date'] else None
        except ValueError:
            effective = expiration = None
        if effective and right < effective or expiration and left >= expiration:
            status = 'invalid'
        elif effective and left < effective <= right or expiration and left < expiration <= right:
            status = 'partially_valid'
        elif effective and expiration and effective <= left and right < expiration:
            status = 'valid'
        else:
            status = 'unknown'
        return {'rule_id': rule_id, 'start': start, 'end': end, 'status': status,
                'effective_date': rule['effective_date'],
                'expiration_date': rule['expiration_date'],
                'document_validity_status': rule['validity_status'],
                'note': 'Document-level dates only; provision amendments are not represented.'}
