"""Client for the user-hosted OpenAI-compatible embedding endpoint."""

import json
import urllib.error
import urllib.request

import numpy as np

from .network import api_base_url, urlopen_direct


class APIEmbedder:
    def __init__(self, base_url, model, api_key='dummy', timeout=120):
        if not base_url or not model:
            raise ValueError('Embedding requires --embedding-base-url and a model')
        self.url = api_base_url(base_url) + '/embeddings'
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def embed(self, texts):
        payload = {'model': self.model, 'input': texts, 'encoding_format': 'float'}
        request = urllib.request.Request(
            self.url, data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + self.api_key})
        try:
            with urlopen_direct(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode('utf-8', errors='replace')
            raise RuntimeError(f'Embedding HTTP {exc.code}: {detail}') from exc
        ordered = sorted(result['data'], key=lambda item: item['index'])
        if (len(ordered) != len(texts) or
                [item['index'] for item in ordered] != list(range(len(texts)))):
            raise ValueError('Embedding server returned incomplete or misordered batch')
        return np.asarray([item['embedding'] for item in ordered], dtype=np.float32)
