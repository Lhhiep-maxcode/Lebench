# Prepare title embeddings for the Vietnamese legal corpus

This workflow reads `Hiepppp/vietnamese-legal-corpus` from Hugging Face, embeds **only** each document's `title`, and writes a new dataset. Each output row retains the original fields (`id`, `full_content`, dates, `relationships`, etc.) with **no embedding array in JSONL**. The vectors are stored in `index.faiss` only. `id_map.jsonl` connects FAISS positions to source document IDs.

## 1. Install and serve the embedding model

```bash
pip install -r requirements-hf-documents.txt
```

Use an OpenAI compatible embedding endpoint for `nvidia/Nemotron-3-Embed-8B-BF16`.

### Two RTX 4060 Ti 16 GB GPUs

First confirm that **both** cards have 16 GB VRAM and support BF16:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -c 'import torch; print("GPU count:", torch.cuda.device_count(), "BF16:", torch.cuda.is_bf16_supported())'
```

The RTX 4060 Ti is an Ada GPU with BF16-capable Tensor Cores. Two 16 GB cards should have enough memory to split the 8B BF16 model, but this exact setup still needs the smoke test below. In a separate terminal, serve it with tensor parallelism 2:

```bash
python3 -m pip install "vllm==0.25.0" requests
CUDA_VISIBLE_DEVICES=0,1 vllm serve nvidia/Nemotron-3-Embed-8B-BF16 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 --dtype bfloat16 \
  --max-model-len 2048 --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 --enforce-eager
```

The checkpoint is about 15.9 GB in total; tensor parallelism divides its weights across the cards. The cards do not have NVLink, so this may be slower than a single sufficiently large GPU. NVIDIA's model card does not explicitly validate BF16 serving on the 4060 Ti. Test the embedding endpoint before preparing the corpus:

```bash
python - <<'PY'
import requests
r = requests.post('http://127.0.0.1:8000/v1/embeddings', json={
    'model': 'nvidia/Nemotron-3-Embed-8B-BF16',
    'input': ['passage: Bộ luật Dân sự 2015'],
}, timeout=120)
r.raise_for_status()
print('dimensions:', len(r.json()['data'][0]['embedding']))
PY
```

Expected output is `dimensions: 4096`. If startup runs out of memory, stop other GPU processes and try `--max-model-len 1024 --max-num-seqs 1`. Use the same model and serving settings when embedding titles and later embedding search queries.

The model produces 4096-dimensional vectors and recommends `passage: ` for stored documents and `query: ` for searches. The preparation command applies these prefixes. `--vector-dim 1024` stores the first 1024 dimensions and normalizes them again, reducing disk and index memory; use `4096` for the full vector. The model card permits this type of dimension reduction.

## 2. Check a small sample

```bash
python3 -m legal_rule_retrieval prepare-hf-documents \
  --output-dir output/hf_documents_faiss_smoke \
  --embedding-base-url http://127.0.0.1:8000/v1 \
  --embedding-model nvidia/Nemotron-3-Embed-8B-BF16 \
  --vector-dim 4096 --batch-size 8 --max-rows 20
```

`--max-rows` is for local checks and cannot be combined with `--upload`. Use a **different output directory** for the full run. To test title retrieval from the generated FAISS index:

```bash
python3 -m legal_rule_retrieval search-hf-documents \
  --prepared-dir output/hf_documents_faiss_smoke \
  --embedding-base-url http://127.0.0.1:8000/v1 \
  --query "hỗ trợ cai nghiện tự nguyện" --top-k 5
```

## 3. Prepare the full corpus and upload

Set `HF_TOKEN` to a Hugging Face token with permission to create and write the **destination dataset repository**. The example destination is `Hiepppp/vietnamese-legal-corpus-faiss-only`; choose another ID if desired. Use a fresh local output directory if you previously prepared the older JSONL-with-vectors format.

```bash
read -rsp 'HF write token: ' HF_TOKEN; echo; export HF_TOKEN
python3 -u -m legal_rule_retrieval prepare-hf-documents \
  --source-repo Hiepppp/vietnamese-legal-corpus \
  --output-dir output/hf_documents_faiss_full \
  --embedding-base-url http://127.0.0.1:8000/v1 \
  --embedding-model nvidia/Nemotron-3-Embed-8B-BF16 \
  --vector-dim 4096 --batch-size 8 --shard-rows 1000 \
  --upload --repo-id Hiepppp/vietnamese-legal-corpus-faiss-only
```

The command resolves the source repository to a fixed commit and streams its `train` split. Each completed JSONL shard is paired with a temporary FAISS checkpoint. If a run is interrupted, rerun the **same command** and output directory: completed pairs are reused and only unfinished documents are embedded again. Once complete, the checkpoints are merged into `index.faiss` and removed. During the merge, FAISS checkpoints and the final index temporarily coexist on disk; the uploaded dataset contains only the final index. The preparation settings, including source revision, embedding model, dimension, and `--max-rows`, must be identical on resume. If the source repository changes between runs, pass its original commit SHA with `--revision` or start a new output directory.

The destination repository receives `data/train-*.jsonl.gz`, `index.faiss`, `id_map.jsonl`, `manifest.json`, and `README.md`. The generated dataset card tells `datasets.load_dataset` to read only the JSONL shards. For example:

```python
from datasets import load_dataset

ds = load_dataset("Hiepppp/vietnamese-legal-corpus-faiss-only", split="train", streaming=True)
row = next(iter(ds))
print(row["title"], "emb_vector" in row)  # False: vectors are in index.faiss
```

The new `manifest.json` has version 4. It is **document level** and cannot be passed as `--final-dir` to the existing `run-python`/`run-dsh` benchmark, which expects provision level `identifier` records and scores individual articles. That benchmark needs a document-to-provision mapping or a provision splitter before it can use this corpus for article recall.
