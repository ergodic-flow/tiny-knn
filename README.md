# Tiny KNN

A small yet mighty brute-force KNN service, tested up to 1M vectors.
Good enough for prototypes and experimentation.

## Usage

Run the server:

```bash
uv run main.py --vectors data/sample_vecs.jsonl
```

Curl the endpoint:

```bash
curl -s -X POST "http://localhost:8000/neighbors?k=5" \
  -H 'Content-Type: application/json' \
  -d '{"v":[0.9,0.1,0.0]}' | jq
```

Responds with:

```
[
  {
    "id": "quux2",
    "score": 0.5623031854629517
  },
  {
    "id": "foo3",
    "score": 0.5607672333717346
  },
  {
    "id": "bar2",
    "score": 0.5517436861991882
  },
  {
    "id": "bar",
    "score": 0.49591827392578125
  },
  {
    "id": "bar3",
    "score": 0.4637247920036316
  }
]
```

