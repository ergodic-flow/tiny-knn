import click
import json
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List


app = FastAPI()

I: List[str] = []
M: np.ndarray | None = None


class Neighbor(BaseModel):
    id: str
    score: float


class EmbeddingRequest(BaseModel):
    v: List[float]


def load_vectors(path: str, normalize: bool) -> tuple[List[str], np.ndarray]:
    """
    Load JSONL serialized vectors from disk to memory.
    """
    ids: List[str] = []
    vecs: List[List[float]] = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            vecs.append(obj["v"])
    matrix = np.asarray(vecs, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        matrix /= norms
    return ids, matrix


@app.post("/neighbors")
def neighbors(
    body: EmbeddingRequest, k: int = Query(default=10, ge=1)
) -> List[Neighbor]:
    """
    Compute K nearest neighbors to the given query vector.
    L2-normalizes vector on request.
    """
    assert M is not None, "index not loaded"
    q = np.asarray(body.v, dtype=np.float32)
    q /= max(np.linalg.norm(q), 1e-10)
    sims = M @ q
    k = min(k, len(sims))
    top_k_idx = np.argpartition(sims, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
    resp = [Neighbor(id=I[i], score=float(sims[i])) for i in top_k_idx]
    return resp


@click.command()
@click.option("--vectors", help="Path to vectors JSONL", required=True)
@click.option("--host", default="0.0.0.0", help="Bind host", required=False)
@click.option("--port", default=8000, type=int, help="Bind port", required=False)
@click.option(
    "--normalize/--no-normalize", default=True, help="L2-normalize vectors at load time", required=False
)
def main(vectors: str, host: str, port: int, normalize: bool) -> None:
    global I, M
    I, M = load_vectors(vectors, normalize)
    print(f"loaded {len(I)} vectors (dim={M.shape[1]})")
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
