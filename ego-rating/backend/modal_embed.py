"""Modal-hosted sentence-embedding service for comparability grouping.

The local box can't load torch/sentence-transformers in-process, so embeddings
run on Modal instead. The model is baked into the image at build time (no runtime
download), and loaded once per warm container via @modal.enter.

Deploy once:

    cd ego-rating && modal deploy backend/modal_embed.py

Then the backend calls it (see pairing._embed_modal):

    Embedder = modal.Cls.from_name("ego-rating-embed", "Embedder")
    vecs = Embedder().embed.remote(list_of_texts)   # -> list[list[float]], L2-normalized
"""

import modal

APP_NAME = "ego-rating-embed"
# all-mpnet-base-v2 separates folding sub-tasks far better than MiniLM (see the
# embedder comparison): same sub-task ~0.83, different sub-task ~0.75, off-task
# ~0.28 — a clean gap that a 0.8 threshold can cut. bge-large was worse (high
# floor: unrelated tasks still ~0.5).
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim()
    .pip_install("sentence-transformers>=2.2", "numpy")
    # Bake the model into the image so containers start without a download.
    .run_commands(
        'python -c "from sentence_transformers import SentenceTransformer; '
        f"SentenceTransformer('{MODEL_NAME}')\""
    )
)


@app.cls(image=image, gpu="T4", scaledown_window=300)
class Embedder:
    @modal.enter()
    def load(self):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(MODEL_NAME, device=device)

    @modal.method()
    def embed(self, texts: list[str]) -> list[list[float]]:
        emb = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=256,
        )
        return emb.tolist()


@app.local_entrypoint()
def main():
    """Smoke test: `modal run backend/modal_embed.py`."""
    import numpy as np

    vecs = Embedder().embed.remote(
        ["fold a shirt neatly", "neatly folding a shirt", "wash the dishes"]
    )
    e = np.asarray(vecs)
    sim = e @ e.T
    print("cos(fold, folding) =", round(float(sim[0, 1]), 3))
    print("cos(fold, dishes)  =", round(float(sim[0, 2]), 3))
