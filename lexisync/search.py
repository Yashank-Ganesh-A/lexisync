
import re
import time
from collections import defaultdict

from rank_bm25 import BM25Okapi


def toks(text):
    """Tokenize words and preserve numeric identifiers."""
    return re.findall(
        r"[a-z0-9]+(?:\.[0-9]+)*",
        text.lower()
    )


class SearchEngine:

    def __init__(self, passages):

        if not passages:
            raise ValueError("No extracted text passages.")

        self.passages = passages

        self.bm = BM25Okapi([
            toks(
                x.get("section", "") + " " +
                x.get("text", "")
            )
            for x in passages
        ])

        self.encoder = None
        self.vec = None
        self.cross = None

        self.embed_error = None
        self.cross_error = None

        # Load dense retrieval model.
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

            self.vec = self.encoder.encode(
                [x.get("text", "") for x in passages],
                normalize_embeddings=True,
                show_progress_bar=False
            )

        except Exception as exc:
            self.embed_error = str(exc)

        # Load cross-encoder reranking model.
        try:
            from sentence_transformers import CrossEncoder

            self.cross = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )

        except Exception as exc:
            self.cross_error = str(exc)

    def find(self, q, k=7, mode="hybrid"):

        start = time.perf_counter()

        # BM25 lexical retrieval.
        lexical_scores = self.bm.get_scores(toks(q))

        lexical_order = sorted(
            range(len(lexical_scores)),
            key=lambda i: lexical_scores[i],
            reverse=True
        )

        dense_order = []
        dense_scores = None

        # Dense semantic retrieval.
        if self.encoder is not None:

            import numpy as np

            query_vector = self.encoder.encode(
                [q],
                normalize_embeddings=True
            )[0]

            dense_scores = np.asarray(self.vec) @ query_vector

            dense_order = sorted(
                range(len(dense_scores)),
                key=lambda i: float(dense_scores[i]),
                reverse=True
            )

        candidate_count = max(20, k * 4)

        if mode == "dense" and dense_order:
            candidates = dense_order[:candidate_count]

        elif mode == "bm25":
            candidates = lexical_order[:candidate_count]

        else:
            # Reciprocal Rank Fusion.
            fused = defaultdict(float)

            for rank, index in enumerate(lexical_order[:60]):
                fused[index] += 1 / (60 + rank)

            for rank, index in enumerate(dense_order[:60]):
                fused[index] += 1 / (60 + rank)

            candidates = sorted(
                fused,
                key=fused.get,
                reverse=True
            )[:candidate_count]

        # Cross-encoder reranking.
        if self.cross is not None and candidates:

            cross_scores = self.cross.predict([
                (
                    q,
                    self.passages[index].get("text", "")
                )
                for index in candidates
            ])

            ordered = sorted(
                zip(candidates, cross_scores),
                key=lambda item: float(item[1]),
                reverse=True
            )[:k]

        else:

            ordered = [
                (index, float(lexical_scores[index]))
                for index in candidates[:k]
            ]

        hits = []

        for index, score in ordered:

            passage = dict(self.passages[index])

            passage["score"] = float(score)

            if dense_scores is not None:
                passage["dense_score"] = float(
                    dense_scores[index]
                )

            passage["bm25_score"] = float(
                lexical_scores[index]
            )

            hits.append(passage)

        diagnostics = {
            "dense": bool(dense_order),
            "cross_encoder": bool(self.cross),
            "mode": mode,
            "passage_count": len(self.passages),
            "candidate_count": len(candidates),
            "returned_count": len(hits),
            "embedding_error": self.embed_error,
            "cross_encoder_error": self.cross_error
        }

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        return hits, elapsed_ms, diagnostics
