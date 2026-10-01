from pathlib import Path
from urllib.parse import urlparse
import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import torch

from PIL import Image
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer, util
from transformers import AutoImageProcessor, AutoModel

from egomimic.utils.aws.aws_data_utils import load_env, get_boto3_s3_client

DEFAULT_SEMANTIC_MODEL = "all-MiniLM-L6-v2"
DEFAULT_SEMANTIC_THRESHOLD = 0.80
DEFAULT_VISUAL_MODEL = "facebook/dinov2-small"
DEFAULT_VISUAL_FRAMES = 5
DEFAULT_REFERENCE_VISUAL_DISTANCE = 0.647402822971344


def clean_series(series):
    series = pd.Series(series).dropna().astype(str).str.strip()
    return series[series != ""]


def normalized_evenness(series):
    series = clean_series(series)
    counts = series.value_counts()
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(entropy(probs) / np.log(len(counts)))


def categorical_diversity(series, reference_richness):
    series = clean_series(series)
    n = len(series)
    if n == 0:
        return {
            "n": 0,
            "richness": 0,
            "reference_richness": int(reference_richness),
            "budget_coverage": 0.0,
            "evenness": 0.0,
            "diversity": 0.0,
        }

    richness = int(series.nunique())
    max_possible_richness = min(n, reference_richness)
    coverage = richness / max_possible_richness if max_possible_richness > 0 else 0.0
    evenness = normalized_evenness(series)
    diversity = coverage * evenness

    return {
        "n": int(n),
        "richness": richness,
        "reference_richness": int(reference_richness),
        "budget_coverage": float(coverage),
        "evenness": float(evenness),
        "diversity": float(diversity),
    }


class BehaviorNormalizer:
    def __init__(self, model_name=DEFAULT_SEMANTIC_MODEL, similarity_threshold=DEFAULT_SEMANTIC_THRESHOLD):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        self.task_to_cluster = {}
        self.reference_cluster_count = None

    @staticmethod
    def _task_vocabulary_hash(task_list):
        payload = "\n".join(sorted(task_list))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def fit(self, tasks, cache_path="track2/cache/behavior_clusters.json"):
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        task_list = clean_series(tasks).drop_duplicates().tolist()
        vocabulary_hash = self._task_vocabulary_hash(task_list)

        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)

                compatible = (
                    cached.get("model_name") == self.model_name
                    and float(cached.get("similarity_threshold", -1)) == float(self.similarity_threshold)
                    and cached.get("task_vocabulary_hash") == vocabulary_hash
                )

                if compatible:
                    self.task_to_cluster = {k: int(v) for k, v in cached["task_to_cluster"].items()}
                    self.reference_cluster_count = int(cached["reference_cluster_count"])
                    print("Loaded behavior clusters:", self.reference_cluster_count)
                    return self
            except Exception:
                pass

        print("Embedding", len(task_list), "unique task labels...")
        embeddings = self.model.encode(
            task_list,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        communities = util.community_detection(
            embeddings,
            threshold=self.similarity_threshold,
            min_community_size=2,
            batch_size=1024,
            show_progress_bar=True,
        )

        mapping = {}
        cluster_id = 0

        for community in communities:
            added = False
            for idx in community:
                task = task_list[idx]
                if task not in mapping:
                    mapping[task] = cluster_id
                    added = True
            if added:
                cluster_id += 1

        for task in task_list:
            if task not in mapping:
                mapping[task] = cluster_id
                cluster_id += 1

        self.task_to_cluster = mapping
        self.reference_cluster_count = cluster_id

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "task_to_cluster": mapping,
                    "reference_cluster_count": self.reference_cluster_count,
                    "similarity_threshold": self.similarity_threshold,
                    "model_name": self.model_name,
                    "task_vocabulary_hash": vocabulary_hash,
                    "raw_task_count": len(task_list),
                },
                f,
                ensure_ascii=False,
            )

        print("Behavior clusters created:", self.reference_cluster_count)
        return self

    def transform(self, tasks):
        tasks = pd.Series(tasks)
        clusters = []

        for task in tasks:
            if pd.isna(task):
                clusters.append(np.nan)
                continue

            task = str(task).strip()
            if task == "":
                clusters.append(np.nan)
                continue

            cluster = self.task_to_cluster.get(task)
            clusters.append(cluster if cluster is not None else f"unknown::{task}")

        return pd.Series(clusters, index=tasks.index)

    def score(self, tasks):
        clusters = self.transform(tasks)
        result = categorical_diversity(clusters, self.reference_cluster_count)
        result["reference_behavior_clusters"] = int(self.reference_cluster_count)
        result["semantic_model"] = self.model_name
        result["semantic_threshold"] = float(self.similarity_threshold)
        return result


class EmbodimentDiversity:
    def __init__(self, reference_df):
        self.reference_embodiment_richness = int(
            clean_series(reference_df["embodiment"]).nunique()
        )

    def score(self, df):
        return categorical_diversity(
            df["embodiment"],
            self.reference_embodiment_richness,
        )


class ContextVisualDiversity:
    def __init__(
        self,
        model_name=DEFAULT_VISUAL_MODEL,
        n_frames=DEFAULT_VISUAL_FRAMES,
        cache_dir="track2/cache/visual",
        reference_visual_distance=DEFAULT_REFERENCE_VISUAL_DISTANCE,
    ):
        load_env()
        self.s3 = get_boto3_s3_client()
        self.model_name = model_name
        self.n_frames = int(n_frames)
        self.reference_visual_distance = float(reference_visual_distance)

        self.cache_dir = Path(cache_dir)
        self.video_dir = self.cache_dir / "videos"
        self.embedding_dir = self.cache_dir / "embeddings"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print("Visual model loaded on:", self.device)

    def _cache_key(self, s3_path):
        payload = f"{self.model_name}|{self.n_frames}|{s3_path}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _download_video(self, s3_path, local_path):
        if local_path.exists():
            return
        parsed = urlparse(s3_path)
        self.s3.download_file(parsed.netloc, parsed.path.lstrip("/"), str(local_path))

    def _extract_uniform_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            cap.release()
            return []

        frame_indices = (
            np.linspace(0.10, 0.90, self.n_frames) * (frame_count - 1)
        ).astype(int)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))

        cap.release()
        return frames

    def episode_embedding(self, s3_path, delete_video_after=True):
        key = self._cache_key(s3_path)
        embedding_path = self.embedding_dir / f"{key}.npy"

        if embedding_path.exists():
            return np.load(embedding_path)

        video_path = self.video_dir / f"{key}.mp4"
        self._download_video(s3_path, video_path)

        frames = self._extract_uniform_frames(video_path)
        if not frames:
            raise RuntimeError(f"No frames extracted: {s3_path}")

        inputs = self.processor(images=frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        frame_embeddings = outputs.last_hidden_state[:, 0, :]
        frame_embeddings = torch.nn.functional.normalize(frame_embeddings, p=2, dim=1)

        episode_embedding = frame_embeddings.mean(dim=0)
        episode_embedding = torch.nn.functional.normalize(episode_embedding, p=2, dim=0)
        episode_embedding = episode_embedding.cpu().numpy()

        np.save(embedding_path, episode_embedding)

        if delete_video_after and video_path.exists():
            video_path.unlink()

        return episode_embedding

    def score(self, paths, sample_size=200, random_state=42):
        paths = clean_series(paths).drop_duplicates()

        if len(paths) <= 1:
            return {
                "n": int(len(paths)),
                "failures": 0,
                "raw_cosine_distance": 0.0,
                "reference_visual_distance": self.reference_visual_distance,
                "relative_context_spread": 0.0,
                "context_visual_diversity": 0.0,
            }

        if sample_size is not None:
            paths = paths.sample(
                n=min(int(sample_size), len(paths)),
                random_state=random_state,
            )

        embeddings = []
        failures = 0

        for i, path in enumerate(paths):
            try:
                embeddings.append(self.episode_embedding(path))
                print(f"Visual {i + 1}/{len(paths)}")
            except Exception as e:
                failures += 1
                print("Visual episode failed:", e)

        if len(embeddings) <= 1:
            return {
                "n": int(len(embeddings)),
                "failures": int(failures),
                "raw_cosine_distance": 0.0,
                "reference_visual_distance": self.reference_visual_distance,
                "relative_context_spread": 0.0,
                "context_visual_diversity": 0.0,
            }

        embeddings = np.stack(embeddings)
        distances = cosine_distances(embeddings)
        upper = distances[np.triu_indices_from(distances, k=1)]
        raw_distance = float(upper.mean())

        relative = float(raw_distance / self.reference_visual_distance)
        bounded = float(np.clip(relative, 0.0, 1.0))

        return {
            "n": int(len(embeddings)),
            "failures": int(failures),
            "raw_cosine_distance": raw_distance,
            "reference_visual_distance": self.reference_visual_distance,
            "relative_context_spread": relative,
            "context_visual_diversity": bounded,
        }


class EgoVerseDiversityEvaluator:
    def __init__(
        self,
        reference_df,
        semantic_model=DEFAULT_SEMANTIC_MODEL,
        semantic_threshold=DEFAULT_SEMANTIC_THRESHOLD,
        visual_model=DEFAULT_VISUAL_MODEL,
        visual_frames=DEFAULT_VISUAL_FRAMES,
        reference_visual_distance=DEFAULT_REFERENCE_VISUAL_DISTANCE,
    ):
        self.behavior = BehaviorNormalizer(
            semantic_model,
            semantic_threshold,
        ).fit(reference_df["task"])

        self.embodiment = EmbodimentDiversity(reference_df)

        self.visual_model = visual_model
        self.visual_frames = visual_frames
        self.reference_visual_distance = reference_visual_distance
        self.visual = None

    def _get_visual(self):
        if self.visual is None:
            self.visual = ContextVisualDiversity(
                model_name=self.visual_model,
                n_frames=self.visual_frames,
                reference_visual_distance=self.reference_visual_distance,
            )
        return self.visual

    def evaluate(self, subset_df, visual_sample_size=200, random_state=42):
        behavior = self.behavior.score(subset_df["task"])
        embodiment = self.embodiment.score(subset_df)

        context = self._get_visual().score(
            subset_df["zarr_mp4_path"],
            sample_size=visual_sample_size,
            random_state=random_state,
        )

        behavior_score = behavior["diversity"]
        context_score = context["context_visual_diversity"]
        embodiment_score = embodiment["diversity"]

        overall = float(
            (behavior_score + context_score + embodiment_score) / 3.0
        )

        return {
            "behavior_diversity": float(behavior_score),
            "context_visual_diversity": float(context_score),
            "embodiment_diversity": float(embodiment_score),
            "overall_diversity": overall,
            "behavior_details": behavior,
            "context_details": context,
            "embodiment_details": embodiment,
        }

    def compare(self, subsets, visual_sample_size=200, random_state=42):
        available = []
        for subset in subsets.values():
            available.append(
                clean_series(subset["zarr_mp4_path"]).drop_duplicates().shape[0]
            )

        fair_visual_n = min(int(visual_sample_size), min(available))

        if fair_visual_n < 2:
            raise ValueError(
                "Each dataset must contain at least two unique video paths."
            )

        print("Matched visual sample size:", fair_visual_n)

        rows = []

        for name, subset in subsets.items():
            print("\nEvaluating:", name)

            report = self.evaluate(
                subset,
                visual_sample_size=fair_visual_n,
                random_state=random_state,
            )

            b = report["behavior_details"]
            v = report["context_details"]
            e = report["embodiment_details"]

            rows.append({
                "subset": name,
                "overall_diversity": report["overall_diversity"],
                "behavior_diversity": report["behavior_diversity"],
                "context_visual_diversity": report["context_visual_diversity"],
                "embodiment_diversity": report["embodiment_diversity"],
                "behavior_richness": b["richness"],
                "behavior_coverage": b["budget_coverage"],
                "behavior_evenness": b["evenness"],
                "reference_behavior_clusters": b["reference_behavior_clusters"],
                "embodiment_richness": e["richness"],
                "embodiment_coverage": e["budget_coverage"],
                "embodiment_evenness": e["evenness"],
                "visual_raw_distance": v["raw_cosine_distance"],
                "context_relative_spread": v["relative_context_spread"],
                "reference_visual_distance": v["reference_visual_distance"],
                "visual_n": v["n"],
                "visual_failures": v["failures"],
            })

        result_df = (
            pd.DataFrame(rows)
            .sort_values("overall_diversity", ascending=False)
            .reset_index(drop=True)
        )

        result_df["rank"] = np.arange(1, len(result_df) + 1)

        first = [
            "rank",
            "subset",
            "overall_diversity",
            "behavior_diversity",
            "context_visual_diversity",
            "embodiment_diversity",
        ]
        rest = [c for c in result_df.columns if c not in first]

        return result_df[first + rest]
