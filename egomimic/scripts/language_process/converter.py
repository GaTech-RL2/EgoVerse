import json
import os
from abc import abstractmethod

from openai import OpenAI

# Annotation ROLE is encoded in the zarr KEY NAME (one array per role) —
# entries are plain ``(text, start_idx, end_idx)`` with no level field. The
# PI subtask-prediction pipeline conditions on ``annotations_task`` and
# predicts ``annotations_subtask``; see
# ``egomimic/rldb/annotation_processing.py``. The task key is populated for
# BOTH pick_place and sort episodes (the instruction / the sort goal); the
# subtask key only where a decomposition exists (sort), or as an identical
# copy of the task instruction when ``subtask_copy`` is set (eva regime).
KEY_TASK = "annotations_task"
KEY_SUBTASK = "annotations_subtask"


def micro_seconds_to_frames(micro_seconds: int, fps: int) -> int:
    return micro_seconds / 1000000 * fps


class ScaleToZarrAnnotationConverter:
    def __init__(self, scale_annotation_dir: str):
        self.scale_annotation_dir = scale_annotation_dir

    def convert(self, tid: str) -> dict[str, list]:
        """Returns ``{annotation_key_name: [(text, start_idx, end_idx), ...]}``
        — one entry list per role-named zarr key."""
        scale_annotation_path = os.path.join(self.scale_annotation_dir, f"{tid}.json")
        with open(scale_annotation_path, "r") as f:
            annotation_dict = json.load(f)
        return self.scale_to_str_format(annotation_dict)

    @abstractmethod
    def scale_to_str_format(self, annotation_dict: dict) -> dict[str, list]:
        pass


class LLMConverter(ScaleToZarrAnnotationConverter):
    def __init__(self, scale_annotation_dir: str, prompt_filepath: str):
        super().__init__(scale_annotation_dir)
        self.client = OpenAI()
        self.model = "gpt-5.4"
        self.fps = 30
        with open(prompt_filepath, "r") as f:
            self.prompt_template = f.read()


class PickPlaceLLMConverter(LLMConverter):
    def __init__(
        self,
        scale_annotation_dir: str,
        prompt_filepath: str,
        augment_prompt_filepath: str | None = None,
        subtask_copy: bool = False,
    ):
        super().__init__(scale_annotation_dir, prompt_filepath)
        if augment_prompt_filepath is not None:
            with open(augment_prompt_filepath, "r") as f:
                self.augment_prompt_template = f.read()
        else:
            self.augment_prompt_template = None
        # When True, the task instructions are ALSO written verbatim under the
        # subtask key (eva regime: identical task prompt and subtask target).
        self.subtask_copy = subtask_copy

    def scale_to_str_format(self, annotation_dict: dict) -> dict[str, list]:
        # Pick-and-place instructions ARE the task-level conditioning for
        # non-sort episodes — written under the task key (no read-time
        # fallback; a missing task never borrows the subtask text).
        task_entries = []
        for prompt_dict, start_idx, end_idx in self._iter_pick_place_clips(
            annotation_dict
        ):
            base_instruction = self.scale_annotation_to_str(prompt_dict)
            instructions = self.augment_instruction(base_instruction, prompt_dict)
            for instruction in instructions:
                task_entries.append((instruction, start_idx, end_idx))
        keyed = {KEY_TASK: task_entries}
        if self.subtask_copy:
            keyed[KEY_SUBTASK] = list(task_entries)
        return keyed

    def _iter_pick_place_clips(self, annotation_dict: dict):
        """Yield ``(prompt_dict, start_idx, end_idx)`` for every valid
        pick-and-place clip.

        Encapsulates the shared parsing/filtering — derive the arm from the
        actuator label, convert the clip's microsecond span to frames, drop
        clips flagged as a mistake and "Adjust" (or action-less) clips — so
        subclasses such as :class:`SortConverter` reuse the exact same
        low-level extraction instead of re-deriving it.
        """
        for annotation in annotation_dict["annotations"]:
            if "label" not in annotation:
                continue
            arm = annotation["label"].split(" ")[0].lower()
            for clip in annotation["clips"]:
                if "attributes" not in clip:
                    # High-level tracks (e.g. the sort "Sorting" track) carry a
                    # bare text clip with no action attributes — not a
                    # pick-and-place motion, so skip it here.
                    continue
                timestamp = micro_seconds_to_frames(clip["timestamp"], self.fps)
                duration = micro_seconds_to_frames(clip["duration"], self.fps)
                attr_dict = {}
                for attribute in clip["attributes"]:
                    attr_dict[attribute["name"]] = attribute["values"][0]
                if attr_dict.get("Mistake") == "Yes":
                    continue
                if "Action" not in attr_dict or attr_dict["Action"] == "Adjust":
                    continue
                prompt_dict = attr_dict.copy()
                prompt_dict.pop("Mistake", None)
                prompt_dict["description"] = clip["text"]
                prompt_dict["arm"] = arm
                yield prompt_dict, timestamp, timestamp + duration

    def scale_annotation_to_str(self, scale_annotation_dict: dict) -> str:
        model_prompt = self.prompt_template + "\n" + json.dumps(scale_annotation_dict)
        response = self.client.responses.create(model=self.model, input=model_prompt)
        return response.output_text

    def augment_instruction(
        self, base_instruction: str, scale_annotation_dict: dict
    ) -> list[str]:
        """Produce language-augmented variants of ``base_instruction``.

        The returned list always contains the original ``base_instruction``
        plus, when an augmentation prompt is configured, LLM-generated
        synonyms and variants that omit arm and place-orientation info.
        """
        return self._augment_with_template(
            base_instruction, scale_annotation_dict, self.augment_prompt_template
        )

    def _augment_with_template(
        self, base_instruction: str, metadata: dict, template: str | None
    ) -> list[str]:
        """Shared augmentation: ask the LLM for a JSON array of variants and
        return ``[base_instruction, *unique_variants]``. Returns just
        ``[base_instruction]`` when no augmentation template is configured."""
        if template is None:
            return [base_instruction]

        model_prompt = (
            template
            + "\n"
            + json.dumps(
                {
                    "instruction": base_instruction,
                    "metadata": metadata,
                }
            )
        )
        response = self.client.responses.create(model=self.model, input=model_prompt)
        variants: list[str] = []
        try:
            parsed = json.loads(response.output_text)
            if isinstance(parsed, list):
                variants = [v for v in parsed if isinstance(v, str) and v.strip()]
        except json.JSONDecodeError:
            pass

        deduped: list[str] = []
        seen: set[str] = set()
        for v in [base_instruction, *variants]:
            if v not in seen:
                deduped.append(v)
                seen.add(v)
        return deduped


class SortConverter(PickPlaceLLMConverter):
    """Convert Scale annotations for *sort* tasks into Zarr annotation tuples.

    A sort episode's Scale annotation has separate label tracks:

      * ``Left Gripper`` / ``Right Gripper`` — the *low-level* pick-and-place
        clips (with Action/Mistake/… attributes), handled exactly like
        :class:`PickPlaceLLMConverter`.
      * ``Sorting`` — the *high-level* sort goals, written by the annotators as
        plain ``text`` clips (no attributes), each spanning the window of the
        sort sub-goal it describes (e.g. "Sort the corn and the croissant on
        the white plate").
      * ``Both Grippers`` — return-to-home transitions (empty ``text``). Kept
        as low-level clips too, mirroring :class:`PickPlaceLLMConverter` (which
        annotates "Return to home" from the Action even with empty text); each
        pairs with the nearest sort goal since they fall between sort windows.

    High-level instructions are therefore *read* from the ``Sorting`` track
    rather than generated. Each low-level pick-and-place clip is paired with the
    sort instruction active during it (max temporal overlap); both sides are
    augmented and truncated to a common count so that **every low-level span
    carries an equal number of high-level sort and low-level pick-and-place
    instructions** — i.e. the two granularities are balanced at all times
    (frames with no pick-and-place clip trivially carry 0 == 0).

    If an annotation has no ``Sorting`` track and ``sort_prompt_filepath`` is
    configured, the converter falls back to LLM-generating a single high-level
    instruction from the pick-and-place steps.
    """

    #: Annotation labels (case-insensitive prefix) that hold the high-level
    #: sort-goal text track.
    SORT_LABEL_PREFIX = "sort"

    def __init__(
        self,
        scale_annotation_dir: str,
        prompt_filepath: str,
        sort_prompt_filepath: str | None = None,
        augment_prompt_filepath: str | None = None,
        sort_augment_prompt_filepath: str | None = None,
    ):
        super().__init__(
            scale_annotation_dir,
            prompt_filepath,
            augment_prompt_filepath=augment_prompt_filepath,
        )
        # Optional LLM-generation prompt — only used as a fallback when the
        # annotation has no human-written "Sorting" track.
        self.sort_prompt_template = self._read_template(sort_prompt_filepath)
        # Augmentation prompt for the high-level sort instructions.
        self.sort_augment_prompt_template = self._read_template(
            sort_augment_prompt_filepath
        )

    @staticmethod
    def _read_template(path: str | None) -> str | None:
        if path is None:
            return None
        with open(path, "r") as f:
            return f.read()

    def scale_to_str_format(self, annotation_dict: dict) -> dict[str, list]:
        # Subtask level: every gripper pick-and-place clip, exactly like
        # PickPlaceLLMConverter (only Mistake/Adjust clips are filtered, inside
        # _iter_pick_place_clips). This keeps "Return to home" transition clips,
        # which pick_place also annotates — the LLM phrases them from the Action
        # even though their text is empty.
        low_clips = list(self._iter_pick_place_clips(annotation_dict))
        if not low_clips:
            return {}

        # Task level: read the human-written "Sorting" track. Fall back to LLM
        # generation only when no such track exists.
        sort_intervals = self._iter_sort_clips(annotation_dict)
        fallback_pool: list[str] | None = None
        if not sort_intervals:
            if self.sort_prompt_template is None:
                return {}
            context = self.build_sort_context(low_clips)
            fallback_pool = self._nonblank(
                self.augment_sort_instruction(
                    self.sort_annotation_to_str(context), context
                )
            )
            if not fallback_pool:
                return {}

        # Augment each distinct sort instruction at most once.
        aug_cache: dict[str, list[str]] = {}

        def high_pool_for(text: str) -> list[str]:
            if text not in aug_cache:
                aug_cache[text] = self._nonblank(
                    self.augment_sort_instruction(text, {"task": "sort"})
                )
            return aug_cache[text]

        task_entries = []
        subtask_entries = []
        for i, (prompt_dict, start_idx, end_idx) in enumerate(low_clips):
            if sort_intervals:
                high_text = self._sort_text_for_span(start_idx, end_idx, sort_intervals)
                high_pool = high_pool_for(high_text) if high_text else []
            else:
                high_pool = fallback_pool

            low_base = self.scale_annotation_to_str(prompt_dict)
            low_pool = self._nonblank(self.augment_instruction(low_base, prompt_dict))
            if not low_pool or not high_pool:
                continue

            # Rotate the high-level pool so successive spans pair with different
            # phrasings, then truncate both sides to equal length so the span
            # carries the same count of each granularity.
            offset = i % len(high_pool)
            high_rotated = high_pool[offset:] + high_pool[:offset]
            low_balanced, high_balanced = self.balance_instructions(
                low_pool, high_rotated
            )
            # Role in the KEY NAME: the sort goal conditions the model
            # (task key), the pick-and-place phrasing is the prediction
            # target (subtask key). Same span on both sides so a frame's two
            # candidate lists stay pairable.
            for instruction in low_balanced:
                subtask_entries.append((instruction, start_idx, end_idx))
            for instruction in high_balanced:
                task_entries.append((instruction, start_idx, end_idx))
        if not task_entries and not subtask_entries:
            return {}
        return {KEY_TASK: task_entries, KEY_SUBTASK: subtask_entries}

    def _iter_sort_clips(self, annotation_dict: dict) -> list[tuple[str, float, float]]:
        """Return ``(text, start_idx, end_idx)`` for the high-level "Sorting"
        track — annotations whose label starts with "sort", whose clips carry
        the sort goal as ``text`` (and have no action attributes)."""
        intervals: list[tuple[str, float, float]] = []
        for annotation in annotation_dict["annotations"]:
            label = annotation.get("label", "") or ""
            if not label.strip().lower().startswith(self.SORT_LABEL_PREFIX):
                continue
            for clip in annotation["clips"]:
                text = (clip.get("text") or "").strip()
                if not text:
                    continue
                start_idx = micro_seconds_to_frames(clip["timestamp"], self.fps)
                end_idx = start_idx + micro_seconds_to_frames(
                    clip["duration"], self.fps
                )
                intervals.append((text, start_idx, end_idx))
        return intervals

    @staticmethod
    def _sort_text_for_span(
        start_idx: float,
        end_idx: float,
        sort_intervals: list[tuple[str, float, float]],
    ) -> str | None:
        """Return the sort instruction active during the clip span
        ``[start_idx, end_idx]``: the one with the most temporal overlap, or —
        for a clip that falls in a gap between sort goals (e.g. a return-to-home
        transition) — the temporally nearest one. ``None`` only when there are
        no sort intervals at all."""
        best_text, best_overlap = None, 0.0
        for text, s, e in sort_intervals:
            overlap = max(0.0, min(end_idx, e) - max(start_idx, s))
            if overlap > best_overlap:
                best_overlap, best_text = overlap, text
        if best_text is not None:
            return best_text
        # No overlap: fall back to the temporally nearest sort instruction.
        nearest_text, nearest_dist = None, None
        for text, s, e in sort_intervals:
            dist = max(s - end_idx, start_idx - e, 0.0)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist, nearest_text = dist, text
        return nearest_text

    @staticmethod
    def _nonblank(instructions: list[str]) -> list[str]:
        """Drop ``None``/blank instructions so an empty LLM response never
        becomes an empty-text annotation (keeps per-span counts meaningful)."""
        return [s for s in instructions if s and s.strip()]

    def build_sort_context(self, clips: list) -> dict:
        """Summarize the episode's pick-and-place sub-actions into a grounded
        context for the LLM-generation *fallback* (used only when there is no
        human-written "Sorting" track).

        The arm/hand is intentionally omitted: a high-level sort goal is
        arm-agnostic and both sort prompts forbid mentioning a hand.

        Args:
            clips: ``(prompt_dict, start_idx, end_idx)`` tuples, in episode order.
        """
        steps = []
        for prompt_dict, _, _ in clips:
            steps.append(
                {
                    "Action": prompt_dict.get("Action"),
                    "description": prompt_dict.get("description"),
                }
            )
        return {
            "task": "sort",
            "steps": steps,
        }

    def sort_annotation_to_str(self, sort_context: dict) -> str:
        """Generate one high-level sort instruction from the episode context
        (fallback path; requires ``sort_prompt_filepath``)."""
        model_prompt = self.sort_prompt_template + "\n" + json.dumps(sort_context)
        response = self.client.responses.create(model=self.model, input=model_prompt)
        return response.output_text

    def augment_sort_instruction(
        self, base_instruction: str, sort_context: dict
    ) -> list[str]:
        """High-level analogue of :meth:`augment_instruction` using the sort
        augmentation prompt. Always includes ``base_instruction`` first."""
        return self._augment_with_template(
            base_instruction, sort_context, self.sort_augment_prompt_template
        )

    @staticmethod
    def balance_instructions(
        low_list: list[str], high_list: list[str]
    ) -> tuple[list[str], list[str]]:
        """Truncate both lists to a common length so a span carries an equal
        number of low-level and high-level instructions."""
        k = min(len(low_list), len(high_list))
        return low_list[:k], high_list[:k]


class HardCodedConverter(ScaleToZarrAnnotationConverter):
    def scale_to_str_format(self, annotation: dict) -> dict:
        pass
