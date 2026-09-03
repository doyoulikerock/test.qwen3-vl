import gc
from typing import NamedTuple

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from . import config


class Answer(NamedTuple):
    """A generated answer plus whether generation stopped on the max_new_tokens budget
    instead of on an end-of-turn token — i.e. whether `text` is cut off mid-sentence.
    Callers surface that, because a silently clipped answer reads like a real one."""

    text: str
    truncated: bool


class Explainer:
    """Loads Qwen3-VL-4B-Instruct on demand for open-ended visual questions over a handful
    of frames (e.g. "how many people are visible?") — must be released before loading the
    embedder or reranker; none of Qwen3-VL's three checkpoints (2B/2B/4B) fit in VRAM
    together (§Context, §5 of PLAN.md)."""

    def __init__(self, model_id: str = config.EXPLAIN_MODEL_ID, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(self.device)

    def release(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ask(
        self,
        prompt: str,
        image_paths: list[str],
        max_new_tokens: int = config.DEFAULT_EXPLAIN_MAX_NEW_TOKENS,
    ) -> Answer:
        """Send all image_paths together as one multi-image turn and return the model's
        text answer. Sharing one turn (rather than asking per-frame) lets the model reason
        across frames, e.g. "the highest count across these is N"."""
        self.load()
        content = [
            {
                "type": "image",
                "image": path,
                "min_pixels": config.MIN_PIXELS,
                "max_pixels": config.MAX_PIXELS,
            }
            for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return Answer(output_text[0].strip(), self._hit_budget(trimmed[0], max_new_tokens))

    def _hit_budget(self, generated, max_new_tokens: int) -> bool:
        """True when generation ran out of budget rather than ending its turn: the full
        budget was spent and the last token is not one of the model's stop tokens (the
        decoded text drops those, so length alone cannot tell the two cases apart)."""
        if len(generated) < max_new_tokens:
            return False
        eos = self._model.generation_config.eos_token_id
        stops = {eos} if isinstance(eos, int) else set(eos or ())
        return int(generated[-1]) not in stops
