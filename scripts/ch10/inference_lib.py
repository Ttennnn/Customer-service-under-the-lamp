"""ONNX Runtime classifier used by the side-channel service (no torch/transformers)."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.taxonomy import TOPIC_CLASSES, labels_from_scores


class Classifier:
    def __init__(
        self,
        model_path,
        tokenizer_path,
        threshold_path,
        max_length: int = 128,
    ) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.np = np
        tokenizer_path = Path(tokenizer_path)
        tokenizer_file = (
            tokenizer_path / "tokenizer.json"
            if tokenizer_path.is_dir()
            else tokenizer_path
        )
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(
            pad_id=pad_id if pad_id is not None else 0,
            pad_token="[PAD]",
        )

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        threshold_data = json.loads(
            Path(threshold_path).read_text(encoding="utf-8")
        )
        self.threshold = float(threshold_data["threshold"])
        self.model_name = Path(model_path).name

    def _feed(self, encodings) -> dict:
        feed = {}
        for name in self.input_names:
            if name == "input_ids":
                values = [encoding.ids for encoding in encodings]
            elif name == "attention_mask":
                values = [encoding.attention_mask for encoding in encodings]
            elif name == "token_type_ids":
                values = [encoding.type_ids for encoding in encodings]
            else:
                continue
            feed[name] = self.np.array(values, dtype=self.np.int64)
        return feed

    def classify(self, texts: list[str]) -> list[dict]:
        encodings = self.tokenizer.encode_batch(texts)
        logits = self.session.run(None, self._feed(encodings))[0]
        probs = 1.0 / (1.0 + self.np.exp(-logits))
        results: list[dict] = []
        for row in probs:
            scores = {
                label: float(row[index])
                for index, label in enumerate(TOPIC_CLASSES)
            }
            results.append(
                {
                    "labels": labels_from_scores(scores, self.threshold),
                    "scores": scores,
                }
            )
        return results
