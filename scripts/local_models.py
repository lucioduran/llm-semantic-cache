"""Explicit free downloads and offline CPU model execution."""

from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def count_tokens(self, text: str) -> int: ...
    def generate(self, prompt: str) -> dict: ...


def model_path(cache: Path, name: str, revision: str) -> Path:
    return cache / name.replace("/", "--") / revision


def download_models(config: dict, cache: Path) -> None:
    from huggingface_hub import snapshot_download

    for field in ("embedding_model", "answer_model"):
        model = config[field]
        snapshot_download(
            repo_id=model["id"],
            revision=model["revision"],
            local_dir=model_path(cache, model["id"], model["revision"]),
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.txt",
                "spiece.model",
                "model.safetensors",
            ],
            token=False,
        )


class LocalModels:
    def __init__(self, config: dict, cache: Path):
        import torch
        from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

        torch.set_num_threads(config["threads"])
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        self.torch, self.config = torch, config
        paths = {
            field: model_path(cache, config[field]["id"], config[field]["revision"])
            for field in ("embedding_model", "answer_model")
        }
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            paths["embedding_model"], local_files_only=True, trust_remote_code=False
        )
        self.answer_tokenizer = AutoTokenizer.from_pretrained(
            paths["answer_model"], local_files_only=True, trust_remote_code=False
        )
        self.embedding_model = AutoModel.from_pretrained(
            paths["embedding_model"],
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).eval()
        self.answer_model = AutoModelForSeq2SeqLM.from_pretrained(
            paths["answer_model"],
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        ).eval()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self.embedding_tokenizer(
            texts, padding=True, truncation=False, return_tensors="pt"
        )
        if encoded["input_ids"].shape[1] > self.config["embedding_max_tokens"]:
            raise ValueError("Embedding input exceeds the frozen token budget")
        with self.torch.inference_mode():
            hidden = self.embedding_model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.torch.nn.functional.normalize(pooled, p=2, dim=1).tolist()

    def count_tokens(self, text: str) -> int:
        return len(self.answer_tokenizer.encode(text, add_special_tokens=True))

    def generate(self, prompt: str) -> dict:
        encoded = self.answer_tokenizer(prompt, return_tensors="pt", truncation=False)
        count = encoded["input_ids"].shape[1]
        if count > self.config["max_input_tokens"]:
            raise ValueError("Generation input exceeds the frozen token budget")
        with self.torch.inference_mode():
            result = self.answer_model.generate(
                **encoded,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=False,
                num_beams=1,
            )
        tokens = result[0].tolist()[1:]
        return {
            "answer": self.answer_tokenizer.decode(result[0], skip_special_tokens=True),
            "input_tokens": count,
            "output_tokens": len(tokens),
            "reached_token_limit": len(tokens) == self.config["max_new_tokens"]
            and tokens[-1] != self.answer_tokenizer.eos_token_id,
        }
