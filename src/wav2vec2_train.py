import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np

from datasets import load_dataset
from datasets.features import Audio as HF_Audio

import torch
from torch import nn

from transformers import (
	AutoProcessor,
	Wav2Vec2ForSequenceClassification,
	Trainer,
	TrainingArguments,
)


NUM_LABELS = 10
TARGET_SR = 8000
DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"


def prepare_datasets(noise_prob: float = 0.0, noise_bg_dir: str | None = None):
	ds_train = load_dataset("mteb/free-spoken-digit-dataset", split="train")
	ds_test = load_dataset("mteb/free-spoken-digit-dataset", split="test")
	# Decode to TARGET_SR
	ds_train = ds_train.cast_column("audio", HF_Audio(decode=True, sampling_rate=TARGET_SR))
	ds_test = ds_test.cast_column("audio", HF_Audio(decode=True, sampling_rate=TARGET_SR))

	# Simple waveform augmentation (optional)
	augmenter = None
	if noise_prob and noise_prob > 0.0:
		try:
			from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise
			augs = [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=noise_prob)]
			if noise_bg_dir and os.path.isdir(noise_bg_dir):
				augs.append(AddBackgroundNoise(sounds_path=noise_bg_dir, p=min(0.3, noise_prob)))
			augmenter = Compose(augs)
		except Exception:
			augmenter = None

	def map_train(ex):
		arr = ex["audio"]["array"]
		if augmenter is not None:
			try:
				arr = augmenter(samples=arr, sample_rate=TARGET_SR)
			except Exception:
				pass
		ex["input_values"] = arr
		return ex

	def map_test(ex):
		ex["input_values"] = ex["audio"]["array"]
		return ex

	ds_train = ds_train.map(map_train, remove_columns=["audio"], num_proc=1)
	ds_test = ds_test.map(map_test, remove_columns=["audio"], num_proc=1)
	return ds_train, ds_test


@dataclass
class DataCollatorAudio:
	processor: Any
	def __call__(self, features: List[Dict[str, Any]]):
		audios = [f["input_values"] for f in features]
		batch = self.processor(audios, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
		labels = torch.tensor([int(f["label"]) for f in features], dtype=torch.long)
		batch["labels"] = labels
		return batch


def compute_metrics(eval_pred):
	logits, labels = eval_pred
	preds = np.argmax(logits, axis=-1)
	acc = (preds == labels).mean().item() if hasattr(acc := (preds == labels).mean(), 'item') else float(acc)
	return {"accuracy": acc}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
	parser.add_argument("--output_dir", type=str, default="models/wav2vec2_digits_8k")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch_size", type=int, default=16)
	parser.add_argument("--lr", type=float, default=5e-5)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--freeze_feature_extractor", action="store_true")
	parser.add_argument("--noise_prob", type=float, default=0.2)
	parser.add_argument("--noise_bg_dir", type=str, default="")
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)

	processor = AutoProcessor.from_pretrained(args.model_name)
	# Set processor sampling rate to 8k for inference-time defaults
	if hasattr(processor, "feature_extractor"):
		try:
			processor.feature_extractor.sampling_rate = TARGET_SR
		except Exception:
			pass
	model = Wav2Vec2ForSequenceClassification.from_pretrained(
		args.model_name,
		num_labels=NUM_LABELS,
		label2id={str(i): i for i in range(NUM_LABELS)},
		id2label={i: str(i) for i in range(NUM_LABELS)},
	)
	if args.freeze_feature_extractor and hasattr(model, "freeze_feature_encoder"):
		model.freeze_feature_encoder()

	ds_train, ds_test = prepare_datasets(noise_prob=args.noise_prob, noise_bg_dir=args.noise_bg_dir or None)
	collator = DataCollatorAudio(processor)

	args_train = TrainingArguments(
		output_dir=args.output_dir,
		eval_strategy="epoch",
		logging_strategy="steps",
		logging_steps=25,
		save_strategy="epoch",
		learning_rate=args.lr,
		weight_decay=args.weight_decay,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		num_train_epochs=args.epochs,
		gradient_accumulation_steps=1,
		fp16=False,
		load_best_model_at_end=True,
		metric_for_best_model="accuracy",
	)

	trainer = Trainer(
		model=model,
		args=args_train,
		train_dataset=ds_train,
		eval_dataset=ds_test,
		data_collator=collator,
		compute_metrics=compute_metrics,
	)

	trainer.train()
	metrics = trainer.evaluate()
	print("Final eval:", metrics)

	# Save artifacts
	trainer.save_model(args.output_dir)
	processor.save_pretrained(args.output_dir)
	print(f"Saved Wav2Vec2 model to: {args.output_dir}")


if __name__ == "__main__":
	main() 