import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import argparse
import queue
import numpy as np
import sounddevice as sd
import joblib
import time
import sys
import threading
import termios
import tty
import select
import librosa
import torch

from src.data_processing import TARGET_SAMPLE_RATE, preprocess_audio_to_features


def _pool(feat, mode: str):
	if mode == "flat":
		return feat.reshape(1, -1)
	if mode == "mean":
		return feat.mean(axis=1, keepdims=False)[np.newaxis, :]
	return feat.max(axis=1, keepdims=False)[np.newaxis, :]


def _rms_energy(x: np.ndarray) -> float:
	return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)))


def _majority_vote(preds: list[int]) -> int:
	if not preds:
		return -1
	vals, counts = np.unique(np.asarray(preds), return_counts=True)
	return int(vals[np.argmax(counts)])


def _toggle_thread(state: dict, key: str):
	old_attrs = termios.tcgetattr(sys.stdin)
	try:
		tty.setcbreak(sys.stdin.fileno())
		while not state.get("stop", False):
			if select.select([sys.stdin], [], [], 0.05)[0]:
				c = sys.stdin.read(1)
				if c == key:
					state["listening"] = not state.get("listening", False)
					print(f"[toggle] mic {'ON' if state['listening'] else 'OFF'}")
	finally:
		termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_attrs)


def _apply_temp(probs: np.ndarray, temp: float) -> np.ndarray:
	if temp is None or temp == 1.0:
		return probs
	p = np.clip(probs.astype(np.float64), 1e-8, 1.0)
	alpha = 1.0 / max(1e-6, temp)
	p = p ** alpha
	p = p / np.sum(p, axis=-1, keepdims=True)
	return p.astype(np.float32)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="models/baseline_svm.joblib")
	parser.add_argument("--duration", type=float, default=0.8, help="Seconds per prediction window")
	parser.add_argument("--stride", type=float, default=0.4, help="Seconds to slide between windows")
	parser.add_argument("--max_len", type=int, default=200)
	parser.add_argument("--n_fft", type=int, default=512)
	parser.add_argument("--hop_length", type=int, default=128)
	parser.add_argument("--add_deltas", action="store_true")
	parser.add_argument("--pool", choices=["flat", "mean", "max"], default="flat")
	# VAD and gating
	parser.add_argument("--vad", action="store_true", help="Enable simple RMS-based VAD gating")
	parser.add_argument("--vad_thresh", type=float, default=0.01, help="RMS threshold for speech activity (0-1)")
	parser.add_argument("--min_speech_ms", type=int, default=250, help="Minimum duration to consider a speech segment")
	parser.add_argument("--hangover_ms", type=int, default=300, help="Keep segment active this long after energy drops")
	parser.add_argument("--conf_thresh", type=float, default=0.6, help="Min predicted probability to emit")
	parser.add_argument("--temp", type=float, default=1.0, help="Softmax temperature (<1 sharpens, >1 flattens)")
	parser.add_argument("--force_emit", action="store_true", help="Emit top-1 even if below conf_thresh at segment end")
	# PTT and toggle
	parser.add_argument("--ptt", action="store_true", help="Push-to-talk mode: press Enter to record one window and predict")
	parser.add_argument("--toggle", action="store_true", help="Toggle mode: press key to start/stop streaming predictions")
	parser.add_argument("--toggle_key", type=str, default="t", help="Single key to toggle mic (default: 't')")
	# Capture rate
	parser.add_argument("--capture_sr", type=int, default=8000, help="Mic capture sample rate (resampled to feature rate)")
	# Architecture: baseline MFCC+SVM, Keras small_cnn, or wav2vec2 transformer
	parser.add_argument("--arch", choices=["baseline", "keras", "wav2vec2"], default="baseline")
	parser.add_argument("--w2v_dir", type=str, default="models/wav2vec2_digits")
	parser.add_argument("--keras_model", type=str, default="models/small_cnn.keras")
	parser.add_argument("--continuous", action="store_true", help="Emit predictions continuously (default: emit once per speech segment)")
	args = parser.parse_args()

	if args.arch == "baseline" and not os.path.isfile(args.model):
		raise FileNotFoundError(f"Model not found: {args.model}")

	pipe = None
	proba_supported = False
	if args.arch == "baseline":
		pipe = joblib.load(args.model)
		proba_supported = hasattr(pipe, "predict_proba")
	elif args.arch == "keras":
		import keras
		from keras import ops
		if not os.path.isfile(args.keras_model):
			raise FileNotFoundError(f"Keras model not found: {args.keras_model}")
		kmodel = keras.saving.load_model(args.keras_model)
	else:
		from transformers import AutoProcessor, Wav2Vec2ForSequenceClassification
		from pathlib import Path
		if not Path(args.w2v_dir).exists():
			raise FileNotFoundError(f"Wav2Vec2 model dir not found: {args.w2v_dir}")
		w2v_processor = AutoProcessor.from_pretrained(args.w2v_dir)
		w2v_model = Wav2Vec2ForSequenceClassification.from_pretrained(args.w2v_dir)
		w2v_model.eval()
		w2v_sr = int(getattr(getattr(w2v_processor, "feature_extractor", {}), "sampling_rate", 16000))

	if args.ptt:
		print(f"Push-to-talk active. Recording {args.duration:.2f}s windows at {args.capture_sr} Hz. Press Enter to capture, Ctrl+C to exit.")
		try:
			while True:
				input()
				rec = sd.rec(int(args.capture_sr * args.duration), samplerate=args.capture_sr, channels=1, dtype='float32')
				sd.wait()
				mono = rec.squeeze(-1).astype(np.float32)
				if args.arch == "wav2vec2":
					if args.capture_sr != w2v_sr:
						mono = librosa.resample(mono, orig_sr=args.capture_sr, target_sr=w2v_sr).astype(np.float32)
					with torch.no_grad():
						inputs = w2v_processor(mono, sampling_rate=w2v_sr, return_tensors="pt", padding=True)
						out = w2v_model(**inputs)
						probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
						probs = _apply_temp(probs, args.temp)
						pred = int(np.argmax(probs))
						conf = float(np.max(probs))
						if conf < args.conf_thresh and not args.force_emit:
							print(f"Low confidence ({conf:.2f}); ignoring")
							continue
						print(f"Prediction: {pred} (conf={conf:.2f})")
				elif args.arch == "keras":
					if args.capture_sr != TARGET_SAMPLE_RATE:
						mono = librosa.resample(mono, orig_sr=args.capture_sr, target_sr=TARGET_SAMPLE_RATE).astype(np.float32)
					feat = preprocess_audio_to_features(mono, TARGET_SAMPLE_RATE, max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False)
					x = feat[np.newaxis, ..., np.newaxis].astype(np.float32)
					logits = kmodel(x, training=False)
					from keras import ops
					probs_t = ops.softmax(logits, axis=-1)
					try:
						probs = probs_t.numpy()[0]
					except Exception:
						probs = probs_t.detach().cpu().numpy()[0]
					probs = _apply_temp(probs, args.temp)
					pred = int(np.argmax(probs))
					conf = float(np.max(probs))
					if conf < args.conf_thresh and not args.force_emit:
						print(f"Low confidence ({conf:.2f}); ignoring")
						continue
					print(f"Prediction: {pred} (conf={conf:.2f})")
				else:
					if args.capture_sr != TARGET_SAMPLE_RATE:
						mono = librosa.resample(mono, orig_sr=args.capture_sr, target_sr=TARGET_SAMPLE_RATE).astype(np.float32)
					feat = preprocess_audio_to_features(mono, TARGET_SAMPLE_RATE, max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False)
					x = _pool(feat, args.pool)
					if proba_supported:
						probs = pipe.predict_proba(x)[0]
						probs = _apply_temp(probs, args.temp)
						pred = int(np.argmax(probs))
						conf = float(np.max(probs))
						if conf < args.conf_thresh and not args.force_emit:
							print(f"Low confidence ({conf:.2f}); ignoring")
							continue
						print(f"Prediction: {pred} (conf={conf:.2f})")
					else:
						pred = int(pipe.predict(x)[0])
						print(f"Prediction: {pred}")
		except KeyboardInterrupt:
			print("Stopped.")
		return

	# Streaming mode with VAD and optional toggle
	q: queue.Queue[np.ndarray] = queue.Queue()
	block_size_cap = int(args.capture_sr * args.stride)
	window_size_cap = int(args.capture_sr * args.duration)
	feat_sr = TARGET_SAMPLE_RATE
	if args.arch == "wav2vec2":
		try:
			feat_sr = int(getattr(getattr(w2v_processor, "feature_extractor", {}), "sampling_rate", 16000))
		except Exception:
			feat_sr = 16000
	window_size = int(feat_sr * args.duration)
	buffer = np.zeros(window_size, dtype=np.float32)

	min_speech_samples = int((args.min_speech_ms / 1000.0) * feat_sr)
	hangover_samples = int((args.hangover_ms / 1000.0) * feat_sr)
	speech_active = False
	speech_accum_samples = 0
	hangover_left = 0
	# Segment accumulators
	prob_sum = None  # type: ignore[var-annotated]
	frame_count = 0
	votes: list[int] = []

	state = {"listening": not args.toggle, "stop": False}
	toggle_thread = None
	if args.toggle:
		print(f"Toggle mode enabled. Press '{args.toggle_key}' to start/stop predictions.")
		toggle_thread = threading.Thread(target=_toggle_thread, args=(state, args.toggle_key), daemon=True)
		toggle_thread.start()

	def audio_callback(indata, frames, time_info, status):  # pylint: disable=unused-argument
		if status:
			print(status)
		mono = indata.mean(axis=1).astype(np.float32, copy=False)
		if args.capture_sr != feat_sr:
			mono = librosa.resample(mono, orig_sr=args.capture_sr, target_sr=feat_sr).astype(np.float32)
		q.put(mono)

	print(f"Listening at {args.capture_sr} Hz (resampling to {feat_sr} Hz)... Press Ctrl+C to stop.")
	with sd.InputStream(channels=1, samplerate=args.capture_sr, callback=audio_callback, blocksize=block_size_cap):
		try:
			while True:
				while not q.empty():
					chunk = q.get()
					shift = len(chunk)
					if shift >= window_size:
						buffer[:] = chunk[-window_size:]
					else:
						buffer[:-shift] = buffer[shift:]
						buffer[-shift:] = chunk

				if not state.get("listening", True):
					sd.sleep(int(args.stride * 1000))
					continue

				energy = _rms_energy(buffer)
				is_speech = (energy >= args.vad_thresh) if args.vad else True

				if is_speech:
					hangover_left = hangover_samples
					speech_accum_samples += int(feat_sr * args.stride)
					speech_active = True
					if args.arch == "wav2vec2":
						from transformers import AutoProcessor
						with torch.no_grad():
							inputs = w2v_processor(buffer, sampling_rate=feat_sr, return_tensors="pt", padding=True)
							out = w2v_model(**inputs)
							probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
							probs = _apply_temp(probs, args.temp)
							pred = int(np.argmax(probs))
							conf = float(np.max(probs))
							if args.continuous and (conf >= args.conf_thresh or args.force_emit):
								print(f"Prediction: {pred} (conf={conf:.2f})")
							# accumulate for segment-level emission
							if prob_sum is None:
								prob_sum = np.zeros_like(probs)
							prob_sum += probs
							frame_count += 1
					elif args.arch == "keras":
						feat = preprocess_audio_to_features(buffer, TARGET_SAMPLE_RATE, max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False)
						x = feat[np.newaxis, ..., np.newaxis].astype(np.float32)
						logits = kmodel(x, training=False)
						from keras import ops
						probs_t = ops.softmax(logits, axis=-1)
						try:
							probs = probs_t.numpy()[0]
						except Exception:
							probs = probs_t.detach().cpu().numpy()[0]
						probs = _apply_temp(probs, args.temp)
						pred = int(np.argmax(probs))
						conf = float(np.max(probs))
						if args.continuous and (conf >= args.conf_thresh or args.force_emit):
							print(f"Prediction: {pred} (conf={conf:.2f})")
						if prob_sum is None:
							prob_sum = np.zeros_like(probs)
						prob_sum += probs
						frame_count += 1
					else:
						feat = preprocess_audio_to_features(buffer, TARGET_SAMPLE_RATE, max_len=args.max_len, n_fft=args.n_fft, hop_length=args.hop_length, add_deltas=args.add_deltas, apply_augmentation=False)
						x = _pool(feat, args.pool)
						if proba_supported:
							probs = pipe.predict_proba(x)[0]
							probs = _apply_temp(probs, args.temp)
							pred = int(np.argmax(probs))
							conf = float(np.max(probs))
							if args.continuous and (conf >= args.conf_thresh or args.force_emit):
								print(f"Prediction: {pred} (conf={conf:.2f})")
							if prob_sum is None:
								prob_sum = np.zeros_like(probs)
							prob_sum += probs
							frame_count += 1
						else:
							pred = int(pipe.predict(x)[0])
							if args.continuous:
								print(f"Prediction: {pred}")
							votes.append(pred)
				else:
					if hangover_left > 0:
						hangover_left -= int(feat_sr * args.stride)
					else:
						# Reset segment state when silence persists
						if speech_active and speech_accum_samples >= min_speech_samples:
							# finalize one prediction for the segment
							if prob_sum is not None and frame_count > 0:
								avg_probs = prob_sum / max(1, frame_count)
								avg_probs = _apply_temp(avg_probs, args.temp)
								pred = int(np.argmax(avg_probs))
								conf = float(np.max(avg_probs))
								if conf >= args.conf_thresh or args.force_emit:
									print(f"Prediction: {pred} (conf={conf:.2f})")
							elif votes:
								vals, counts = np.unique(np.asarray(votes), return_counts=True)
								final_pred = int(vals[np.argmax(counts)])
								print(f"Prediction: {final_pred}")
						# clear segment accumulators
						speech_active = False
						speech_accum_samples = 0
						prob_sum = None
						frame_count = 0
						votes.clear()
			sd.sleep(int(args.stride * 1000))
		except KeyboardInterrupt:
			print("Stopped.")
		finally:
			state["stop"] = True
			if toggle_thread:
				toggle_thread.join(timeout=0.2)


if __name__ == "__main__":
	main() 