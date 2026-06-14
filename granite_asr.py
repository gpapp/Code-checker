#!/usr/bin/env python3
"""
granite_asr.py
ASR using ibm-granite/granite-speech-4.1-2b-plus with word-level timestamps.
"""
import os
import re
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "ibm-granite/granite-speech-4.1-2b-plus"

def _system_prompt():
    from datetime import date
    today = date.today()
    return (
        "Knowledge Cutoff Date: April 2024.\n"
        f"Today's Date: {today.strftime('%B %d, %Y')}.\n"
        "You are Granite, developed by IBM. You are a helpful AI assistant"
    )

TS_PROMPT = (
    "<|audio|> Timestamps: Transcribe the speech. "
    "After each word, add a timestamp tag showing the end time in centiseconds, "
    "e.g. hello [T:45] world [T:82]"
)

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
CHUNK_OVERLAP = 2
MAX_NEW_TOKENS = 10000


class GraniteASR:
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = None

    def _load_model(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Granite Speech model: {MODEL_NAME} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME, device_map=self.device, dtype=torch.bfloat16
        )
        self.model.eval()
        logger.info("Granite Speech model loaded.")

    def transcribe(
        self,
        audio_path: str,
        language: str = "hu",
        cache_path: str = None,
    ) -> tuple[str, list[dict]]:
        import torch
        import librosa

        self._load_model()

        tmp_path = cache_path + ".tmp" if cache_path else None
        all_words: list[dict] = []

        if tmp_path and os.path.exists(tmp_path):
            try:
                with open(tmp_path, "r", encoding="utf-8") as f:
                    checkpoint = json.load(f)
                    all_words = checkpoint.get("words", [])
                    logger.info(
                        f"Resuming {os.path.basename(audio_path)} from checkpoint "
                        f"({len(all_words)} words)."
                    )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {tmp_path}: {e}. Starting from scratch.")

        logger.info(f"Loading audio: {os.path.basename(audio_path)}...")
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        total_samples = len(audio)
        total_seconds = total_samples / SAMPLE_RATE
        logger.info(f"Audio duration: {total_seconds:.1f}s")

        if all_words:
            resume_end = all_words[-1]["end"]
            logger.info(f"Resuming from word end time: {resume_end:.2f}s")
            start_sample = int(resume_end * SAMPLE_RATE)
            audio = audio[start_sample:]
            offset_seconds = resume_end
        else:
            offset_seconds = 0.0

        chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
        overlap_samples = CHUNK_OVERLAP * SAMPLE_RATE
        prev_transcript = ""
        chunk_idx = 0

        pos = 0
        while pos < len(audio):
            end_pos = min(pos + chunk_samples, len(audio))
            chunk = audio[pos:end_pos]
            chunk_offset = offset_seconds + pos / SAMPLE_RATE

            chunk_text, chunk_words = self._transcribe_chunk(
                chunk, chunk_offset, prev_transcript
            )

            if chunk_words:
                all_words.extend(chunk_words)
                prev_transcript = (prev_transcript + " " + chunk_text).strip()

            if tmp_path:
                try:
                    full_text = " ".join(w["word"] for w in all_words).strip()
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump({"text": full_text, "words": all_words}, f, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Checkpoint save failed: {e}")

            pos = end_pos - overlap_samples if end_pos < len(audio) else len(audio)
            chunk_idx += 1

        final_text = " ".join(w["word"] for w in all_words).strip()

        if tmp_path and os.path.exists(tmp_path) and cache_path:
            if os.path.exists(cache_path):
                os.remove(cache_path)
            os.rename(tmp_path, cache_path)

        logger.info(f"Granite transcription complete: {len(all_words)} words")
        return final_text, all_words

    def _transcribe_chunk(
        self,
        audio: np.ndarray,
        offset_seconds: float,
        prefix_text: str,
    ) -> tuple[str, list[dict]]:
        import torch

        @torch.inference_mode()
        def _run():
            chat = [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": TS_PROMPT},
            ]
            extra = {"prefix_text": prefix_text} if prefix_text else {}
            prompt_text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, **extra
            )
            inputs = self.processor(
                prompt_text, audio, device=self.device, return_tensors="pt"
            ).to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
            new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
            return self.tokenizer.decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )

        ts_text = _run()
        words = self._parse_timestamps(ts_text, offset_seconds)
        plain_text = re.sub(r"\s*\[T:\d+\]", "", ts_text).strip()
        return plain_text, words

    @staticmethod
    def _parse_timestamps(ts_text: str, offset_seconds: float) -> list[dict]:
        ts_words = re.split(r"\[T:(\d+)\]", ts_text)
        words = []
        last_end_time = 0.0
        offset_time = 0.0

        raw_pairs = list(zip(ts_words[::2], ts_words[1::2])) if len(ts_words) > 1 else []

        for raw_word, ts_str in raw_pairs:
            word = raw_word.strip()
            if not word:
                continue
            try:
                word_end_time_cs = float(ts_str)
            except ValueError:
                continue

            word_end_time = word_end_time_cs / 100.0
            while word_end_time + offset_time < last_end_time:
                offset_time += 10.0

            absolute_end = word_end_time + offset_time + offset_seconds
            absolute_end = round(absolute_end, 3)

            if words:
                word_start = words[-1]["end"]
            else:
                word_start = round(offset_seconds, 3)

            if absolute_end > word_start:
                words.append({
                    "word": word,
                    "start": word_start,
                    "end": absolute_end,
                    "probability": 1.0,
                })
                last_end_time = word_end_time + offset_time

        return words

    def unload(self):
        import torch
        if self.model is not None:
            del self.model
            self.model = None
        self.processor = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Granite Speech model unloaded.")
