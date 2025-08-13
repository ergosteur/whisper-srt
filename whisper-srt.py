# whisper_srt.py
# Usage examples:
#   python whisper_srt.py --audio input.mp3 --model-dir ./anime_whisper
#   python whisper_srt.py --audio input.mp4 --model-id openai/whisper-small --weights anime_whisper.safetensors

import argparse, os, math, torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor, AutoConfig
from safetensors.torch import load_file as safe_load

def srt_timestamp(seconds: float) -> str:
    if seconds is None or seconds < 0: seconds = 0.0
    h = int(seconds // 3600); m = int((seconds % 3600) // 60)
    s = int(seconds % 60); ms = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

@dataclass
class Caption:
    idx: int; start: float; end: float; text: str

def captions_from_chunks(chunks: List[dict]) -> List[Caption]:
    caps, i = [], 1
    for ch in chunks:
        start = ch.get("start") or (ch.get("timestamp") or [0,0])[0]
        end   = ch.get("end")   or (ch.get("timestamp") or [0,0])[1]
        text  = (ch.get("text") or "").strip()
        if text:
            caps.append(Caption(i, float(start or 0), float(end or 0), text)); i += 1
    return caps

def write_srt(caps: List[Caption], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for c in caps:
            f.write(f"{c.idx}\n{ srt_timestamp(c.start) } --> { srt_timestamp(c.end) }\n{c.text}\n\n")

def load_model(model_dir: Optional[str], model_id: Optional[str], weights: Optional[str], device: str):
    if model_dir:
        proc = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype=torch.float16 if device!="cpu" else torch.float32
        )
        return model, proc
    if model_id:
        proc = WhisperProcessor.from_pretrained(model_id)
        cfg  = AutoConfig.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_config(cfg)
        if weights:
            state = safe_load(weights)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:   print(f"[warn] missing keys: {len(missing)} (showing first 5): {missing[:5]}")
            if unexpected:print(f"[warn] unexpected keys: {len(unexpected)} (first 5): {unexpected[:5]}")
        else:
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16 if device!="cpu" else torch.float32
            )
        return model, proc
    raise ValueError("Provide --model-dir (preferred) or --model-id (with optional --weights).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--model-id")
    ap.add_argument("--weights")
    ap.add_argument("--language", default=None)      # e.g., en, fr, ja (None -> detect)
    ap.add_argument("--task", default="transcribe", choices=["transcribe","translate"])
    ap.add_argument("--chunk-length-s", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)        # cpu, cuda, or auto
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    assert os.path.exists(args.audio), f"Missing audio: {args.audio}"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, proc = load_model(args.model-dir, args.model_id, args.weights, device)
    model.to(device)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        return_timestamps="word",
        torch_dtype=torch.float16 if device!="cpu" else torch.float32,
        device=0 if device=="cuda" else -1,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        generate_kwargs={"task": args.task, "language": args.language},
    )

    print(f"[info] transcribing on {device} …")
    result = asr(args.audio)

    chunks = result.get("chunks") or result.get("segments") or []
    if not chunks:
        text = (result.get("text") or "").strip()
        caps = [Caption(1, 0.0, 0.0, text)] if text else []
    else:
        caps = captions_from_chunks(chunks)

    out_srt = args.output or os.path.splitext(args.audio)[0] + ".srt"
    write_srt(caps, out_srt)
    print(f"[ok] wrote {out_srt}")

if __name__ == "__main__":
    main()
