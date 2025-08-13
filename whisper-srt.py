# whisper_srt.py  — Whisper → .srt with hybrid GPU/CPU offload support
# Examples:
#   python whisper_srt.py --audio clip.mp4 --model-dir ./anime_whisper --device cuda \
#       --device-map auto --gpu-mem 3GiB --cpu-mem 24GiB --offload-folder ./.offload \
#       --language ja --task transcribe
#
#   python whisper_srt.py --audio clip.mp4 --model-dir ./anime_whisper --device cpu

import argparse, os, math, torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor, AutoConfig
from safetensors.torch import load_file as safe_load

def srt_timestamp(seconds: float) -> str:
    if seconds is None or seconds < 0: seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

@dataclass
class Caption:
    idx: int
    start: float
    end: float
    text: str

def captions_from_chunks(chunks: List[dict]) -> List[Caption]:
    caps: List[Caption] = []
    i = 1
    for ch in chunks:
        # transformers returns either start/end or timestamp=(start,end)
        start = ch.get("start")
        end = ch.get("end")
        if start is None or end is None:
            ts = ch.get("timestamp") or ch.get("timestamps") or (0.0, 0.0)
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                start, end = ts
            else:
                start, end = 0.0, 0.0
        text = (ch.get("text") or "").strip()
        if text:
            caps.append(Caption(i, float(start or 0.0), float(end or 0.0), text))
            i += 1
    return caps

def write_srt(caps: List[Caption], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for c in caps:
            f.write(f"{c.idx}\n")
            f.write(f"{srt_timestamp(c.start)} --> {srt_timestamp(c.end)}\n")
            f.write(c.text + "\n\n")

def build_max_memory(gpu_index: int, gpu_mem: Optional[str], cpu_mem: Optional[str]) -> Optional[Dict[Any, str]]:
    if not gpu_mem and not cpu_mem:
        return None
    mm: Dict[Any, str] = {}
    if gpu_mem:
        mm[gpu_index] = gpu_mem
    if cpu_mem:
        mm["cpu"] = cpu_mem
    return mm

def load_model(model_dir: Optional[str],
               model_id: Optional[str],
               weights_path: Optional[str],
               device: str,
               device_map: Optional[str],
               max_memory: Optional[Dict[Any, str]],
               offload_folder: Optional[str]) -> (WhisperForConditionalGeneration, WhisperProcessor):

    if model_dir:
        processor = WhisperProcessor.from_pretrained(model_dir)
        # If using device_map (hybrid), let Accelerate place layers; otherwise load normally.
        if device_map:
            model = WhisperForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
            )
        else:
            model = WhisperForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            )
        return model, processor

    if model_id:
        processor = WhisperProcessor.from_pretrained(model_id)
        if weights_path:
            # Load config + inject your .safetensors into from_pretrained
            state = safe_load(weights_path)
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                state_dict=state,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
            )
        else:
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
            )
        return model, processor

    raise ValueError("Provide --model-dir (preferred) or --model-id (plus optional --weights).")

def main():
    ap = argparse.ArgumentParser(description="Transcribe audio/video to .srt using a local Whisper model")
    ap.add_argument("--audio", required=True, help="Audio/video path (anything ffmpeg can read)")
    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--model-dir", help="Local HF-style Whisper folder (includes model.safetensors + configs)")
    group.add_argument("--model-id", help="HF model id for base cfg/tokenizer (e.g., openai/whisper-small)")
    ap.add_argument("--weights", help="Optional .safetensors (used with --model-id)")

    ap.add_argument("--language", default=None, help="Source language code (e.g., en, ja). Omit for detection.")
    ap.add_argument("--task", default="transcribe", choices=["transcribe","translate"], help="Task mode")
    ap.add_argument("--chunk-length-s", type=int, default=30, help="Chunk length seconds for streaming ASR")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size per chunk")
    ap.add_argument("--device", default=None, help="cpu, cuda, or auto (default: auto)")

    # Hybrid offload controls
    ap.add_argument("--device-map", default=None,
                    help="Set placement strategy for Accelerate (e.g., auto, balanced, sequential). If set, enables hybrid offload.")
    ap.add_argument("--gpu-index", type=int, default=0, help="GPU index to reserve memory on (default 0)")
    ap.add_argument("--gpu-mem", default=None, help="GPU memory cap, e.g., 3GiB or 3500MiB")
    ap.add_argument("--cpu-mem", default=None, help="CPU RAM cap for offload, e.g., 24GiB")
    ap.add_argument("--offload-folder", default=None, help="Folder for CPU/NVMe offload (recommend fast NVMe)")

    ap.add_argument("--output", default=None, help="Output .srt (default: input basename + .srt)")
    args = ap.parse_args()

    assert os.path.exists(args.audio), f"Missing audio: {args.audio}"

    # Choose device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build memory budget if requested
    max_memory = build_max_memory(args.gpu_index, args.gpu_mem, args.cpu_mem)

    # Load model + processor
    model, processor = load_model(
        args.model_dir, args.model_id, args.weights,
        device=device,
        device_map=args.device_map,
        max_memory=max_memory,
        offload_folder=args.offload_folder
    )

    # Only move to a single device if we are NOT sharding with device_map
    if not args.device_map:
        model.to(device)

    # Pick pipeline device:
    # - If using device_map (sharded), keep pipeline device=-1 (don’t force move)
    # - Else, use GPU (0) if cuda, otherwise CPU (-1)
    pipe_device = -1 if args.device_map else (0 if device == "cuda" else -1)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps="word",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device=pipe_device,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        generate_kwargs={"task": args.task, "language": args.language},
    )

    print(f"[info] device={device}, device_map={args.device_map}, chunk={args.chunk_length_s}s, batch={args.batch_size}")
    result = asr(args.audio)

    chunks = result.get("chunks") or result.get("segments") or []
    if chunks:
        caps = captions_from_chunks(chunks)
    else:
        text = (result.get("text") or "").strip()
        if not text:
            raise RuntimeError("No transcription result returned.")
        caps = [Caption(1, 0.0, 0.0, text)]

    out_srt = args.output or os.path.splitext(args.audio)[0] + ".srt"
    write_srt(caps, out_srt)
    print(f"[ok] wrote subtitles: {out_srt}")

if __name__ == "__main__":
    main()
