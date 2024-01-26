#!/usr/bin/env python
import json
import os
from glob import glob

import pandas as pd
import torch
import whisperx
import yaml
from dotenv import dotenv_values
from tqdm import tqdm

with open("params.yaml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)["transcriptions"]


HF_TOKEN = dotenv_values(".env.secret")["HUGGINFACE_KEY"]
INPUT_DIR = "./data/wav"  # e.g., EMA
OUTPUT_DIR = "./data/transcriptions"

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

audio_paths = sorted(glob(f"{INPUT_DIR}/**/*.wav", recursive=True))
transcriptions = {}
# 1. load all models
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
model_a, metadata = whisperx.load_align_model(
    language_code=params["language"], device=device
)
diarize_model = whisperx.DiarizationPipeline(
    use_auth_token=HF_TOKEN,
    device=device,
)
for audio_file in tqdm(audio_paths):
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language=params["language"])

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=True,
    )

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    # add min/max number of speakers if known
    if params["diarize"]:
        min_speakers = params.get("min_speakers")
        max_speakers = params.get("max_speakers")

        diarize_segments = diarize_model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)
    output_path = os.path.splitext(
        os.path.join(OUTPUT_DIR, os.path.relpath(audio_file, INPUT_DIR))
    )[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(f"{output_path}.json", "w") as fp:
        json.dump(result, fp)
    flat_transcription = " ".join([segment["text"] for segment in result["segments"]])
    with open(f"{output_path}.txt", "w") as fp:
        fp.write(str(flat_transcription.encode("utf-8")))

    # Andreas csv transformations
    pd.DataFrame(result["word_segments"]).to_csv(
        f"{output_path}.words.csv", index=False
    )
    df = pd.DataFrame(
        [
            {key: x[key] for key in x if key not in ["words", "chars"]}
            for x in result["segments"]
        ]
    )
    if "text" in df:
        df["text"] = df["text"].apply(lambda x: x[1:] if x[0] == " " else x)
        df.to_csv(os.path.join(f"{output_path}.text.csv"), index=False)
    pd.DataFrame([y for x in result["segments"] for y in x["chars"]]).to_csv(
        f"{output_path}.chars.csv", index=False
    )
