import argparse
import json
from glob import glob
from os import makedirs
from os.path import dirname, join, relpath, splitext

import librosa
import pandas as pd
import soundfile as sf
from rich.progress import track


def determine_patient(segments):
    speaker_texts = {"SPEAKER_00": [], "SPEAKER_01": []}
    for segment in segments:
        if "speaker" in segment:
            speaker_texts[segment["speaker"]].append(segment["text"])
    speaker_lens = {speaker: len(texts) for speaker, texts in speaker_texts.items()}
    speaker_ratio = min(speaker_lens.values()) / max(speaker_lens.values())
    if speaker_ratio < 0.05:
        return None
    speaker_texts = {
        speaker: " ".join(texts) for speaker, texts in speaker_texts.items()
    }
    number_of_questions = {
        speaker: text.count("?") for speaker, text in speaker_texts.items()
    }
    patient = min(number_of_questions, key=number_of_questions.get)
    return patient


def filter_segments(segments, speaker):
    segments_filtered = []
    for segment in segments:
        if speaker is None:  # diarization failed, filter out all the questions instead
            if "speaker" in segment and "?" not in segment["text"]:
                segments_filtered.append(segment)
        else:
            if "speaker" in segment and segment["speaker"] == speaker:
                segments_filtered.append(segment)
    return segments_filtered


def csv_transforms(segment, output_path):
    word_df = pd.DataFrame(segment["words"])
    if not word_df.empty:
        word_df.to_csv(f"{output_path}.words.csv", index=False)

    char_df = pd.DataFrame([x for x in segment["chars"]])
    if not char_df.empty:
        char_df.to_csv(f"{output_path}.chars.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Segment wavs based on whisperx transcriptions and diarization. Only write patients' segments."
    )
    parser.add_argument(
        "--wav-dir", required=True, help="Path to audio files for segmentation"
    )
    parser.add_argument(
        "--transcription-dir", required=True, help="Path to transcriptions"
    )
    parser.add_argument("--dest", required=True, help="Path to store outputs")
    args = parser.parse_args()
    wavs = glob(f"{args.wav_dir}/**/*.wav", recursive=True)
    for wav in track(wavs):
        audio_data, sr = librosa.core.load(wav, sr=None)
        matching_json = join(
            args.transcription_dir, relpath(wav, args.wav_dir)
        ).replace(".wav", ".json")
        bn_wav = splitext(join(args.dest, "wav", relpath(wav, args.wav_dir)))[0]
        bn_transc = splitext(
            join(args.dest, "transcriptions", relpath(wav, args.wav_dir))
        )[0]

        with open(matching_json) as f:
            segments = json.load(f)["segments"]

        patient = determine_patient(segments)

        filtered_segments = filter_segments(segments, patient)
        full_transcription_path = f"{bn_transc}.json"
        makedirs(dirname(full_transcription_path), exist_ok=True)
        with open(full_transcription_path, "w") as f:
            json.dump(filtered_segments, f)
        for segment in filtered_segments:
            wav_path = f'{bn_wav}.{segment["start"]}-{segment["end"]}.wav'
            makedirs(dirname(wav_path), exist_ok=True)
            segment_trans_path = f'{bn_transc}.{segment["start"]}-{segment["end"]}'
            csv_transforms(segment, segment_trans_path)
            sf.write(
                wav_path,
                audio_data[
                    int(sr * float(segment["start"])) : int(sr * float(segment["end"]))
                ],
                samplerate=sr,
            )
