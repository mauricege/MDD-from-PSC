#!/usr/bin/env python
import logging
from csv import reader
from datetime import datetime
from glob import glob
from io import StringIO
from os import makedirs
from os.path import basename, dirname, join, normpath, sep, splitext
import regex as re

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from rich.progress import track

makedirs("./log", exist_ok=True)
logging.basicConfig(
    filename=join("./log", f"{splitext(basename(__file__))[0]}.log"),
    level=logging.INFO,
)
logger = logging.getLogger(__file__)

INPUT_PATH = "./data/raw/audio"
OUTPUT_PATH = "./data"

label_dict = {"filename": [], "SubjectID": [], "t": []}

fn_regex = re.compile(
    r"./data/raw/audio/(?P<t>\w+)/(?P<subjectID>[A-Z]+_[A-Z]+_\d+)(.*).mp3"
)


all_wavs = sorted(glob(f"{INPUT_PATH}/**/*.mp3", recursive=True))
for wav in track(all_wavs):
    matches = fn_regex.match(wav)
    subject_id = matches.group("subjectID")
    t = matches.group("t")

    filename = f"{subject_id}/{t.lower()}.wav"
    label_dict["filename"].append(filename)
    label_dict["SubjectID"].append(subject_id)
    label_dict["t"].append(t.lower())
    outpath = join(OUTPUT_PATH, "wav", filename)
    makedirs(dirname(outpath), exist_ok=True)
    try:
        wav_data, sr = librosa.core.load(wav, sr=16000)
        wav_data /= np.max(np.abs(wav_data))
    except:
        logger.error(f"{wav}: Corrupted audio file!")
        continue
    sf.write(outpath, wav_data, sr)

label_df = pd.DataFrame(label_dict).drop_duplicates().sort_values(by="filename")
label_df.to_csv(join(OUTPUT_PATH, "labels.csv"), index=None)
