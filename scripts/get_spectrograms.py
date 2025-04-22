# Calculate spectrograms from audio files in the dataset

import os
import numpy as np
import base64
import zipfile
from math import isclose
from typing import Literal
import time
import traceback
from tqdm.auto import tqdm, trange
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import random
import datetime
from PIL import Image
from multiprocessing import Process, Queue

try:
    from pytubefix import Playlist, YouTube, Channel
except ImportError:
    try:
        from pytube import Playlist, YouTube, Channel  # type: ignore
    except ImportError:
        raise ImportError("Please install the pytube library to download the audio. You can install it using `pip install pytube` or `pip install pytubefix`")

from AutoMasher.fyp.audio.dataset import DatasetEntry, SongDataset, create_entry
from AutoMasher.fyp import Audio
from AutoMasher.fyp.audio.analysis import BeatAnalysisResult, DeadBeatKernel, analyse_beat_transformer
from AutoMasher.fyp.audio.separation import DemucsAudioSeparator, DemucsCollection
from AutoMasher.fyp.util import (
    clear_cuda,
    YouTubeURL,
)
from itertools import zip_longest
import json
import torch
from remucs.constants import (
    BEAT_MODEL_PATH,
    PROCESSED_SPECTROGRAMS_URLS
)


def main(path: str):
    song_ds = SongDataset(path, max_dir_size=None)
    audio_urls = song_ds.list_urls("audio")
    demucs = DemucsAudioSeparator()

    def mark_processed(url):
        song_ds.write_info(PROCESSED_SPECTROGRAMS_URLS, url)
        processed.add(url)

    processed = song_ds.read_info_urls(PROCESSED_SPECTROGRAMS_URLS)
    for url in tqdm(song_ds.list_urls("parts"), desc="Updating processed URLs"):
        if url in processed:
            continue
        mark_processed(url)

    for url in audio_urls:
        clear_cuda()

        t = time.time()
        path = song_ds.get_path("audio", url)
        if not os.path.exists(path):
            # Not necessary, but just to be safe
            continue

        spec_path = song_ds.get_path("parts", url)
        if os.path.exists(spec_path):
            continue

        if url in processed:
            continue

        try:
            audio = Audio.load(path)
        except Exception as e:
            tqdm.write(f"Error loading audio: {e}")
            processed.add(url)
            continue

        tqdm.write(f"Processing {url}")
        try:
            parts = demucs.separate(audio)
        except Exception as e:
            tqdm.write(f"Error separating audio: {e}")
            processed.add(url)
            continue

        tqdm.write(f"Finished processing parts for {url} (t={time.time()-t:.2f}s)")

        parts_path = song_ds.get_path("parts", url)
        parts.save(parts_path)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
