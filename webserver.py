from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
import io
import json
from pydub import AudioSegment
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union
import numpy as np
import onnxruntime
from espeak_phonemizer import Phonemizer
from functools import partial
import logging
import logging.config
import time


_FILE = Path(__file__)
_DIR = _FILE.parent

FORMAT = "%(levelname)s:%(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG)
model = None
synthesize = None
_BOS = "^"
_EOS = "$"
_PAD = "_"

@dataclass
class PiperConfig:
    num_symbols: int
    num_speakers: int
    sample_rate: int
    espeak_voice: str
    length_scale: float
    noise_scale: float
    noise_w: float
    phoneme_id_map: Mapping[str, Sequence[int]]


class Piper:
    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        use_cuda: bool = False,
    ):
        if config_path is None:
            config_path = f"{model_path}.json"

        self.config = load_config(config_path)
        self.phonemizer = Phonemizer(self.config.espeak_voice)
        self.model = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=onnxruntime.SessionOptions(),
        providers=[
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider"
        ],
        )

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
    ) -> bytes:
        """Synthesize WAV audio from text."""
        if length_scale is None:
            length_scale = self.config.length_scale

        if noise_scale is None:
            noise_scale = self.config.noise_scale

        if noise_w is None:
            noise_w = self.config.noise_w

        phonemes_str = self.phonemizer.phonemize(text)
        phonemes = [_BOS] + list(phonemes_str)
        phoneme_ids: List[int] = []

        for phoneme in phonemes:
            phoneme_ids.extend(self.config.phoneme_id_map[phoneme])
            phoneme_ids.extend(self.config.phoneme_id_map[_PAD])

        phoneme_ids.extend(self.config.phoneme_id_map[_EOS])

        phoneme_ids_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
        phoneme_ids_lengths = np.array([phoneme_ids_array.shape[1]], dtype=np.int64)
        scales = np.array(
            [noise_scale, length_scale, noise_w],
            dtype=np.float32,
        )

        # if (self.config.num_speakers > 1) and (speaker_id is not None):
        #     # Default speaker
        #     speaker_id = 0

        sid = None

        if speaker_id is not None:
            sid = np.array([speaker_id], dtype=np.int64)

        # Synthesize through Onnx
        audio = self.model.run(
            None,
            {
                "input": phoneme_ids_array,
                "input_lengths": phoneme_ids_lengths,
                "scales": scales,
                "sid": sid,
            },
        )[0].squeeze((0, 1))
        audio = audio_float_to_int16(audio.squeeze())
        return audio, self.config.sample_rate

def load_config(config_path: Union[str, Path]) -> PiperConfig:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = json.load(config_file)
        inference = config_dict.get("inference", {})

        return PiperConfig(
            num_symbols=config_dict["num_symbols"],
            num_speakers=config_dict["num_speakers"],
            sample_rate=config_dict["audio"]["sample_rate"],
            espeak_voice=config_dict["espeak"]["voice"],
            noise_scale=inference.get("noise_scale", 0.667),
            length_scale=inference.get("length_scale", 1.0),
            noise_w=inference.get("noise_w", 0.8),
            phoneme_id_map=config_dict["phoneme_id_map"],
        )


def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Normalize audio and convert to int16 range"""
    audio_norm = audio * (max_wav_value / max(0.01, np.max(np.abs(audio))))
    audio_norm = np.clip(audio_norm, -max_wav_value, max_wav_value)
    audio_norm = audio_norm.astype("int16")
    return audio_norm


def load_model():
    global synthesize
    if synthesize:
        return synthesize
    model = 'en-us-ryan-low.onnx'
    speaker_id=None
    voice = Piper(model)
    synthesize = partial(
        voice.synthesize,
        speaker_id=speaker_id,
        length_scale=None, 
        noise_scale=0.5,
        noise_w=0.2,)
    logging.debug("Model loaded.")
    return  synthesize


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Run at startup!")
    classifier = load_model()
    print(classifier)
    yield
    print("Run on shutdown!")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
   return {"message": "Hello World"}


@app.post("/predict")
async def predict(text: str): 
    t0 = time.time()
    audio_norm, sample_rate = synthesize(text) 
    t1 = time.time() 
    return {
       'data': audio_norm.tolist(),
       'sample-rate': sample_rate,
       'inference':  t1 -t0,
   }

if __name__ == '__main__':
    uvicorn.run("__main__:app", host='127.0.0.1', port=8000)