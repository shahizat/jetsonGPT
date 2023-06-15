## How to Run a ChatGPT-Like LLM on NVIDIA Jetson Xavier NX board([Hackster.io](https://www.hackster.io/shahizat/how-to-run-a-chatgpt-like-llm-on-nvidia-jetson-board-41fd79))

JetsonGPT is a python based voice assistant that takes two different wake up words running on the Nvidia Jetson Xavier NX. One for the activation of VOSK API Automatic Speech recognition and the other will prompt the [FastChat-T5](https://github.com/lm-sys/FastChat) Large Larguage Model to generated answer based on the user's prompt. For transcribing user's speech implements [Vosk API](https://github.com/alphacep/vosk-api). Text-to-speech is done using [Piper](https://github.com/rhasspy/piper) TTS.

#### Main Requirements
* python >= 3.7
* numpy
* fastapi
* espeak_phonemizer
* uvicorn
* onnxruntime-gpu
* vosk

### Usage
* Download a Piper TTS model voice from [here](https://github.com/rhasspy/piper/releases/tag/v0.0.2) and extract the .onnx and .onnx.json files.
* Download the Vosk model for ASR from [here](https://alphacephei.com/vosk/models).

* Open a terminal and run Piper TTS server program
```
python3 webserver.py
```
* Open another terminal and run a main program
```
python3 chat.py
```

### Acknowledgements
The implementation of the project relies on:
* [An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and FastChat-T5](https://github.com/lm-sys/FastChat)
* [Offline speech recognition API for Android, iOS, Raspberry Pi and servers with Python, Java, C# and Node](https://github.com/alphacep/vosk-api)
* [A fast, local neural text to speech system - Piper TTS](https://github.com/rhasspy/piper)

I thank the original authors for their open-sourcing.


