'''
Implements Vosk speech recognition
'''

import json

import vosk
import pyaudio
import numpy as np

class SpeechRecognize:
    def __init__(self):
        with open('vosk_config.json', 'r') as FP:
            self.config = json.load(FP)
        vosk.SetLogLevel(-1)
        model = vosk.Model(self.config['model'])
        self.recognizer = vosk.KaldiRecognizer(model, 16000)
        
    def speech_to_text(self):
        print('\rListening...      ', end='')
        mic = pyaudio.PyAudio()
        stream = mic.open(format=pyaudio.paInt16,
                        channels = self.config['channels'], 
                        rate=self.config['rate'],
                        input=True,
                        frames_per_buffer=self.config['chunk'] * 2)
        stream.start_stream()

        while True:
            data = stream.read(self.config['chunk'])
            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                break
        return result['text']


def test():
    sr = SpeechRecognize()
    text = sr.speech_to_text()
    print(text)

if __name__ == '__main__':
    test()
