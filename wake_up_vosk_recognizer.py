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
        self.wakeup_word = 'hello'
        self.stop_word = 'please'
        #self.wakeup_word = 'салем'
        #self.stop_word = 'мархабат'
        self.detected_wakeup = False
        self.detected_stop = False
        
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
                text = result['text'].lower()
                if not self.detected_wakeup and self.wakeup_word in text:
                    self.detected_wakeup = True
                    print(f'\nWakeup word "{self.wakeup_word}" detected!')
                if self.detected_wakeup:
                    print(f'\rRecognized: {text}     ', end='')
                    if self.stop_word in text:
                        self.detected_stop = True
                        #return text
                        break
            else:
                self.recognizer.PartialResult()
        stream.stop_stream()
        stream.close()
        mic.terminate()
        chat_prompt = result['text'].replace(self.stop_word, "")
        print(chat_prompt)
        return chat_prompt
    

def test():
    sr = SpeechRecognize()
    text = sr.speech_to_text()
    print(f'\nFinal Text: {text}')

if __name__ == '__main__':
    test()