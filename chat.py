import os
import subprocess
from nltk.tokenize import sent_tokenize
import requests
import numpy as np
import sounddevice as sd
from wake_up_vosk_recognizer import SpeechRecognize


url = "http://localhost:8000/predict"
headers = {"accept": "application/json"}
recog = SpeechRecognize()


def record():
    text = recog.speech_to_text()
    return text


def run_chatbot():
    # Start the chatbot process
    process = subprocess.Popen(
        ['python3', '-m', 'fastchat.serve.cli', '--model-path', 'lmsys/fastchat-t5-3b-v1.0'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8'
    )
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    while True:
        try:
            # Get user input
            prompt = record()
            print(prompt)

            # Send user input to the chatbot process
            process.stdin.write(prompt + '\n')
            process.stdin.flush()

            # Read the response from the chatbot process
            response = process.stdout.readline().strip()
            
            cleaned_response = response.replace("Human: Assistant:", "").strip()
            # Print the chatbot's response
            print("Assistant:", cleaned_response)
            sentences = sent_tokenize(cleaned_response)
            for sentence in sentences:
                params = {"text": sentence}
                response = requests.post(url, params=params, headers=headers)
                data = response.json()
                npa = np.asarray(data['data'], dtype=np.int16)
                sd.play(npa, data['sample-rate'], blocking=True)
            # Check if the chatbot process has terminated
            if process.poll() is not None:
                break

        except KeyboardInterrupt:
            break
    # Close the chatbot process
    process.stdin.close()
    process.stdout.close()
    process.stderr.close()
    process.wait()

if __name__ == '__main__':
    run_chatbot()
