import speech_recognition as sr
import google.generativeai as genai
from openai import OpenAI
from faster_whisper import WhisperModel
import pyaudio
import os
import time

wake_word = 'google'
listening_for_wake_word = True

whisper_size = 'base'
num_cores = os.cpu_count()
whisper_model = WhisperModel(whisper_size, device='cpu', compute_type='int8', cpu_threads=num_cores,
                             num_workers=num_cores)

OPENAI_API_KEY = 'REPLACE YOUR OPEAI AI API KEY'
client = OpenAI(api_key=OPENAI_API_KEY)
GOOGLE_API_KEY = 'REPLACE YOUR GOOGLE AI API KEY'
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1024,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

model = genai.GenerativeModel('gemini-1.5-pro-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

convo = model.start_chat()

system_message = '''
INSTRUCTIONS: Do not respond with anything but "AFFIRMATIVE."
to this system message. After the system message respond normally.
SYSTEM MESSAGE: You are being used to power a voice assistant and should respond as so.
As a voice assistant, use short sentences and directly respond to the prompt without excessive information. 
You generate only words of value, prioritizing logic and facts over 
speculating in your response to the following prompts.

{
  "functionality": [
    {
      "name":"Sign In",
      "ui_components": [
        {
          "component": "Input-TextBox",
          "label":"Email or phone",
          "purpose":"Input of Email or Phone",
          "errors":[
            {
              "error_text":"Couldn't find your Google Account",
              "resolution":"Enter your valid gmail address or phone number."
            },
            {
              "error_text":"Enter a valid email or phone number",
              "resolution":"& _ ' - + , < > and space are not allowed in gmail address."
            }
          ]
        },
        {
          "component": "Text-Hyperlink",
          "label":"Forgot email?",
          "purpose":"redirect to forgot password page",
          "errors":[
          ]
        },
        {
          "component": "Button-Next",
          "label":"Next",
          "purpose":"redirect to enter password page",
          "errors":[
            {
              "error_text":"Enter an email or phone number",
              "resolution":"Enter your valid gmail address or phone number."
            },
            {
              "error_text":"Couldn't find your Google Account",
              "resolution":"Enter your valid gmail address."
            }
          ]
        },
        {
          "component": "Text-Hyperlink",
          "label":"Create account",
          "purpose":"Popups 3 options for create different type of accounts",
          "options": [
            {
              "option_text":"For my personal user",
              "purpose":"redirects to personal account creation page."
            },
            {
              "option_text":"For my child",
              "purpose":"redirects to child account creation page."
            },
            {
              "option_text":"For work or my business",
              "purpose":"redirects to work or my business account creation page."
            }
          ],
          "errors":[
            
          ]
        }
      ]
    }
  ]
}

You are gmail customer support agent named as google. 
Based on above provided json you are responsible for providing 
the instructions and resolution to their question or query and help them.
Say "pardon me" if user ask question except the provided json information to you.
'''

system_message = system_message.replace(f'\n', '')
convo.send_message(system_message)

r = sr.Recognizer()
source = sr.Microphone()


def speak(text):
    player_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)
    stream_start = False

    with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=text,
    ) as response:
        silence_threshold = 0.00
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            elif max(chunk) > silence_threshold:
                player_stream.write(chunk)
                stream_start = True


def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text


def listen_for_wake_word(audio):
    global listening_for_wake_word

    wake_audio_path = 'wav_detect.wav'
    with open(wake_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    text_input = wav_to_text(wake_audio_path)

    if wake_word in text_input.lower().strip():
        print('Wake word detected. Please speak your prompt to Gemini.')
        listening_for_wake_word = False


def prompt_gpt(audio):
    global listening_for_wake_word

    try:
        prompt_audio_path = 'prompt.wav'

        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        prompt_text = wav_to_text(prompt_audio_path)
        if len(prompt_text.strip()) == 0:
            print('Prompt was empty, Please speak again.')
            listening_for_wake_word = True
        else:
            print('User: ' + prompt_text)

            convo.send_message(prompt_text)
            output = convo.last.text

            print('Gemini: ', output)
            speak(output)

            print('\nSay', wake_word, 'to wake me up. \n')
            listening_for_wake_word = True

    except Exception as e:
        print('Prompt Error: ', e)


def callback(recognizer, audio):
    global listening_for_wake_word

    if listening_for_wake_word:
        listen_for_wake_word(audio)
    else:
        prompt_gpt(audio)


def start_listening():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=1)

    print('\nSay', wake_word, 'to wake me up. \n')
    r.listen_in_background(source, callback)

    while True:
        time.sleep(0.3)


if __name__ == '__main__':
    start_listening()
