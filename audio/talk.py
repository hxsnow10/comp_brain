#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       sound_to_text
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/10/21
#   Description:    ---
# Python program to translate
# speech to text and text to speech
 
 
import sys
import speech_recognition as sr
import pyttsx3
 
def wav_recognize():
    r = sr.Recognizer()
    WAV = sr.AudioFile(sys.argv[1])
    with WAV as source:
        audio = r.record(source)
    print(r.recognize_google(audio, show_all=True, language="zh_CN")) 
    # to add data for sphinx
    # print(r.recognize_sphinx(audio, language='zh_CN'))
wav_recognize()

def SpeakText(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def talk_to_microphone():
    while(1):
        try:
            with sr.Microphone() as source:
                audio = r.listen(source)
                input_text = r.recognize_google(audio)
                # here could be some dialogue function
                output_text = input_text.lower()
                SpeakText(output_text)
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("unknown error occured")

