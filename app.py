# importing libraries 
import speech_recognition as sr 
import pandas as pd

import os 
  
from pydub import AudioSegment 
from pydub.silence import split_on_silence 

from rasa.nlu.training_data import load_data
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Trainer
from rasa.nlu import config
from rasa.nlu.model import Metadata, Interpreter

import re
  
# a function that splits the audio file into chunks 
# and applies speech recognition 

def silence_based_conversion(path): 
    k=0

    # open the audio file stored in 
    # the local system as a wav file. 
    song = AudioSegment.from_wav(path) 
    
    # open a file where we will concatenate   
    # and store the recognized text 
    fh = open("recognized.txt", "w+") 
          
    # split track where silence is 0.5 seconds  
    # or more and get chunks 
    chunks = split_on_silence(song,  min_silence_len = 1000,  silence_thresh = -50) 
    #print(chunks)
      
   
    # create a directory to store the audio chunks. 
    try: 
        os.mkdir('audio_chunks') 
    except(FileExistsError): 
        pass
  
    # move into the directory to 
    # store the audio files. 
    os.chdir('audio_chunks') 
  
    i = 0
    # process each chunk 
 
    for chunk in chunks: 
       
        # Create 0.5 seconds silence chunk 
        chunk_silent = AudioSegment.silent(duration = 10) 
  
        # add 0.5 sec silence to beginning and  
        # end of audio chunk. This is done so that 
        # it doesn't seem abruptly sliced. 
        audio_chunk = chunk_silent + chunk + chunk_silent 
  
        # export audio chunk and save it in  
        # the current directory. 
    
        ##------print("saving chunk{0}.wav".format(i)) 
        # specify the bitrate to be 192 k 
        audio_chunk.export("./chunk{0}.wav".format(i), bitrate ='192k', format ="wav") 
  
        # the name of the newly created chunk 
        filename = 'chunk'+str(i)+'.wav'
  
        ##------print("Processing chunk "+str(i)) 
  
        # get the name of the newly created chunk 
        # in the AUDIO_FILE variable for later use. 
        file = filename 
  
        # create a speech recognition object 
        r = sr.Recognizer() 
  
        # recognize the chunk 
        with sr.AudioFile(file) as source: 
            # remove this if it is not working 
            # correctly. 
            r.adjust_for_ambient_noise(source, duration = 0.1) 
            audio_listened = r.listen(source) 
  
        try: 
            # try converting it to text 
            rec = r.recognize_google(audio_listened) 
            # write the output to the file. 
            #text =+ rec
            fh.write(rec+". ") 
  
        # catch any errors. 
        except sr.UnknownValueError: 
            k+=1 
  
        except sr.RequestError as e: 
            print("Could not request results. check your internet connection") 
  
        i += 1
  
    os.chdir('..') 
    

    
def intent_classifier():
    
    model_directory = 'models/pariksha/'
    interpreter = Interpreter.load(model_directory)
    
    file = open('recognized.txt')
    mystring = file.read()
    mystring = mystring.lower()
    text1 = re.sub('[^a-z0-9 ]+', '', mystring)
    intent = interpreter.parse(text1)['intent']['name']
    confidence = interpreter.parse(text1)['intent']['confidence']
    
    return intent, confidence;

def test():
    
    df = pd.DataFrame(columns=['filename','intent','confidence'])    
    loc=0
    folderpath = 'audio/'
    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
        fd = open(filepath, 'r')
        #print(filepath)
        silence_based_conversion(filepath)
    
        intent, confidence = intent_classifier()
        
        
        df = df.append({'filename' : filename,'intent': intent, 'confidence' : confidence} , ignore_index=True)
    
        print("Intent: "+intent)
        print("Confidence: ",confidence)
    print(df)
    df.to_csv('Intent.csv',index=False)
        

if __name__ == "__main__":

    test()
    