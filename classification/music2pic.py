
# coding: utf-8

# In[1]:


import sys
import os
from scipy import signal
from scipy.io import wavfile
import glob
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks

# In[2]:


#TODO: dynamic PATH to the music file
#TODO: change SOX to smthin in python
#TODO: Kill redundancies


# In[3]:


#function for change audiofile to wav, it uses SOX, you have to download it
def Change2Wav(directory,form,home):
    #right now it used directory looks like: dir/styles/audioFiles
    #right now it can use windows
    Len = len(form) + 1
    #os.chdir(directory)
    print("%s" %directory)
    print("Starting conversion from " + form + " to wav.")
    for direct in os.listdir(directory):
        for music in os.listdir("{0}\{1}".format(directory,direct)):
            os.chdir(directory)
            os.chdir(direct)
            os.system("sox " + str(music) + " " + str(music[:-Len]) + ".wav")
        os.system('del *.{0}'.format(form))
        os.chdir(home)
        
    print("Finished conversion from " + form + " to wav.")


def Slices(directory,home):
    #os.chdir(directory)
    print("Starting slicing.")
    for direct in os.listdir(directory):
        for music in os.listdir("{0}\{1}".format(directory,direct)):
            #process wav file
            name = music[:-4]
            os.chdir("{0}\{1}".format(directory,direct))

            myaudio = AudioSegment.from_file(music , "wav") 
            chunk_length_ms = 1000 # pydub calculates in millisec
            chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

            for i, chunk in enumerate(chunks):
                if i < 10:
                    chunk_name = name + ".0{0}.wav".format(i)
                else:
                    chunk_name = name + ".{0}.wav".format(i)

                print("exporting {0}".format(chunk_name))
                chunk.export(chunk_name, format="wav")
            os.system('del {0}'.format(music))

        
            os.chdir(home)
    
    print("Finished slicing.")


# In[4]:


#function for make spectograms from wav file
def Conversion2Spectrogram(directory,To,home):
    os.makedirs(To,exist_ok=True)
    #os.chdir(directory)
    print("Starting conversions to spectrogram.")
    for direct in os.listdir(directory):
        for music in os.listdir("{0}\{1}".format(directory,direct)):
            #process wav file
            os.chdir("{0}\{1}".format(directory,direct))
            rate, data = wavfile.read(music)
            #frequencies, times, spectrogram = signal.spectrogram(data, rate)
            os.chdir(home)
            #change directory where I want to save images
            os.chdir(To)
            os.makedirs(direct,exist_ok=True)
            os.chdir(direct)
            #make&save img
            #plt.pcolormesh(times, frequencies, spectrogram)
            plt.specgram(data,Fs=rate)
            plt.savefig("{0}.png".format(music[:-4]))
            plt.close()
            #go back directory
            os.chdir(home)
    
    print("Finished conversions to spectrogram.")


# In[ ]:


#function for make spectograms from wav file
def Conversion2STFT(directory,To,home):
    os.makedirs(To,exist_ok=True)
    #os.chdir(directory)
    print("Starting conversions to STFTs.")
    for direct in os.listdir(directory):
        for music in os.listdir("{0}\{1}".format(directory,direct)):
            #process wav file
            os.chdir("{0}\{1}".format(directory,direct))
            rate, data = wavfile.read(music)
            frequencies, times, STFT = signal.stft(data, rate)
            os.chdir(home)
            #change directory where I want to save images
            os.chdir(To)
            os.makedirs(direct,exist_ok=True)
            os.chdir(direct)
            #make&save img
            plt.pcolormesh(times, frequencies, np.abs(STFT))
            #plt.specgram(data,Fs=rate)
            plt.savefig("{0}.png".format(music[:-4]))
            plt.close()
            #go back directory
            os.chdir(home)
    
    print("Finished conversions to STFTs.")


# In[ ]:


home = os.getcwd()
directory=os.getcwd() + "\genres"
form='au'
SpectDir=os.getcwd() + "\spectrogram"
StftDir=os.getcwd() + "\stft"
Change2Wav(directory,form,home)
Slices(directory,home)
#Conversion2Spectrogram(directory,SpectDir,home)
Conversion2STFT(directory,StftDir,home)

