# miscFuns.py
# Module containing helpful functions for E190AP Final Project.

import numpy as np
import librosa as lb
import matplotlib.pyplot as plt

def arg(ang):
    ''' Principal argument function: Wraps input angle to value between -pi and pi. '''
    if ang>np.pi:
        return arg(ang-2*np.pi)
    elif ang<-np.pi:
        return arg(ang+2*np.pi)
    else:
        return ang
    
def halfWave(r):
    ''' returns r, half-wave rectified'''
    if r >= 0:
        return r
    else:
        return 0
    
def loadAudioCalcSTFT(mp3file):
    ''' Loads audio and calculates its STFT 
    '''
    x, sr = lb.core.load(mp3file)
    S = lb.core.stft(x)
    Smag = np.abs(S) # magnitude
    Sphase = np.angle(S) #phase
    N = 1025 #librosa default window size
    return x, sr, N, S, Smag, Sphase


def plotWave(audio, t=None, t0=0, title='', sr=None, short=False):
    ''' Plot the time-energy waveform of a sound. Note that sr defaults to 22050 Hz.
        
        @param t      time range for plotting, given as [t0, t1]
        @param t0     start time, mainly if entire input audio is an excerpt from a longer recording
        @param short  toggles returning trimmed audio
    '''
    
    if sr==None:
        print('WARNING: Sample rate not provided to plotWave. Sample rate has defaulted to 22050 Hz.')
        sr=22050
    
    if t==None:
        start = 0
        end = len(audio)-1
    else:
        start = int(t[0]*sr)
        end = int(t[1]*sr)
        t0 = t[0]
             
    short_audio = audio[start:end]
    samples = np.arange(len(short_audio))
    time = samples/sr + t0
    
    plt.plot(time,short_audio)
    plt.title(title)
    plt.xlabel('Time (s)')
    
    if short: return short_audio
    

def envelope(sound, M=20, sr=22050):
    ''' Generates 1-sided sound envelope based on simple moving averaging method. In order to track the sound
        onset better, the averaging window is biased towards the past data points, rather than centered
        around the current point.
        
        @param sound  sound observation in time-energy domain
        @param M      1/3 of window size of moving average (window from n-2M to n+M)
    '''
    if sr==None:
        print('WARNING: Sample rate not provided to plotWave. Sample rate has defaulted to 22050 Hz.')
        sr=22050
    
    #### NEW METHOD ######
#     sound = np.array(list(map(mf.halfWave, sound)))
    
#     retain only the peaks (filter?)
#     use linear interpolation between the points
#     ignore any pre-peak valleys (may not be necessary?)

#     Ask Prof. Tsai about using autocorrelation for this
    
    #### OLD METHOD ######
    
    sound = np.abs(sound)
        
    for frame in np.arange(len(sound)):
        start = max(0,frame-2*M) # Bias window towards past data
        end = min(len(sound)-1, frame+M)
        sound[frame] = sum(sound[start:end])/(max(end-start,1)) 
    
    return sound


def calcADSR(sound, env, sr=None):
    ''' sound is the time-energy representation (raw audio data)
        x variables are in samples, t variables are in seconds
        
        This function assumes that the sound has been windowed appropriately
        such that the first index is the beginning of the sound.
        
        I categorized sounds into 2 categories based on the severity of release.
        If the post-decay section is a gradual release, this counts it as 0 sustain and full release.
        IF the post-decay section is followed by a hard release section, both sustain and release will be positive.
    '''
    if sr==None:
        print('WARNING: Sample rate not provided to ADSR. Sample rate has defaulted to 22050 Hz.')
        sr=22050
    
    thresh = 0.001 # Minimum sound level (envelope)
    min_slope = 3e-6 # minimum slope threshold (for steady state vs. release)
    slope_run = int(0.1*sr) # Step width for average slope calculation
    
    
#     x_peak = argmax(sound) # Should this be used?
    x_peak_env = np.argmax(env)
    
    # Initialize variables
    x_start = 0
    x_end = len(sound)-1
    
    # Look for start of sound
    for x,eVal in enumerate(env):
        if eVal > thresh: # Start of sound
            x_start = x
            break
    
    # Look for post-peak decay, sustain, and release.
    x_ss1 = x_ss2 = x_peak_env
    inDecay = True
    for x,eVal in enumerate(env[x_peak_env:]):
        x += x_peak_env
        slope = (env[min(x+slope_run, x_end)]-env[x])/(min(x+slope_run, x_end) - x)
#         print([x/sr+t[0],slope])

        if (eVal < thresh) or (x == x_end-1): # End of sound
            x_end = x # Minor off-by-one error?           
            break
            
        elif inDecay and (np.abs(slope) < min_slope): # Start of a sustain/gradual release section
            x_ss1 = x_ss2 = x # Initialize x_ss2 in case of no hard release
            inDecay = False
        elif (inDecay==False) and (np.abs(slope) >= min_slope): # Start of a hard release
            x_ss2 = x
            
    #print('x_start: {}, x_peak_env: {}, x_ss1: {}, x_ss2: {}, x_end: {}'.format(x_start, x_peak_env, x_ss1, x_ss2, x_end))
    attack = (x_peak_env-x_start)/sr
    decay = (x_ss1-x_peak_env)/sr 
    sustain = (x_ss2-x_ss1)/sr 
    release = (x_end-1-x_ss2)/sr
    
    return attack, decay, sustain, release