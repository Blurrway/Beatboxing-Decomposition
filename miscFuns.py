# miscFuns.py
# Module containing helpful functions for E190AP Final Project.

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


def plotWave(audio, t=None, t0=0, title='', sr=sr, short=False):
    ''' Plot the time-energy waveform of a sound.
        
        @param t      time range for plotting, given as [t0, t1]
        @param t0     start time, mainly if entire input audio is an excerpt from a longer recording
        @param short  toggles returning trimmed audio
    '''
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