import numpy as np


# True ground truths
beat1 = '{ B t / K t / t B / Pf t } { t B / Pch t / t B / ^Ksh t } { B t / K t / t B / Pf t } { t B / Pch t / t B / ^Ksh t }'
beat2 = '{ B ts / Pch tk / B rrh / Pch } { B ts / Pch tk / BB / Pch } { B ts / Pch tk / B rrh / Pch } { B ts / Pch tk / BB / Pch }'
beat3 = '{ B ts / Pch ts / k / k } { B ts / Pch ts / k / k } { B ts / Pch ts / B ts / Pch ts } { B ts / t / - dsh / - dsh } { B ts / Pch ts / k / k } { B ts / Pch ts / k / k } { B ts / Pch ts / B ts / Pch ts } { B ts / t / - dsh / - dsh }'

# Semi-true ground truths (1 of them is true)
beat4a = '{ B t / B - / B ts / B - }{ B rrh / K t t B / K t t B / Pch } { B t / B - / B ts / B - } { B rrh / K t t B / K t t B / Pch }'
beat4b = '{ ts ts / ^Ksh ts / ts ts / ^Ksh ts }{ ts ts / ^Ksh ts / Pch - t / Pch - Pch } { ts ts / ^Ksh ts / ts ts / ^Ksh ts } { Pch t t/ Pch - B / Pch - B / Pch } {dsh / - / - / -}'
beat4c = '{ BB / Pf k t k / B k t k / Pf } { BB / Pf k t k / B k t k / Pf Pf } { BB / Pf k t k / B k t k / Pf } { BB / Pf k t k / BB BB / dsh }'


def sbn2gtDict(sbn, bpm, offset=0):
    '''
    Generate ground truth sound dictionary from an SBN beat pattern (given as string). Outputs times at which each sound occurs,
    where the times are based on the given bpm.

    offset is an optional parameter for specifying how much offset the query has between the start of the recording
    and the 
    '''
    beat = sbn.split() # splits up the beat by spaces
    period = 60/bpm # single beat period in seconds

    soundDict = {}
    t = offset # initialize time
    for i,sound in enumerate(beat):
        if sound == '{':
            sList = []
            continue
            
        elif (sound == '/') or (sound == '}'): # end of a beat period
            numSounds = len(sList)
            dt = period/numSounds
            times = t + dt*np.arange(numSounds) # 0 time increase for first sound

            for i,s in enumerate(sList): # Loop through sounds in this measure (again)

                if s == 'tk': # Only duplet case in our beats
                    if 't' not in soundDict:
                        soundDict['t'] = []
                    if 'k' not in soundDict:
                        soundDict['k'] = []

                    soundDict['t'].append(times[i])
                    soundDict['k'].append(times[i] + dt/2) # Approximation of duplet timing
                    continue
                
                elif s not in soundDict:
                    soundDict[s] = []
                soundDict[s].append(times[i])

            t += period # Move t to the start of next measure
            sList = []            

        else: # You saw an actual sound (or rest)
            if sound == 'dsh': 
                sound = 'ts' # Change to a 'ts' for dictionary purposes
            sList += [sound]        

    if '-' in soundDict:
        rests = soundDict.pop('-') # remove rests from dictionary

    return soundDict
