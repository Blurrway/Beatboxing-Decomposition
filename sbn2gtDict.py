import numpy as np


# True ground truths
beat1 = '{ B t / K t / t B / Pf t } { t B / Pch t / t B / ^Ksh t } { B t / K t / t B / Pf t } { t B / Pch t / t B / ^Ksh t }'
beat2 = '{ B ts / Pch tk / B rrh / Pch } { B ts / Pch tk / BB / Pch } { B ts / Pch tk / B rrh / Pch } { B ts / Pch tk / BB / Pch }'
beat3 = '{ B ts / Pch ts / k / k } { B ts / Pch ts / k / k } { B ts / Pch ts / B ts / Pch ts } { B ts / t / - dsh / - dsh } { B ts / Pch ts / k / k } { B ts / Pch ts / k / k } { B ts / Pch ts / B ts / Pch ts } { B ts / t / - dsh / - dsh }'

# Semi-true ground truths (1 of them is true)
beat4a = '{ B t / B - / B ts / B - }{ B rrh / K t t B / K t t B / Pch } { B t / B - / B ts / B - } { B rrh / K t t B / K t t B / Pch }'
beat4b = '{ ts ts / ^Ksh ts / ts ts / ^Ksh ts }{ ts ts / ^Ksh ts / Pch - t / Pch - Pch } { ts ts / ^Ksh ts / ts ts / ^Ksh ts } { Pch t t/ Pch - B / Pch - B / Pch } {dsh / - / - / -}'
beat4c = '{ BB / Pf k t k / B k t k / Pf } { BB / Pf k t k / B k t k / Pf Pf } { BB / Pf k t k / B k t k / Pf } { BB / Pf k t k / BB BB / dsh }'


def sbn2gtDict(sbn, bpm, offset=0, listOut=False):
    '''
    Generate ground truth sound dictionary from an SBN beat pattern (given as string). Outputs times at which each sound occurs,
    where the times are based on the given bpm.

    offset is an optional parameter for specifying how much offset the query has between the start of the recording
    and the first onset. While the offset can always be added afterwards, it may be nice to add it here if it is known.

    If listOut is set to True, the function outputs a list of tuples (sound-time pairs in their order of occurence).
    This is a better output form for looking at the sequence of sounds. 
    '''
    beat = sbn.split() # splits up the beat by spaces
    period = 60/bpm # single beat period in seconds
    
    totalSoundList = []
    totalSoundDict = {}
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
                    
                    if listOut:
                        totalSoundList += [('t', times[i])]
                        totalSoundList += [('k', times[i] + dt/2)] # Approximation of duplet timing
                    else:
                        if 't' not in totalSoundDict:
                            totalSoundDict['t'] = []
                        if 'k' not in totalSoundDict:
                            totalSoundDict['k'] = []

                        totalSoundDict['t'].append(times[i])
                        totalSoundDict['k'].append(times[i] + dt/2) # Approximation of duplet timing
                    continue
                
                elif listOut:
                    totalSoundList += [(s, times[i])]
                
                else:
                    if s not in totalSoundDict:
                        totalSoundDict[s] = []
                    totalSoundDict[s].append(times[i])

            t += period # Move t to the start of next measure
            sList = []            

        else: # You saw an actual sound (or rest)
            if sound == 'dsh': 
                sound = 'ts' # Change to a 'ts' for dictionary purposes
            sList += [sound]        

    if '-' in totalSoundDict:
        rests = totalSoundDict.pop('-') # remove rests from dictionary

    if listOut: 
        return totalSoundList
    else: 
        return totalSoundDict