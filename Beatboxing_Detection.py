
# coding: utf-8

# # E190AP Final Project - Automatic Beatboxing Recognition
# ### Team: Ankoor Apte and Sidney Cozier

# ###### Final design:
# Input - audio query file (.wav)
# 
# Output - soundDict, a Python dictionary with 10 keys corresponding to the 10 pre-defined percussion sounds, and values corresponding to time locations in the recording where the given sound type occured.
# 
# We chose to implement a Hidden Markov Model with 10 *states* corresponding to each of the percussion sounds. The *observations* are each of the sounds made in the audio query (time-domain signals), and our *models* are multivariate normal distributions that are unique to each sound. 
# 
# 
# ### Set up dependencies
# Import statements, matplotlib and global variables

# %%


# get_ipython().run_line_magic('matplotlib', 'inline')


# %%


import numpy as np
import os
import librosa as lb
import matplotlib.pyplot as plt
# import IPython.display as ipd
import scipy.stats as ss
import sklearn.metrics as skm
import pandas as pd

import miscFuns as mf # Misc. helper functions
import multiML # Machine learning pipeline


# %%


#Global variables
numDictSounds = 10
sr = 22050


# ### Observations functions

# %%


def loadAudioCalcSTFT(queryfile, sr=22050, hop_size=512, win_size=2048):
    ''' input: audio filename for wav file
        output: audio time-energy signal, sample rate of loaded audio data and STFT magnitude
    '''
    y, sr = lb.core.load(queryfile, sr=sr)
    S = lb.core.stft(y, n_fft=win_size, hop_length=hop_size)
    Smag = np.abs(S)
    return y, sr, Smag #we only use Smag


# %%


def obtainObservations(audio, sr = 22050, backtrack=1000, numObsv=40, model=False, disp='all'):
    ''' input: audio data obtained from loadAudioandCalcSTFT
        output: tempo_bpm - int specifying estimated tempo
                beats - list of sample indices of beat onsets
                obsvArray - np array of size (win_size, numObsv) where win_size is the block size of an eighth measure in samples
                            numObsv is the number of onsets in the recording
    '''
    
    #get tempo
    tempo_bpm = lb.beat.tempo(audio, sr=sr)
    tempo_bpm = int(round(tempo_bpm[0]))
    
    #correct for rec1 error:
    if tempo_bpm <= 50:
        if disp == 'all':
            print('Tempo miscalculated (<=50), multiplying by 4')
        tempo_bpm *= 4
    
    if disp == 'all':
        print("Tempo in bpm: ", tempo_bpm)
    
    #determine eighth measure window size in samples
    tempo_bps = tempo_bpm/60.0
    quarterMeasure = 1/tempo_bps #time period for 1 beat, i.e. a quarter measure
    eighthMeasure = quarterMeasure/2
    win_size = int(round(eighthMeasure*sr)) #get window size in samples
    
    if disp == 'all':
        print("8th measure window size:", win_size)
    
    #set librosa peak picker parameters
    premax = int((win_size/2.0)/512.0)
    postmax = int((win_size/2.0)/512.0)
    preavg = int((win_size/2.0)/512.0)
    postavg = int((win_size/2.0)/512.0)
    delta = 6.5
    wait = int((win_size/2.0)/512.0)
    
    #compute onset envelope for audio and then use librosa peak picker to get beat locations
    oenv = lb.onset.onset_strength(audio, sr=sr)
    beats_frames = lb.util.peak_pick(oenv, pre_max=premax, post_max=postmax, pre_avg=preavg, post_avg=postavg, delta=delta, wait=wait)
    
    #convert to samples
    beats = lb.frames_to_samples(beats_frames)
    print(beats.shape)
    
    #create observations array
    if model == True: 
    #we hard-code size of obsvArray to be 40 for model training, because all our training data contains 40 known observations
    #this would be addressed in future iterations
        numObsv = 40 
    # else:
    #     numObsv = beats.shape[0]
    obsvArray = np.zeros((win_size, numObsv))
    for i in range(numObsv):
        if i < beats.shape[0]:
            onset = beats[i]
        else:
            onset = beats[-1] #imperfect solution to cases where we get <40 observations, we never hit this case if model==False
        
        #using backtrack parameter, obtain an observation (zero-padded) from the input audio by taking a window of size win_size, starting at onset-backtrack
        isolatedSound = audio[onset-backtrack:onset+win_size-backtrack]
        if (isolatedSound.shape[0] == win_size):
            obsvArray[:,i] = audio[onset-backtrack:onset+win_size-backtrack]
    
    return tempo_bpm, beats, obsvArray

# %%


def calcFeatures(obsv, returnFVL=False): 
    '''single obsv: a 1D array representing samples over an eighth measure containing a percussion sound
       returns a feature vector'''
    fVecLen = 5 
    F = np.zeros(fVecLen)
    stft = np.abs(lb.core.stft(obsv))
    env = mf.envelope(obsv)
    
    #calculate feature parameters
    maxInd = np.argmax(stft) # Linear/Flattened Index
    maxInds = np.unravel_index(maxInd, stft.shape)
    # attack, decay, sustain, release = mf.calcADSR(obsv, env, sr=22050)
    attack, release = mf.calcAR(obsv, env, sr=22050)
    
    # F[0] = np.average(obsv) # May or may not keep
    F[0] = maxInds[0] # Frequency (DFT index)
    F[1] = maxInds[1] # Time (in frames)
    F[2] = np.max(stft)   
    F[3] = attack
    F[4] = release
    # F[6] = sustain
    # F[7] = release
    
    if returnFVL:
        return F, fVecLen
    else:
        return F


# ### Model functions
# Functions used to train the model - there are two types, a general model that includes all the training data, and an individual-specific model that uses the data for one person

# %%


def makeGeneralModel(rec1_directory='training_data/', disp='all', modelType='hmm', params=None, toCSV=[False,'rec1_feats.csv'], fVecLen=8, numObsv=40):
    ''' input: 
        path to folder containing all rec1s
    
        internal variables:
        numFiles - number of files in rec1 folder
        fVec_accum - an array of size (fVecLen, 10) where fVec_accum[:,i] contains a running sum of all fVecs *known* to have state i (of 10 percussion sounds)
        cov_accum - an array of size (fVecLen, 10, fVecLen) where cov_accum[:,i,:] contains a running sum of the outer product of 
        
        output:
        model - (transitions_matrix, means, covs)
                transitions_matrix - np.ones((10,10))/100.0, equal transition probabilities
                means - an array of size (fVecLen, 10) where means[:,i] specifies the mean feature vector for sound type i
                covs - an array of size (fVecLen, 10, fVecLen) where cov[:,i,:] specifies the covariance matrix for sound type i
    '''
    filenames = list(os.walk(rec1_directory))
    filenames = filenames[0][2] # isolate the list of filenames
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    
    #parameters
    numFiles = len(filenames)
    
   
    
    # get fVec_accum, which will hold the SUMS of feature vectors for each sound
    fVec_accum = np.zeros((fVecLen, numDictSounds))
    all_fVecs = np.zeros(fVecLen+1)
    if disp == 'all':
        print('Calculating means...')
    for file in filenames:
        if disp == 'all':
            print('File:', file)
        #load audio and get observations+tempo
        audio, sr, Smag = loadAudioCalcSTFT(rec1_directory+file)
        tempo, beats, obsvArray = obtainObservations(audio, sr=22050, numObsv=numObsv) #tempo and observations need to be looked at, so that we're getting clear samples of each sound
        numObsv = obsvArray.shape[1]
        if disp == 'all':
            print('numObsv:', numObsv)
        
        #add breakpoint here if numObsv != 40, or if anything else is unexpected (e.g. tempo != 120)
        #we need this for the rest of the code to work
        
        #iterate through observations
        for obsv in range(numObsv):
            (sNum, oNum) = divmod(obsv, 4) # Sound Number and observation number (4 observations of each sound)
            thisFeatVec = calcFeatures(obsvArray[:,obsv])
            fVec_accum[:,sNum] += thisFeatVec
            if toCSV[0]:
                # snd = mf.num2Sound[sNum] # SBN text representation of sound
                csv_row = np.append(thisFeatVec, sNum) # Label the features with the ground truth sound
                all_fVecs = np.vstack(([all_fVecs, csv_row])) # Add labeled features to array
                # mf.write_to_csv('rec1_feats.csv', row=csv_row)
            

    if modelType == 'hmm':

        # output data initialization
        means = np.zeros((fVecLen, numDictSounds))
        covs = np.zeros((fVecLen, numDictSounds, fVecLen))
        transition_matrix = np.ones((numDictSounds,numDictSounds))/(numDictSounds**2) #Equal probabilities to avoid bias

        #calculate means
        means = np.divide(fVec_accum,4*numFiles)
        
        #get cov_accum
        cov_accum = np.zeros((fVecLen, numDictSounds, fVecLen))
        
        if disp == 'all':
            print('')
            print('Calculating covs...')
            
        for file in filenames:
            if disp == 'all':
                print('File:', file)
                
            #load audio and get observations+tempo
            audio, sr, Smag = loadAudioCalcSTFT(rec1_directory+file)
            tempo, beats, obsvArray = obtainObservations(audio, sr=22050, model=True) #tempo and observations need to be looked at, so that we're getting clear samples of each sound
            numObsv = obsvArray.shape[1]
            if disp == 'all':
                print('numObsv:', numObsv)
            
            #add breakpoint here if numObsv != 40, or if anything else is unexpected (e.g. tempo != 120)
            #we need this for the rest of the code to work
            
            #iterate through observations
            for obsv in range(numObsv):
                (sNum, oNum) = divmod(obsv, 4) # Sound Number and observation number (4 observations of each sound)
                thisFeatVec = calcFeatures(obsvArray[:,obsv])
                diff = thisFeatVec - means[:,sNum]
                cov_accum[:,sNum,:] += np.outer(diff, diff)
        
        #calculate covs
        covs = np.divide(cov_accum,(4*numFiles)-1)
        
        model = (transition_matrix, means, covs)
        return model

    else: # All other ML classifiers
        
        featfile = toCSV[1]
        if toCSV[0]:
            # col_names = ['Avg. Energy', 'Freq. of Max', 'Time of Max', 'Max Energy', 'Attack', 'Decay', 'Sustain', 'Release', 'Sound SBN']
            col_names = ['Freq. of Max', 'Time of Max', 'Max Energy', 'Attack', 'Release', 'Sound SBN'] # For reduced features
            featVec_df = pd.DataFrame(all_fVecs[1:,:], columns=col_names)
            featVec_df['Sound SBN'] = featVec_df['Sound SBN'].map(mf.num2Sound)
            featVec_df.to_csv(featfile)
            # return featVec_df
        
        


def makeIndividualModel(person):
    ''' Function for making model based on individual sound recordings, i.e. rec1
        
        @param person  The name associated with the recordings for which you want to make a model.
                       This is the first part of the filename, before the underscore.
    '''
    filename = 'audio_data/' + person + '_rec1.wav'
    
    transition_matrix = np.ones((10,10))/100 # Equal probabilities to avoid bias
    
    audio, sr, Smag = loadAudioCalcSTFT(filename)
    tempo, beats, obsvArray = obtainObservations(audio, model=True)
    
    # Loop through all observations
    for obsv in np.arange(obsvArray.shape[1]):
        (sNum, oNum) = divmod(obsv, 4) # Sound Number and observation number (4 observations of each sound)
        
        # Catch Errors
        if sNum > numDictSounds-1:
            print('Warning: Sound count higher than expected. Observations possibly flawed.')
            break
        
        # Calculate feature vector of observation
        thisFeatVec, fVecLen = calcFeatures(obsvArray[:,obsv], returnFVL=True)
        
        if obsv==0: # First observation of a sound 
            thisSound_fVecs = np.zeros((fVecLen,4)) # 4 feature vectors for 1 sound
            thisSound_covs = np.zeros((fVecLen,4,fVecLen)) # 4 Covariance Matrices for each sound
            fVecs_avg = np.zeros((fVecLen,numDictSounds)) # Initialize total means matrix
            fVecs_cov = np.zeros((fVecLen,numDictSounds,fVecLen)) # Each covariance hass fvl x fvl dimensions
        
        # add feature vector to current group
        thisSound_fVecs[:,oNum] = thisFeatVec
        
        # add Covariance matrix to current group
        d1 = (thisFeatVec - fVecs_avg[:,sNum])
        outer = np.outer(d1.reshape((fVecLen,1)), d1.reshape((1,fVecLen))) #don't need to reshape if we are using outer
        thisSound_covs[:,oNum,:] = outer
        
        if oNum == 3: # Last observation of a sound
            fVecs_avg[:,sNum] = np.mean(thisSound_fVecs, axis=1)
            fVecs_cov[:,sNum,:] = (np.sum(thisSound_covs, axis=1))/3

    return (transition_matrix, fVecs_avg, fVecs_cov)


# ### State functions
# Functions used to estimate the states, given observations and a model.

# %%


def generatePairwiseSimilarityMatrix(F, means, covar):
    '''
    Compute a similarity matrix containing the conditional pdf value of a given feature vector given the percussion type.
    
    Arguments:
    M is feature vector size
    F -- feature matrix of size (M, N), where N is the number of audio frames
    means -- a matrix of size (M, 10) whose i-th column specifies the mean chroma feature vector for
        the ith percussion sound
    covar -- matrix of size (M,10,M) specifying the estimated covariance matrix for all percussion sounds

    
    Returns:
    S -- matrix of size (10, N) whose (i,j)-th element specifies the log of the conditional pdf value 
        of observing the j-th feature vector F[:,j] given that the percussion sound was of type i (following the order from our presentation)
    '''
    numObsv = F.shape[1]
    numDictSounds = means.shape[1]
    S = np.zeros((numDictSounds, numObsv))
    
    for i in range(numDictSounds):
        cov = covar[:,i,:]
        for j in range(numObsv):
            S[i,j] = ss.multivariate_normal.logpdf(F[:,j], mean = means[:,i], cov = cov, allow_singular=True) #non-ideal
    return S


# %%


def runBeatboxingRecognitionHMM(beats, obsvArray, model=None, toCSV=[False, '', 'add'], numFeats=8):
    '''
    Estimate the beatboxing sound given an array of sounds
    
    Arguments:
    obsvArray -- array of shape (m,n) where m is the 8th measure window size and n is the number of observations in the audio recording
    model -- trained hidden markov model (fine to skip for CSV output)
    toCSV -- toCSV[0] toggles whether CSV output is executed.
             toCSV[1] is the title of the output file.
             toCSV[2] determines whether to overwrite current file or add to it. ('add' or 'over')
    
    Returns:
    soundDict -- dictionary where the keys are the ten pre-defined beatboxing sounds, and the values are lists of the 
    location(s) of the sound in the query file
    this can be used to make transcriptions of the beatboxing recording
    '''
    soundDict = {}
    numObsv = obsvArray.shape[1]
        
    F = np.zeros((numFeats,numObsv))
    for obsv in range(numObsv):
        F[:,obsv] = calcFeatures(obsvArray[:,obsv])

    if model != None:
        (transition_matrix, means, covar) = model
        S = generatePairwiseSimilarityMatrix(F, means, covar) 

        beat_transcription = []
        for obsv in range(numObsv):
            beat_transcription += [(mf.num2Sound[np.argmax(S[:,obsv])], beats[obsv]/np.float64(sr))]

    if toCSV[0]:
        all_fVecs = np.transpose(F) # Put fVecs in rows

        exists = os.path.isfile(toCSV[1]) # Checks if the file exists
        
        if (toCSV[2] == 'over') or (not exists):
            col_names = ['Avg. Energy', 'Freq. of Max', 'Time of Max', 'Max Energy', 'Attack', 'Decay', 'Sustain', 'Release', 'Sound SBN']
            featVec_df = pd.DataFrame(all_fVecs[1:,:], columns=col_names)
            featVec_df['Sound SBN'] = featVec_df['Sound SBN'].map(mf.num2Sound)
            featVec_df.to_csv(toCSV[1])

        # elif toCSV[2] == 'add':


        return beat_transcription, featVec_df
    
    return beat_transcription



# ### Performance Evaluation

# %%


# ABSOLUTE GROUND TRUTHS

## rec2-4, tempo 80/100/140
beat1 = '{ B t / K t / t B / Pf t } { t B / Pch t / t B / ^Ksh t } { B t / K t / t B / Pf t } { t B / Pch t / t B / ^Ksh t }'
## rec5-7, tempo 80/100/140
beat2 = '{ B ts / Pch tk / B rrh / Pch } { B ts / Pch tk / BB / Pch } { B ts / Pch tk / B rrh / Pch } { B ts / Pch tk / BB / Pch }'

# SEMI-ABSOLUTE GROUND TRUTHS

## rec8, tempo unknown
beat3 = '{ B ts / Pch ts / k / k } { B ts / Pch ts / k / k } { B ts / Pch ts / B ts / Pch ts } { B ts / t / - dsh / - dsh } { B ts / Pch ts / k / k } { B ts / Pch ts / k / k } { B ts / Pch ts / B ts / Pch ts } { B ts / t / - dsh / - dsh }'
## rec9, tempo unknown, 1/3 chance on beat
beat4a = '{ B t / B - / B ts / B - }{ B rrh / K t t B / K t t B / Pch } { B t / B - / B ts / B - } { B rrh / K t t B / K t t B / Pch }'
beat4b = '{ ts ts / ^Ksh ts / ts ts / ^Ksh ts }{ ts ts / ^Ksh ts / Pch - t / Pch - Pch } { ts ts / ^Ksh ts / ts ts / ^Ksh ts } { Pch t t/ Pch - B / Pch - B / Pch } {dsh / - / - / -}'
beat4c = '{ BB / Pf k t k / B k t k / Pf } { BB / Pf k t k / B k t k / Pf Pf } { BB / Pf k t k / B k t k / Pf } { BB / Pf k t k / BB BB / dsh }'

rec9_choices = [0,0,2,0,0,1] # The actual beats performed in the rec9 recordings, where (0,1,2) maps to (4a,4b,4c) above


# %%

## Helper Functions ##
def filt(recNum, filenames):
    ''' Filter filenames for 1 recording number'''
    recStr = 'rec'+str(recNum)
    f = lambda x: recStr in x
    filenames = list(filter(f, filenames))
    
    return filenames


def multiFilt(recNums, filenames):
    ''' Filter filenames for multiple recording numbers using filt'''
    newFilenames = []
    for num in recNums:
        newFilenames += filt(num, filenames)
    
    return newFilenames


## Main Function ##
def batchBeatboxingRecognition(dataPath='query_data/', recNums=[0], model=None, disp='all'):
    ''' disp = 'all', 'res', or 'none'. 'res' gives results only
    '''
    # Retrieve files from directory
    filenames = list(os.walk(dataPath))
    filenames = filenames[0][2]
    
    # If specific recording specified, only take those
    if recNums[0]:
        filenames = multiFilt(recNums, filenames)
    
    # Remove any extra files
    if 'desktop.ini' in filenames:
        filenames.remove('desktop.ini')
    numFiles = len(filenames)
    
    #Train general model - this takes about 2 minutes
    modelOut=False
    if model == None:
        modelOut = True
        model = csv2model() # Possibly error-check for csv files
#         model = makeGeneralModel('training_data/')
    
    multChoiceRes = []
    freeTempoRes = []
    absGTRes = [] # Different groups of results

    for file in filenames:
        fullName = dataPath + file
        if disp == 'all':
            print(file)
        
        #Get observations from audio
        audio, sr, Smag = loadAudioCalcSTFT(fullName)
        tempo, beats, obsvArray = obtainObservations(audio, disp=disp)        
        output = runBeatboxingRecognitionHMM(beats, obsvArray, model) # List of sound-time tuples
        
        # Separate into sounds and times
        output_sounds = [pair[0] for pair in output]
        output_times = [pair[1] for pair in output]
        
        recNum = int(file[-5]) # Recording number
        result = testResults(output_sounds, recNum, tempo) # Test procedure changes depending on recording
        
        if recNum == 9:
            if disp == 'all':
                print('Adding result to multChoiceRes')
            multChoiceRes += [result]
        elif recNum == 8:
            if disp == 'all':
                print('Adding result to freeTempoRes')
            freeTempoRes += [result]
        else:
            if disp == 'all':
                print('Adding result to absGTRes')
            absGTRes += [result]     
        if disp == 'all':
            print('')
    
    # Print results
    if (disp == 'res') or (disp == 'all'):
        print('Raw sound match percentages (absGTRes, freeTempoRes):')
        print([absGTRes, freeTempoRes])
    
        
        agtNums = [n for n in recNums if n <= 4]
        ftrNums = [n for n in recNums if n in [5,6,7]]
        
        # Print Summary statistics
        if len(absGTRes) > 0:
            agtrStats = ss.describe(absGTRes)
            print('Absolute Ground Truth Stats for {}, recording(s) {}:'.format(dataPath[:-1], agtNums))
            print(agtrStats)
            print('')
        if len(freeTempoRes) > 0:
            ftrStats = ss.describe(freeTempoRes)
            print('Free Tempo Stats for {}, recording(s) {}:'.format(dataPath[:-1], ftrNums))
            print(ftrStats)
            print('')
        if len(multChoiceRes) > 0:
            matches = 0
            for i, res in enumerate(multChoiceRes):
                if res == rec9_choices[i]:
                    matches += 1
            
            mcrStats = float(matches)/float(len(multChoiceRes))*100
            print('Multiple Choice Ground Truth Results for {}, recording 9:'.format(dataPath[:-1]))
            print('Predicted patterns:', multChoiceRes)
            print('Ground Truth patterns:', rec9_choices)
            print('Match rate: {}%'.format(mcrStats))
            print('')

        print('') 
    
    
    if modelOut:
        return multChoiceRes, freeTempoRes, absGTRes, model
    else:
        return multChoiceRes, freeTempoRes, absGTRes
    
    
# %%


def testResults(output_sounds, recNum, tempo):
    recNum2Tempo = {2: 80, 3: 100, 4: 140, 5: 80, 6: 100, 7: 140}
    
    if recNum == 1:
        raise ValueError('Training data (rec1) provided as input to testResults. Please use data other than training data for testing system.')
    
    elif recNum <= 8:
        if recNum <= 4:
            gtlabels = mf.sbn2gtlabels(beat1, recNum2Tempo[recNum], listOut=True)
        elif recNum <= 7:
            gtlabels = mf.sbn2gtlabels(beat2, recNum2Tempo[recNum], listOut=True)
        elif recNum == 8:
            gtlabels = mf.sbn2gtlabels(beat3, tempo, listOut=True) # Based on the program's calculated tempo      
        
        # Separate sound and time to match predicted output
        gtSounds = [pair[0] for pair in gtlabels] 
        gtTimes = [pair[1] for pair in gtlabels]
        
        # Just truncate whichever one is longer (TODO: change this)
        numObs = min(len(output_sounds), len(gtSounds))
        result = skm.f1_score(gtSounds[:numObs],output_sounds[:numObs], average='micro') # calculates numMatches/numObsv
        
    elif recNum == 9:
        gt1 = mf.sbn2gtlabels(beat4a, tempo, listOut=True) # Based on the program's calculated tempo
        gt2 = mf.sbn2gtlabels(beat4b, tempo, listOut=True) # Based on the program's calculated tempo
        gt3 = mf.sbn2gtlabels(beat4c, tempo, listOut=True) # Based on the program's calculated tempo
        
        gtSounds1 = [pair[0] for pair in gt1] 
        gtTimes1 = [pair[1] for pair in gt1]
        gtSounds2 = [pair[0] for pair in gt2] 
        gtTimes2 = [pair[1] for pair in gt2]
        gtSounds3 = [pair[0] for pair in gt3] 
        gtTimes3 = [pair[1] for pair in gt3]
        
        scores = []
        # Just truncate whichever one is longer (TODO: change this)
        numObs = min(len(output_sounds), len(gtSounds1))
        scores += [skm.f1_score(gtSounds1[:numObs],output_sounds[:numObs], average='micro')] # calculates numMatches/numObsv
        
        # Just truncate whichever one is longer (TODO: change this)
        numObs = min(len(output_sounds), len(gtSounds2))
        scores += [skm.f1_score(gtSounds2[:numObs],output_sounds[:numObs], average='micro')] # calculates numMatches/numObsv
        
        # Just truncate whichever one is longer (TODO: change this)
        numObs = min(len(output_sounds), len(gtSounds3))
        scores += [skm.f1_score(gtSounds3[:numObs],output_sounds[:numObs], average='micro')] # calculates numMatches/numObsv
        
        result = np.argmax(scores) # Pick the one that's most similar to the estimated pattern
        
    return result



def model2csv(model, title='', output=False):
    ''' model2csv(model, title='', output=False)
    Makes csv files of the means and covariances as a record of the model.
    
    @param model  tuple of transition matrix, means, and covariances from makeGeneralModel or makeIndividualModel
    @param output  toggle whether function returns dataframe or not
    
    @return 

    '''
    means = model[1] # 2d array of mean feat. vecs
    covs = model[2] # 3d array of covs (inds 0 and 2 hold one matrix)
    
    ## MEANS ##
    # Create axis labels
    sounds = list(mf.sound2Num.keys())
    features = ['Average Energy', 'Freq of Max Energy', 'Time of Max Energy', 'Max Energy Value',
               'Attack', 'Decay', 'Sustain', 'Release']
    
    # Create dictionary of means
    meanDict = {}
    for sNum, snd in enumerate(sounds):
        for fNum, feat in enumerate(features):
            if snd not in meanDict:
                meanDict[snd] = []          
            meanDict[snd] += [means[fNum, sNum]]
    
    # Create dataframe from dictionary
    meanDF = pd.DataFrame(data=meanDict, index=features) # Data frame with feat vec. for each sound as column
    
    # Create csv file from dataframe
    mStr = title+'_means.csv'; meanDF.to_csv(mStr)
    
    
    ## COVARIANCES ##
    fVecLen = covs.shape[0]
    numSounds = covs.shape[1]
    
    # Create flattened array of covariances
    flatCovs = np.zeros((fVecLen*numSounds, fVecLen))
    for sInd in np.arange(numSounds):
        start = fVecLen*sInd
        flatCovs[start:start+fVecLen, :] = covs[:,sInd,:]
    
    # Create dataframe from array
    covDF = pd.DataFrame(data=flatCovs)
    
    # Create csv file from dataframe
    cStr = title+'_covs.csv'; covDF.to_csv(cStr)
        
    if output: return meanDF, covDF, fVecLen


    
def csv2model(mean_csv='general-model_means.csv', cov_csv='general-model_covs.csv', transMat_csv=None, fVecLen=8):
    ''' Converts csv files of model means and covariances into a usable model. The initial csv files can be generated
        with model2csv.
        
        @param mean_csv, cov_csv  CSV files containing model means and covariances
        @param fVecLen            Length of feature vector, needed to interpret flattened covariance data
        
        @return model  A tuple of transition matrix, means, and co-variances.
    '''
    sounds = list(mf.sound2Num.keys())
    numSounds = len(sounds)
    features = ['Average Energy', 'Freq of Max Energy', 'Time of Max Energy', 'Max Energy Value',
               'Attack', 'Decay', 'Sustain', 'Release']
    
    ## TODO: Implement else case (once model2csv part is done)
    if transMat_csv == None:
        transition_matrix = np.ones((10,10))/100 # Equal probabilities to avoid bias
    
    meanDF = pd.read_csv(mean_csv, index_col=0)
    covDF = pd.read_csv(cov_csv, index_col=0)
    
    means = np.zeros((fVecLen,numSounds))
    covs = np.zeros((fVecLen, numSounds, fVecLen))
    for sInd, snd in enumerate(sounds):
        means[:,sInd] = meanDF[snd]
        
        flatCovs = covDF.values
        start = fVecLen*sInd
        covs[:,sInd,:] = flatCovs[start:start+fVecLen,:]
    
    return (transition_matrix, means, covs)


# %%

def main():
    print('Starting main...')
    fVecLen = 5
    # featVec_df = makeGeneralModel(rec1_directory='training_data/', disp='all', modelType='rf', toCSV=[True, 'rec1_feats.csv', 'over'], fVecLen=fVecLen)
    # featVec_df = makeGeneralModel(rec1_directory='freestyle_data/', disp='all', modelType='rf', toCSV=[True, 'freestyle_feats.csv', 'over'], fVecLen=fVecLen)

    beat1_obsv = 32
    beat2_obsv = 26
    beat3_obsv = 38

    featVec_df = makeGeneralModel(rec1_directory='query_data/beat1/', disp='all', modelType='rf', 
        toCSV=[True, 'beat1_feats.csv', 'over'], fVecLen=fVecLen, numObsv=beat1_obsv)


    # audio, sr, Smag = loadAudioCalcSTFT(filepath)
    # tempo, beats, obsvArray = obtainObservations(audio, sr=22050, model=True)
    # beat_transcription, runBeatboxingRecognitionHMM(beats, obsvArray, toCSV=[False, '', 'add'])

    # filenames = list(os.walk('query_data/beat1/'))
    # print(filenames[0][2])

if __name__ == '__main__':
    main()


