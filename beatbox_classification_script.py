# Beatboxing Classification Script
#
# Final script for running beatboxing detection and comparing
# different approaches and results

import Beatboxing_Detection as bdet
import multiML as mml
import miscFuns as mf # Optional import for direct access to helper functions


# Use bdet.makeGeneralModel function to either construct an HMM model
# (last semester's system) or create a CSV file of feature vectors
# for the ML classifiers. modelType='hmm' does HMM, anything other value
# creates a csv.

fVecLen = 5 # Current amount of features used
# featVec_df = makeGeneralModel(rec1_directory='training_data/', disp='all', modelType='rf', toCSV=[True, 'rec1_feats.csv', 'over'], fVecLen=fVecLen)
# featVec_df = makeGeneralModel(rec1_directory='freestyle_data/', disp='all', modelType='rf', toCSV=[True, 'freestyle_feats.csv', 'over'], fVecLen=fVecLen)
featVec_df = makeGeneralModel(rec1_directory='query_data/beat1/', disp='all', modelType='rf', toCSV=[True, 'beat1_feats.csv', 'over'], fVecLen=fVecLen, numObsv=beat1_obsv)

# Use mml.trainML
# kn1 will run cross-validation to find a good k value
kn1_params = {'P': 10, 'target': 0.0001, 'avgLim': 0.95, 'disp': 1}
kn2_params = {'k': 18} # Best K for freestyle data

rf_params = {'max_depth': 4, 'num_trees': 100, 'feat_import': True}
nn_params = {'hidden_layer_sizes': (10,10)} # 'max_iter': 200, 'alpha': 1e-4, 'learning_rate_init': 0.1
