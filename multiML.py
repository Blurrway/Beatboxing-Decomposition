# Multi-ML pipeline
# -----------------
# Machine Learning Pipeline with options for multiple different
# classifiers.

# %%
# Import Statements
import numpy as np            
import pandas as pd
import math
# import matplotlib.pyplot as plt

# import miscFuns # Stores old helper functions and variables

# Random Forests
from sklearn import ensemble
# Neural Nets
from sklearn.neural_network import MLPClassifier
# k-neighbors
from sklearn.neighbors import KNeighborsClassifier

try: # different imports for different versions of scikit-learn
    from sklearn.model_selection import cross_val_score   # simpler cv this week
except ImportError:
    try:
        from sklearn.cross_validation import cross_val_score
    except:
        print("No cross_val_score!")


# %%
# Helper function for trainML
def cvTuning(X_train, Y_train, P=75, target=0.0001, avgLim=0.95, disp=1):
    ''' Function for using cross-validation (cv) to find the best K value for KNN classifier.

        @param X_train  [dataframe] features of training dataset
        @param Y_train  [dataframe] labels of training dataset
        @param P        [int] Proportional Control constant
        @param target   [double] Maximum difference in consecutive cv results
                                that will stop the loop (generally << 1)
        @param avgLim   [double] Minimum accuracy of cv results that
                                will stop the loop (value from 0-1)
    '''
    k = 50   # not likely to be a good value...
    run = 0
    avDiff = 0.4 # Fake initial diff to start loop
    av = 0.5 # Fake initial average to start loop
    while ((avDiff > target) & (av < avgLim)): # Once change in cv averages is minimal, stop
        k_old = k

        if disp == 2:
            print('Proportional control loop #{}'.format(run))
            print('Old k-value:', k_old)
        
        if run != 0:
            err = 1 - av # How far away are we from 100%?
            k -= math.ceil(P*err) # Ceiling so that minimum step size is 1
            if disp == 2: print('New k-value:', k)
        
        knn = KNeighborsClassifier(n_neighbors=k)   # here, k is the "k" in kNN

        # cross-validation
        #
        cv_scores = cross_val_score( knn, X_train, Y_train, cv=5 ) # cv is the number of splits
        av = cv_scores.mean()

        if disp == 2:
            print('\nthe cv_scores are')
            for s in cv_scores:
                # we format it nicely...
                s_string = "{0:>#7.4f}".format(s) # docs.python.org/3/library/string.html#formatexamples
                print("   ",s_string)
        
        if run == 0:
            avDiff = av
        else:
            avDiff = av-old_av

        old_av = av

        run += 1

    if disp:
        print('+++ with average: ', av)
        print()
        print('avDiff =', avDiff)

    if (avDiff < 0): # Means that the avg decreased in last iteration
        best_k = k_old   # Return to old k value
    else:
        best_k = k  # Keep most recent k value

    if disp:
        print('Best k:', best_k)

    return best_k


# %%
def trainML(dataFile, classifier, params={}):
    ''' Train a machine learning model on training data

            @param dataFile    [str] Filename of csv file with 
                                    training data
            @param classifier  [str] 2-letter flag that indicates 
                                    which classifier to use.
                                    'nn' = Neural Net
                                    'kn' = K-Nearest Neightbors
                                    'rf' = Random Forest
            @param params  [dict] Dictionary of additional
                                parameters (classifier-specific)            
    '''

    print("+++ Start of pandas' datahandling +++\n")
    df = pd.read_csv(dataFile, header=0) # read the file w/header row #0
    # df.head()                                 # first five lines
    # df.info()                                 # column details

    X_train = df.iloc[:,1:6]
    Y_train = df['Sound SBN']
    # Y_train = df['Sound SBN'].map(mf.num2Sound) # In case we need numbers instead of letters


    #
    # it's important to keep the input values in the 0-to-1 or -1-to-1 range
    #    This is done through the "StandardScaler" in scikit-learn
    # 
    USE_SCALER = False # Do we need this section?
    if USE_SCALER == True:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)   # Fit only to the training dataframe
        # now, rescale inputs -- both testing and training
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_unknown = scaler.transform(X_unknown)


    if classifier == 'nn':
        print('Using Neural Network classifier...')

        # Default parameters
        hls = (10,10) # Hidden layer sizes
        max_iter = 200
        alpha = 1e-4
        lri = 0.1 # Initial Learning rate

        # Set custom params if available
        if 'hidden_layer_sizes' in params.keys():
            hls = params['hidden_layer_sizes']
        if 'max_iter' in params.keys():
            max_iter = params['max_iter']
        if 'alpha' in params.keys():
            alpha = params['alpha']
        if 'learning_rate_init' in params.keys():
            lri = params['learning_rate_init']
        

        model = MLPClassifier(hidden_layer_sizes=hls, max_iter=max_iter, alpha=alpha,
                    solver='sgd', verbose=True, shuffle=True, early_stopping = False, # tol=1e-4, 
                    random_state=None, # reproduceability
                    learning_rate_init=lri, learning_rate = 'adaptive')
        model = model.fit(X_train, Y_train)

    elif classifier == 'kn':
        print('Using K-Neighbors classifier...')

        if not params: # Check if params were provided
            k_val = 3
        elif len(params) > 1:
            print('Starting parameter tuning (cross-validation)...')
            k_val = cvTuning(X_train, Y_train, P=params['P'], target=params['target'], avgLim=params['avgLim'], disp=params['disp'])
        else:
            k_val = params['k']
            print('Fixed k =', k_val)

        model = KNeighborsClassifier(n_neighbors=k_val)
        model = model.fit(X_train, Y_train)

    elif classifier == 'rf':
        print('Using Random Forest classifier...')

        MAX_DEPTH = params['max_depth']
        NUM_TREES = params['num_trees']
        model = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
        model = model.fit(X_train, Y_train)

        if params['feat_import']:
            print("\nrforest.feature_importances_ are\n      ", model.feature_importances_) 
            # print("Order:", feature_names[0:4])


    else:
        print('WARNING: No classifier provided. Defaulting to knn...')

        if not params: # Check if params were provided
            k_val = 3
        elif len(params) > 1:
            print('Starting parameter tuning (cross-validation)...')
            k_val = cvTuning(X_train, Y_train, P=params['P'], target=params['target'], avgLim=params['avgLim'], disp=params['disp'])
        else:
            k_val = params['k']

        model = KNeighborsClassifier(n_neighbors=k_val)
        model = model.fit(X_train, Y_train)

    
    return model


# %%

def main():

    # kn1 will run cross-validation to find a good k value
    kn1_params = {'P': 10, 'target': 0.0001, 'avgLim': 0.95, 'disp': 1}
    kn2_params = {'k': 18} # Best K for freestyle data

    rf_params = {'max_depth': 4, 'num_trees': 100, 'feat_import': True}
    nn_params = {'hidden_layer_sizes': (10,10)} # 'max_iter': 200, 'alpha': 1e-4, 'learning_rate_init': 0.1


    model = trainML('rec1_feats.csv', 'rf', params=rf_params)

    freestyle_df = pd.read_csv('beat1_feats.csv', header=0) # read the file w/header row #0
    X_unlabeled = freestyle_df.iloc[:,1:6]
    Y_unlabeled = freestyle_df['Sound SBN']

    pred_labels = model.predict(X_unlabeled)
    print('\nBeat 1 sound predictions')
    print(pred_labels[:32])
    print('')
    print(pred_labels[32:64])
    print('')
    print(pred_labels[64:96])


if __name__ == '__main__':
    main()

# %%