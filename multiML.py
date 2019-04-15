# Multi-ML pipeline
# -----------------
# Machine Learning Pipeline with options for multiple different
# classifiers.

# %%
# Import Statements
import numpy as np            
import pandas as pd
# import math
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
# print("+++ Start of pandas' datahandling +++\n")
df = pd.read_csv('rec1_feats.csv', header=0)
# df.head()
# df.info()
print(df.iloc[0:2,1:9])

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

    X_train = df.iloc[:,1:9]
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

        # TODO: parameterize these settings
        model = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=200, alpha=1e-4,
                    solver='sgd', verbose=True, shuffle=True, early_stopping = False, # tol=1e-4, 
                    random_state=None, # reproduceability
                    learning_rate_init=.1, learning_rate = 'adaptive')
        model = model.fit(X_train, Y_train)

    elif classifier == 'kn':
        print('Using K-Neighbors classifier...')

        # TODO: Add cross-validation loop with parameterized settings

        if not params: # Check if params were provided
            k_val = 3
        else:
            k_val = params['k']

        model = KNeighborsClassifier(n_neighbors=k_val)
        model = model.fit(X_train, Y_train)

    elif classifier == 'rf':
        print('Using Random Forest classifier...')

        MAX_DEPTH = params['max_depth']
        NUM_TREES = params['num_trees']
        model = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
        model = model.fit(X_train, Y_train)

    else:
        print('WARNING: No classifier provided. Defaulting to knn...')

        if not params: # Check if params were provided
            k_val = 3
        else:
            k_val = params['k']

        model = KNeighborsClassifier(n_neighbors=k_val)
        model = model.fit(X_train, Y_train)

    
    return model


# %%

def main():

    model = trainML('rec1_feats.csv', 'nn', params={'k': 5})

    freestyle_df = pd.read_csv('freestyle_feats.csv', header=0) # read the file w/header row #0
    X_unlabeled = freestyle_df.iloc[:,1:9]
    Y_unlabeled = freestyle_df['Sound SBN']

    pred_labels = model.predict(X_unlabeled)
    print(pred_labels[:20])


if __name__ == '__main__':
    main()

# %%