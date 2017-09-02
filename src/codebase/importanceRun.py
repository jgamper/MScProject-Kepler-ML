###############################################################################

## Runs feature importance testing according to methods implemented in
## McCauliff, Jenkins "Automatic Claassification of Kepler Planetary Transit Candidates"
## Saves the selected features data into a file named importanceRunMDHM.csv
## Saves the importance values into file importanceDictMDHM.pickle, along with RA importance

###############################################################################
import os
import operator
import pickle
import datetime
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import implemented feature selection methods
sys.path.append('..')
from randomforestspaper import corrThresholdRange, removeCorrAtributes, randomAttributeImportance, \
randomForestOOBFeatureImportanceCC

def pickleDump(now, variable, directory):
    target = os.path.join(directory, now +'.pickle')
    with open(target, 'wb') as f:
        pickle.dump(variable, f)

def makeOutputDir(path, run_name):
    """
    Makes directory to store outputs of the current run
    Returns path where all output will be saved
    """
    now = datetime.datetime.now()
    now = 'M' + str(now.month) + 'D' + str(now.day) + 'H' + str(now.hour) + 'M' + str(now.minute)
    run_name = run_name + '_' + now
    directory = os.path.join(path, run_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory, now

def classCondMeans(data):
    means = data.groupby(['koi_disposition']).mean()
    for col_index, col in enumerate(data):
        for index, row in enumerate(data[col].values):
            if pd.isnull(row):
                if data.koi_disposition.values[index] == 'FALSE POSITIVE':
                    data.ix[index, col_index] = means.loc['FALSE POSITIVE', col]
                if data.koi_disposition.values[index] == 'CONFIRMED':
                    data.ix[index, col_index] = means.loc['CONFIRMED', col]
    return data

def main(data_path):

    # Get name of current run
    run_name = os.path.basename(__file__)

    # Test if directory already exists, return directory and now prefix
    path = '/home/jevgenji/Dropbox/Studies/MscProject/Kepler/KeplerDataInvestigation/outputs'
    directory, now = makeOutputDir(path, run_name)

    # Load data
    data = pd.read_csv(data_path, sep=',')
    # Keep kepid_pnlt for later
    kepid_pnlt = data.kepid_pnlt.values

    data.drop(['kepid_pnlt','Unnamed: 0', 'koi_fwm_sra', 'tce_minmes', 'tce_slogg_err'], axis=1, inplace=True)

    # Class cond means to plug in for missing values
    data = classCondMeans(data)

    # Get kepid_plnt and original labels
    labels_orig = data.koi_disposition.values

    # Encode labels
    labelsEncoder = LabelEncoder()
    Y = labelsEncoder.fit_transform(labels_orig)
    X = data.drop(['koi_disposition'], axis=1)

    # Get column names, and convert X into matrix
    col_names = list(X.columns.values)
    X = X.as_matrix()

    # Test feature importance and store the dictionary of importance of each feature
    print('Feature Importance running..')
    imp_array = randomForestOOBFeatureImportanceCC(100, X, Y, 3)
    imp_dict = {col_names[i]: imp_array[i] for i in range(len(imp_array))}
    imp_sorted = sorted(imp_dict.items(), key=operator.itemgetter(1))

    # Test random attribute importance, and get threshold
    print('RA Importance running..')
    ra_importance = randomAttributeImportance(100, X, Y, 3, 50)
    ra_threshold = np.std(ra_importance)*6

    # Format data
    X = pd.DataFrame(X, columns = col_names)
    imp_array = [value for value in imp_dict.values() if value > ra_threshold]

    # Get the column names below threshold of importance
    col_names = [key for key,value in imp_dict.items() if value < ra_threshold]
    X.drop(col_names, axis=1, inplace=True)
    col_names = X.columns.values
    X = X.as_matrix()

    # Select threshold for correlation filter
    min_max_inc = [0.15, 1.0, 0.05]
    print('Correlation Threshold running..')
    rf_oob_array = corrThresholdRange(X, Y, min_max_inc, 3, 100, imp_array, col_names)

    thresh = np.arange(0.15, 1.0, 0.05)[np.argmin(rf_oob_array)]

    # Get the final data
    print('Removing Correlated Attributes ..')
    X_final, col_names = removeCorrAtributes(thresh, imp_array, X, col_names)
    X_final = pd.DataFrame(X_final, columns=col_names)
    X_final['label'] = labelsEncoder.inverse_transform(Y)
    X_final['kepid_plnt'] = kepid_pnlt

    # Dump all the stuff before testing classifiers
    print('Dumping all ..')
    filename = os.path.join(directory, 'classificationReadyJul{}.csv'.format(now))
    X_final.to_csv(filename, sep=',', index_label=False, index=False)
    pickleDump(now, [imp_sorted, ra_threshold, rf_oob_array, thresh, X_final], directory)

if __name__ == '__main__':
    data_path = '/home/jevgenji/Dropbox/Studies/MscProject/Kepler/fp_conf_tableMonJul21.csv'
    main(data_path)
