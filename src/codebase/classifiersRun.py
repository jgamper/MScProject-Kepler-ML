import os
import sys
import pickle
import datetime
import copy
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, log_loss, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# import classifiers
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Multiple Classifier Testing
sys.path.append('..')
from multipleclassifiertesting import MultipleClassifierTesting as MCT

#####################################
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

def train(models, parameters, dataset_path_or_data, include_valid=False, save_summary=False, save_models=False, get_confusion=False, directory=None, now=None):

    if save_models or save_summary or get_confusion:
        assert directory is not None, "You want to do saving, but did not provide the directory"
        assert now is not None, "You want to do saving, but did not provide now"

    if isinstance(dataset_path_or_data, str):
        # Load dataset
        data = pd.read_csv(dataset_path_or_data)
        # data.drop(['Unnamed: 0'], axis=1, inplace=True)

        labelsEncoder = LabelEncoder()
        labels = data.label.values
        Y = labelsEncoder.fit_transform(labels)
        data.drop(['label', 'koi_dikco_mdec', 'koi_dicco_mra', 'tce_minmesd', 'tce_ptemp', 'tce_prad', 'tce_dikco_mra_err',\
        'tce_fwm_sdeco', 'tce_steff_err', 'boot_mesmean', 'tce_smet', 'tce_sradius_err', 'tce_dor', 'koi_dikco_msky_err', 'tce_time0bk', 'koi_fwm_prao_err', 'tce_duration', 'wst_depth', 'tce_mesmad', 'tce_ror', 'tce_dikco_msky', 'tce_ingress_err', 'tce_depth_err', 'tce_dof1'], axis=1, inplace=True)
        X = data.as_matrix()
        print(X.shape)
        # Get name of current run
        run_name = os.path.basename(__file__)
    else:
        X, Y = dataset_path_or_data

    print('Initializing the pipeline..')
    # Instantiate the multiple classifier testing
    pipeline = MCT(models, parameters)

    print('Splitting the data..')
    X, Y = shuffle(X, Y, random_state=32)

    # Split the data into train and validation set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=32, stratify=Y)
    # Split test set into validation and training
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    if include_valid == True:
        X_train = np.vstack((X_train, X_valid))
        y_train = np.concatenate((y_train, y_valid), axis=0)

    print('Fitting the models..')
    # Do gridsearchcv and fitting
    pipeline.fit(X_train, y_train, verbose=0, scoring='accuracy', refit=True)

    if save_summary == True:
        print('Saving summary..')
        # Save the summary
        filename = os.path.join(directory, 'summary_{}.csv'.format(now))
        pipeline.saveSummaryCSV(filename=filename)

    if save_models == True:
        print('Saving models..')
        # Save models
        pipeline.saveModels(path=directory, now=now)

    if get_confusion == True:
        if include_valid == True:
            print('Getting confusion matrices')
            # Save confusion matrices
            pipeline.getConfusion(X_train, y_train, random_state=42, n_splits=10, path=directory, now=now)
        else:
            X_train = np.vstack(X_train, X_valid)
            y_train = np.vstack(y_train, y_valid)
            print('Getting confusion matrices')
            # Save confusion matrices
            pipeline.getConfusion(X_train, y_train, random_state=42, n_splits=10, path=directory, now=now)

    if 'labelsEncoder' in locals():
        return pipeline, X_train, y_train, X_valid, y_valid, X_test, y_test, labelsEncoder
    else:
        return pipeline, X_train, y_train, X_valid, y_valid, X_test, y_test

def test(pipeline, X_test, y_test):
    """
    Tests the model on a test set, returns plot dictionary
    Returns:
                plot_dict - dictionary for roc plot for each model
                cnf_matrices - confusion matrix of the results
    """
    models = dict(pipeline.grid_search_objects, **pipeline.no_grid_search)
    plot_dict = {}
    cnf_matrices = {}
    for model_name, model in models.items():

        if hasattr(model, "predict_proba"):
            probas_ = model.predict_proba(X_test)
            probas_ = probas_[:,1]
            fpr, tpr, thresholds = roc_curve(y_test, probas_)
        else:
            probas_ = model.decision_function(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, probas_)
            probas_ = (probas_ - probas_.min()) / (probas_.max() - probas_.min())

        y_pred = model.predict(X_test)

        roc_auc = auc(fpr, tpr)

        logloss = log_loss(y_test, y_pred)

        precision = precision_score(y_test, y_pred)

        recall = recall_score(y_test, y_pred)

        brier_score = brier_score_loss(y_test, probas_)

        cnf_matrices[model_name] = confusion_matrix(y_test, y_pred)

        plot_dict[model_name] = [fpr, tpr, roc_auc, logloss, precision, recall, brier_score]

    return plot_dict, cnf_matrices

def calibrationTest(pipeline, X_valid, y_valid):
    """
    Produces the plots to evaluate the reliability of the classifiers
    Returns:
            plot_dict
    """
    models = dict(pipeline.grid_search_objects, **pipeline.no_grid_search)
    plot_dict = {}

    for model_name, model in models.items():

        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_valid)[:, 1]
        else:
            prob = model.decision_function(X_valid)
            prob = (prob - prob.min()) / (prob.max() - prob.min())

        fraction_of_pos, mean_predicted_val = calibration_curve(y_valid, prob, n_bins=10)

        plot_dict[model_name] = [fraction_of_pos, mean_predicted_val, prob]

    return plot_dict

def calibrate(pipeline, calibration_specifics, X_valid, y_valid):
    """
    Calibrates classifiers according calibration_specifics,
    returns updated calibrated pipeline
    """
    calibrated_pipeline = {}
    for model_name in pipeline.model_names:
        if not pipeline.parameters[model_name]:
            # then it is in no_grid_search
            flag = True
            model = pipeline.no_grid_search[model_name]
        else:
            model = pipeline.grid_search_objects[model_name]

        model_copy = copy.deepcopy(model)
        calibrated = CalibratedClassifierCV(model_copy, method=calibration_specifics[model_name], cv='prefit')
        calibrated.fit(X_valid, y_valid)
        if 'flag' in locals():
            if flag == True:
                calibrated_pipeline[model_name] = calibrated
                flag = False
        else:
            calibrated_pipeline[model_name] = calibrated

    return calibrated_pipeline


def wholeDataCV(X, Y, classifier, n_splits, method='isotonic'):
    """
    Performs CV fitting, calibration and testing with a given classifier. All the testing predictions are
    returned for a fair comparison with VESPA results.
    Inputs:
        X - training data, dataframe
        Y - labels, dataframe
    Outputs:
        CV_scores - a list of scores of the classifier
        Y_probs - dataframe of probabilities assigned to TCEs
    """
    y_copy = Y.copy(deep=True).values
    y_source = pd.DataFrame(y_copy, index=Y.index)
    x_copy = X.copy(deep=True)
    result = []

    roc_auc_list, logloss_list, precision_list, recall_list, brier_score_list = [], [], [], [], []

    # Does a deep copy of the classifier, along with best parameters, but without fitting to data
    classifier_clone = clone(classifier)

    # Split data into n_splits, and use 2 for validation and 1 for testing, save testing results into dataframe
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=3)

    for train_index, test_index in skf.split(x_copy, y_copy):
        xx_train, xx_test = x_copy.iloc[train_index], x_copy.iloc[test_index]

        yy_train, yy_test = pd.DataFrame(y_source.iloc[train_index].values,
                                         index=y_source.iloc[train_index].index),\
                            pd.DataFrame(y_source.iloc[test_index].values,
                                         index=y_source.iloc[test_index].index)

        if method != 'None':
            xx_train, xx_valid, yy_train, yy_valid = train_test_split(xx_train, yy_train, test_size=0.20,
                                                                     random_state=2)

            classifier_clone.fit(xx_train, yy_train)


            calibrated = CalibratedClassifierCV(classifier_clone, method=method, cv='prefit')

            calibrated.fit(xx_valid, yy_valid)



            if hasattr(calibrated, "predict_proba"):
                probas_ = calibrated.predict_proba(xx_test)[:,1]
                probas_ = pd.Series(probas_, index=xx_test.index.values)
                fpr, tpr, thresholds = roc_curve(yy_test, probas_.values)
            else:
                probas_ = calibrated.decision_function(xx_test)
                fpr, tpr, thresholds = roc_curve(yy_test, probas_)
                probas_ = (probas_ - probas_.min()) / (probas_.max() - probas_.min())
                probas_ = pd.Series(probas_, index=xx_test.index.values)

        else:
            classifier_clone.fit(xx_train, yy_train)
            calibrated = classifier_clone

            if hasattr(calibrated, "predict_proba"):
                probas_ = calibrated.predict_proba(xx_test)[:,1]
                probas_ = pd.Series(probas_, index=xx_test.index.values)
                fpr, tpr, thresholds = roc_curve(yy_test, probas_.values)
            else:
                probas_ = calibrated.decision_function(xx_test)
                fpr, tpr, thresholds = roc_curve(yy_test, probas_)
                probas_ = (probas_ - probas_.min()) / (probas_.max() - probas_.min())
                probas_ = pd.Series(probas_, index=xx_test.index.values)

        yy_pred = calibrated.predict(xx_test)

        roc_auc_list.append(auc(fpr, tpr))

        logloss_list.append(log_loss(yy_test, yy_pred))

        precision_list.append(precision_score(yy_test, yy_pred))

        recall_list.append(recall_score(yy_test, yy_pred))

        brier_score_list.append(brier_score_loss(yy_test, probas_))

        result.append(probas_)

    indexes = [series.index.values for series in result]
    indexes = np.concatenate(indexes)
    probas = [series.values for series in result]
    probas = np.concatenate(probas)

    result = pd.Series(probas, index=indexes)

    return result, [roc_auc_list, logloss_list, precision_list, recall_list, brier_score_list]

def main():
    """
    Testing script
    """
    from sklearn import datasets
    import matplotlib.pyplot as plt

    models = {'QDA': QuadraticDiscriminantAnalysis(),
            'Decision Tree': DecisionTreeClassifier()}

    parameters = {'Decision Tree': { 'max_depth' : [10, 20] },
                  'QDA':{}}

    cancer = datasets.load_breast_cancer()
    X = cancer.data
    Y = cancer.target

    pipeline, X_train, y_train, X_valid, y_valid, X_test, y_test = train(models, parameters, dataset_path_or_data=(X, Y), include_valid=False)

    pipeline.replaceGSCVwithBest()

    calib_test = calibrationTest(pipeline, X_valid, y_valid)

    plot_dict, cnf_matrices = test(pipeline, X_test, y_test)

    calibration_specs = {'QDA':'isotonic', 'Decision Tree':'isotonic'}
    calibrated_pipeline = calibrate(pipeline, calibration_specs, X_valid, y_valid)
    plot_dict, cnf_matrices = test(calibrated_pipeline, X_test, y_test)


################################################################################
if __name__ == '__main__':

    main()
