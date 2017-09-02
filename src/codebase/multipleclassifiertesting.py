import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split, KFold, cross_val_score
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix as confMat

class MultipleClassifierTesting():
    """
    A helper class to test multiple classifiers, find optimal hyper parameters using GridSearchCV,
    output all the results, as well as, save their confusion matrices for Bayesian Modell Combination.

    For more, check out GridSearchCV documentation:

    http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

    """

    def __init__(self, models, parameters):
        """
        When initialising a class, the inputs are:
            models - dictionary; {'RandomForestClassifier': RandomForestClassifier(),
                                  'SVC': SVC(),
                                  and etc..}
            paramaters - dictionary; {'RandomForestClassifier': { 'n_estimators': [16, 32] },
                                      'SVC': [
                                              {'kernel': ['linear'], 'C': [1, 10]},
                                              {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]}
                                              ]
                                     }
        """

        # Check if keys of models dictionary and keys of parameters are correct
        #assert set(models.keys()).issubset(set(parameters.keys())), \
        #            "These models are missing parameters {}".format(list(set(models.keys()) - set(elpparameters.keys())))

        # Instantiate class variables
        self.models = models
        self.parameters = parameters
        self.model_names = models.keys()
        # Dictionaries storing trained models
        self.grid_search_objects = {}
        self.no_grid_search = {}
        # rows to be appended for models without parameters
        self.other_rows = {}

    def fit(self, X, y, n_jobs=1, verbose=0, scoring='accuracy', refit=True):
        """
        Performs grid search CV for each of the models and saves the GridSearchCV trained object
        into a dictionary self.grid_search_objects.
            Inputs:
                X               - training data array
                y               - training data labels
                cv_fold_size    - number of K folds
                n_jobs          - # to paralelize
                verbose         - how much to print out into terminal, should be zero if run on cluster
                scoring         - scoring function to use
                refit           - should the best model be fit to a whole training set
        """
        # Set up leave one out cross-validation generator
        loo = KFold(n_splits=10)
        for model_name in self.model_names:
            # Get the model
            model = self.models[model_name]

            # check if model has parameters, i.e if empty then perform custom CV
            if not self.parameters[model_name]:
                scores = self.customCV(X, y, model, loo)
                self.other_rows[model_name] = scores
                # fit to full data and leave for saving
                model.fit(X, y)
                self.no_grid_search[model_name] = model
            else:
                # get parameters
                parameters = self.parameters[model_name]
                # Perform grid search cross-validation
                grid_search_object = GridSearchCV(model, parameters, cv=loo, n_jobs=n_jobs, verbose = verbose,
                                                 scoring=scoring, refit=refit)
                grid_search_object.fit(X, y)

                # Save grid_search_object into grid_search_objects dictionary
                self.grid_search_objects[model_name] = grid_search_object

        return True

    def customCV(self, X, y, model, loo):
        """
        Performs custom cv for models without parameters as GridSearchCV fails in those cases
        """
        scores = cross_val_score(model, X, y, cv=loo)

        return np.array(scores)


    def getConfusion(self, X, Y, n_splits, random_state, path=None, now=''):
        """
        Get estimates for the confusion matrices
        """
        # initialize a dictionary, storing array of confusion matrices for
        # each fold for each model as key
        self.conf_dict = {}

        # initialize kfold
        kf = KFold(n_splits=n_splits)

        for model_name in self.model_names:

            # add list to conf_dict
            self.conf_dict[model_name] = []

            # check if model had parameters
            if not self.parameters[model_name]:
                model = self.models[model_name]

                for train_index, test_index in kf.split(X):

                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]

                    model.fit(X_train, Y_train)

                    Y_pred = model.predict(X_test)

                    self.conf_dict[model_name].append(confMat(Y_test, Y_pred))
            else:
                # get best parameters estimated using GridSearchCV
                best_params = self.grid_search_objects[model_name].best_params_

                # get the model and initialize the classifier
                model = self.models[model_name]

                model.set_params(**best_params)

                # estimate confusion matrices
                for train_index, test_index in kf.split(X):
                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]

                    model.fit(X_train, Y_train)

                    Y_pred = model.predict(X_test)

                    self.conf_dict[model_name].append(confMat(Y_test, Y_pred))

        if path==None:
            with open('{}confMatrices.pickle'.format(now), 'wb') as f:
                pickle.dump(self.conf_dict, f)
        if path != None:
            filename = os.path.join(path, '{}confMatrices.pickle'.format(now))
            with open(filename, 'wb') as f:
                pickle.dump(self.conf_dict, f)


    def saveModels(self, path=None, now=''):
        """
        Saves trained grid_search_objects, as well as,
        separate dictionary of best model parameters
            Inputs:
                path - to save the grid_search_objects
        """
        parameters_dict = {}
        if path != None:
            for model_name in self.model_names:
                path_to = os.path.join(path, '{}{}pickledModel.pkl'.format(now, model_name))
                if not self.parameters[model_name]:
                    parameters_dict[model_name] = {}
                    joblib.dump(self.no_grid_search[model_name], path_to)
                else:
                    parameters_dict[model_name] = self.grid_search_objects[model_name].best_params_
                    joblib.dump(self.grid_search_objects[model_name], path_to)
            with open(os.path.join(path,'parameters_dict.pickle'), 'wb') as f:
                pickle.dump(parameters_dict, f)
        else:
            for model_name in self.model_names:
                if not self.parameters[model_name]:
                    parameters_dict[model_name] = {}
                    joblib.dump(self.no_grid_search[model_name], '{}{}pickledModel.pkl'.format(now, model_name))
                else:
                    parameters_dict[model_name] = self.grid_search_objects[model_name].best_params_
                    joblib.dump(self.grid_search_objects[model_name], '{}{}pickledModel.pkl'.format(now, model_name))
            with open('parameters_dict.pickle', 'wb') as f:
                pickle.dump(parameters_dict, f)

    def saveSummaryCSV(self, filename, sort_by="mean_score"):
        """
        Returns a dataframe with scores for each classifier
            Outputs:
                data frame object
        """
        ### Uses .grid_scores_ attribute, which is going to be deprecated in sklearn v0.20
        rows = []
        for model_name in self.model_names:
            if not self.parameters[model_name]:
                continue
            for grid_search_object in self.grid_search_objects[model_name].grid_scores_:
                rows.append(makeRow(model_name, grid_search_object.cv_validation_scores, grid_search_object.parameters))

        for model_name, scores in self.other_rows.items():
            rows.append(customRow(model_name, scores))
        df = pd.concat(rows, axis=1).T.sort_values(by=[sort_by], ascending=False)

        columns = ['model', 'min_score', 'max_score', 'mean_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        df = df[columns]
        df.to_csv(filename, sep=',')

    def replaceGSCVwithBest(self):
        """
        Replaces GridSearchCV objects with base_estimator_
        """

        for model_name in self.model_names:
            if not self.parameters[model_name]:
                continue
            best_estimator = self.grid_search_objects[model_name].best_estimator_
            self.grid_search_objects[model_name] = best_estimator

################################################################################
### HELPER FUNCTIONS
################################################################################

def makeRow(key, scores, params):
    """
    Makes row for results data frame
    """
    dic = {
            'model': key,
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
    }
    return pd.Series(dict(**params, **dic))

def customRow(key, scores):
    """
    Makes rows for no GridSearchCV models
    """
    dic = {
            'model': key,
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
    }
    return pd.Series(dict(**{}, **dic))

def testMultipleClassifierTesting():
    """
    Function to test the class defined above
    """
    from sklearn import datasets
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    models = {'RandomForestClassifier': RandomForestClassifier(),
            'SVC': SVC(), 'Logistic Reg': LogisticRegression()}

    parameters = {'RandomForestClassifier': {'n_estimators': [16, 32]},
                'SVC': [{'kernel': ['linear'], 'C':[1, 10]},
                        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]}
                        ],
                'Logistic Reg': {}
                }

    # Instantiate the models pipeline
    pipeline = MultipleClassifierTesting(models, parameters)

    # Get iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Perform Grid search on each
    pipeline.fit(X_train, y_train, verbose=0)
    # save the summary of the GridSearchCV
    pipeline.saveSummaryCSV(filename='test.csv')
    # Save models
    pipeline.saveModels()
    # perfrom confusion matrix estimation
    pipeline.getConfusion(X_train, y_train, n_splits=10, random_state=3)

if __name__=='__main__':
    # Test
    testMultipleClassifierTesting()
