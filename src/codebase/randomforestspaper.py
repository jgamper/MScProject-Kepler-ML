
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
import logging
import numpy as np
import random

def randomForestOOBFeatureImportanceCC(num_of_trees, X, Y, random_state):
    """
    Computes Class Conditional feature importance, using OOB score
    Input:
        num_of_trees - number of trees for RF
        X            - matrix of shape (# samples, # attributes); training data
        Y            - array of shape (# samples); target labels; binary in our case
        random_state - random state for Random Forest
    Output:
        an array of shape (# 1, # attributes); mean attribute importance defined by OOB difference
        between permuted attribute and actuall attribute OOB score

    Alternative methods to consider:
        "Permutation Importance" - Genuer et. al. 2010 [https://hal.archives-ouvertes.fr/hal-00755489/file/PRLv4.pdf],
        "OOB Randomization"      - Hastie et. al. 2009,
        "Wavelet-based VI"       - Elisha & Dekel 2016 [http://www.jmlr.org/papers/volume17/15-203/15-203.pdf]

    "Traditional" methods:
        Node impurity, by Gini Index or IG (in case of regression - residual sum of squares).
        It is common to multiply the IG of node by its size (Raileanu and Stoffel 2004),
        (Du and Zhan 2002), (Rokach and Maimon 2005).

    Cons:
        Both ‘Impurity gain’ and ‘Permutation’ have some pitfalls.
        As shown by (Strobl et. al. 2006), the‘Impurity gain’ tends to be in favor
        of variables with more varying values. As shown in (Strobl et. al. 2008),
        ‘Permutation’ tends to overestimate the variable importance of highly
        correlated variables.
    """

    # Create an array of shape (# of trees, # of features) to store OOB error differences
    array_of_OOB_error_diffs = np.zeros((num_of_trees, X.shape[1]))

    # Train the Random Forest
    clf = RandomForestClassifier(warm_start=True, max_features=None, oob_score=True, random_state=random_state, \
                                 n_estimators=num_of_trees)
    clf.fit(X, Y)

    # Iterate over each column in X
    for column_id in range(X.shape[1]):

        # Iterate over each tree in Random Forest
        for tree_id, tree in enumerate(clf.estimators_):

            # Get indices of the OOB dataset for a given tree
            unsampled_indices = _generate_unsampled_indices(tree.random_state, X.shape[0])

            # Get OOB X and Y
            y_local = Y[unsampled_indices].copy()

            x_local = X[unsampled_indices, :].copy()

            class_OOB = []
            for cl in np.unique(y_local):

                # Extract indicies for a given class
                cl_ind = np.where(y_local == cl)[0]

                # Extract the corresponding Y and X rows
                y_local_class = y_local[cl_ind].copy()
                x_local_class = x_local[cl_ind, :].copy()
                test = x_local_class.copy()

                # Compute class OOB error
                OOB_error = 1 - tree.score(x_local_class, y_local_class)

                # Compute OOB error given a permuted feature
                x_local_class[:,column_id] = random.sample(list(x_local_class[:,column_id]),
                                                           len(x_local_class[:,column_id])) # permute ith feature

                OOB_error_permuted = 1 - tree.score(x_local_class, y_local_class)

                # Compute difference
                diff = OOB_error_permuted - OOB_error

                # Append diff to class_OOB
                class_OOB.append(diff)

            # Store the difference for all the trees
            array_of_OOB_error_diffs[tree_id, column_id] = max(class_OOB)

    # return the importance of each feature by computing the mean over each tree, i.e. over axis = 1
    return np.mean(array_of_OOB_error_diffs, axis = 0) #, clf

def randomAttributeImportance(num_of_trees, X, Y, random_state, num_of_rf=150):
    """
    Computes the importance of a randomly generated attribute (generated from normal dist with param (0, 1))
        Input:
            num_of_trees       - number of trees per random forest
            num_of_rf          - number of random forests to fit
            X                  - an array of shape (# samples, # attributes); training data
            Y                  - an array of shape # samples; labels for training data - binary
            random_state       - random state
        Output:
            ra_importance - an array of size num_of_rf; a distribution of importance for random attribute

    According to the paper a cut off, was chosen to be 6 std_dev above zero.
    """

    # Pre-allocate an array
    ra_importance = np.zeros(num_of_rf+1)

    i = 0

    for i in range(num_of_rf):

        # Add random column
        local_X = X.copy()

        random_column = np.random.normal(size=X.shape[0])

        local_X = np.column_stack((local_X, random_column))

        # Get random column index
        random_column_index = local_X.shape[1] - 1

        # Compute the importance using randomForestOOBFeatureImportanceCC()
        mean_importance = randomForestOOBFeatureImportanceCC(num_of_trees=num_of_trees, X=local_X, Y=Y,
                                                                  random_state=random_state)
        # Extract the importance of random feature
        ra_importance[i] = mean_importance[random_column_index]

    return ra_importance

def removeCorrAtributes(pearson_corr_threshold, cc_attribute_importance, X, column_names):
    """
    Removes one of the significantly correlated attributes given the threshold, which
    attribute to remove is judged by the class conditional attribute importance.
        Input:
            pearson_corr_threshold  - type float; correlation threshold for significane determination
            cc_attribute_importance - type array of size # attributes in X
            X                       - type array of size (# samples, # attributes)
            column_names            - list of column names
        Output:
            X_output                - filtered X
            column_names            - filtered column names list
    """

    # Compute Pearson's correlation matrix
    corr_mat = np.corrcoef(X, rowvar=0)

    # Get all unique comparisons
    # List of tuples
    ind_unique = list(zip(np.triu_indices(corr_mat.shape[0])[0], np.triu_indices(corr_mat.shape[0])[1]))

    # Exclude diagonals
    ind_unique = [tup for tup in ind_unique if tup not in [(i,i) for i in range(corr_mat.shape[0])]]

    # Initialize a list of indices to be deleted
    to_delete = []

    # Check each unique pair in ind_unique
    for tup in ind_unique:

        i, j = tup

        # Check correlation coef
        if corr_mat[i, j] > pearson_corr_threshold:

            # Compare importance
            if cc_attribute_importance[i] > cc_attribute_importance[j]:
                to_delete.append(j)

            else:
                to_delete.append(i)


    # Delete corresponding columns
    X_output = np.delete(X, to_delete, axis=1)
    column_names = np.delete(column_names, to_delete, axis=0)
    return X_output, column_names

def corrThresholdRange(X, Y, min_max_inc, random_state, num_of_trees, cc_attribute_importance, column_names):
    """
    Computes the range of OOB errors for RF given varying correlation thresholds
        Input:
            num_of_rf          - number of random forests to fit
            X                  - an array of shape (# samples, # attributes); training data
            Y                  - an array of shape # samples; labels for training data - binary
            random_state       - random state
            min_max_inc        - tuple of min, max threshold and increment
            cc_attribute_importance - type array of size # attributes in X
            column_names            - list of column names
        Output:
            rf_oob_array - an array of oob errors for a range of thresholds
    """
    thresh_range = np.arange(min_max_inc[0], min_max_inc[1], min_max_inc[2])
    rf_oob_array = np.zeros(thresh_range.shape[0])
    for i, thresh in enumerate(thresh_range):

        X_output, column_names = removeCorrAtributes(thresh, cc_attribute_importance, X, column_names)

        clf = RandomForestClassifier(warm_start=True, max_features=None, oob_score=True, random_state=random_state, \
                                     n_estimators=num_of_trees)
        clf.fit(X_output, Y)

        oob_error = 1 - clf.oob_score_
        rf_oob_array[i] = oob_error

    return rf_oob_array
