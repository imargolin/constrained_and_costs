import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Union

from sklearn.feature_selection import mutual_info_classif

from .classes import Constraint, AbsoluteConstraint, RelativeConstraint, Budget
from .logger import get_logger
from .logger import RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, END
logger = get_logger(__name__)

def _prediction_up_to_constraint(y_pred_probs,constraint):
    D={}
    for val in y_pred_probs:
        D[val] = D.get(val,0) +1
    dictionary_items = D.items()
    sorted_D = sorted(dictionary_items,reverse=True)
    sorted_k = [sorted_D[i][0] for i in range(len(sorted_D))]
    #np.random.seed(23)
    number_of_positive = 0
    y_pred = np.zeros(len(y_pred_probs))
    classified = []
    for s_i in sorted_k:
        for i,y in enumerate(y_pred_probs):
            if number_of_positive >= constraint:
                return y_pred
            if i in classified:
                continue
            if y >= s_i:
                y_pred[i] = 1
                classified.append(i)
                number_of_positive +=1
    return y_pred

def _get_dynamic_threshold(y_pred_probs,constraint, t):
    """Explanation about the function:
    y_pred_probs: the predicted probabilities of the positive class
    constraint: the number of positive instances that we want to predict
    t: the threshold
    """
    D={} # dictionary to count the number of instances for each probability
    for val in y_pred_probs:
        D[val] = D.get(val,0) +1
    dictionary_items = D.items()

    sorted_D = sorted(dictionary_items,reverse=True)
    sorted_k = [sorted_D[i][0] for i in range(len(sorted_D))]

    

    np.random.seed(23)
    number_of_positive = 0 # number of positive instances that we have predicted
    y_pred = np.zeros(len(y_pred_probs)) # the predicted labels
    classified = [] # the indices of the instances that we have already classified
    dynamic_t = t # the dynamic threshold
    #print("sorted_k", sorted_k)
    #print("buga", buga)
    for s_i in sorted_k:
        # print("number_of_positive", number_of_positive)
        # print("dynamic_t", dynamic_t)
        # print("s_i", s_i)
        # print(classified)
        print("s_i", s_i)
        print("dynamic_t", dynamic_t)



        if s_i < t: # if the probability is less than the threshold, we stop
            return dynamic_t
        for i,y in enumerate(y_pred_probs): # for each instance
            if number_of_positive >= constraint: # if we have already predicted the number of positive instances that we want 
                dynamic_t = s_i
                return dynamic_t
            if i in classified:
                continue #continue means that we go to the next iteration
            if y >= s_i:
                y_pred[i] = 1
                classified.append(i)
                number_of_positive +=1
        
    return dynamic_t

def prediction_up_to_constraint(y_pred: np.ndarray, 
                                #The constraint could be either float or int
                                constraint: Union[float, int]):
    
    constraint = _validate_constraint(constraint, len(y_pred))

    out = pd.Series(y_pred) # convert to pandas series
    n_largest = out.nlargest(constraint).index # n largest indices
    out = out.index.isin(n_largest).astype(float) # 1 if index is in n_largest, 0 otherwise
    return out

def get_dynamic_threshold(y_preds_prob: np.ndarray, 
                          constraint: Union[int, float], 
                          t: float):
    """
    Explanation about the function:
    y_pred_probs: the predicted probabilities of the positive class
    constraint: The ratio of the positive instances that we want to predict
    t: the original threshold.
    """

    constraint = _validate_constraint(constraint, len(y_preds_prob))
    assert 0 <= t <= 1.0, "t must be a number between 0 and 1"

    y_preds_prob = pd.Series(y_preds_prob)
    y_preds_prob = y_preds_prob[y_preds_prob>=t]
    all_classified = y_preds_prob.sort_values(ascending=False).head(constraint)

    if len(all_classified) < constraint:
        dynamic_t = t
    else:
        dynamic_t = all_classified.iloc[-1]

    return dynamic_t

def find_effective_threshold(y_preds_prob: np.ndarray, 
                             constraint: Union[float, int], 
                             t: float):
    
    assert 0 <= constraint <= 1.0, "constraint must be a number between 0 and 1"
    assert 0 <= t <= 1.0, "t must be a number between 0 and 1"
    constraint = int(len(y_preds_prob) * constraint) # number of positive instances that we want to predict

    y_preds_prob = pd.Series(y_preds_prob).sort_values(ascending=False).head(constraint)
    t_last = y_preds_prob.iloc[-1]
    return max(t_last, t)

def prepare_for_cost_cle(n, cost_matrix):
    """
    Utility function for creating the matrix for the CSDecisionTreeClassifier.
    This is bascially a workaround because their implementation is stupid.
    """
    out = np.zeros((n, 4))
    out[:, 0] = cost_matrix[0, 1]
    out[:, 1] = cost_matrix[1, 0]
    return out

def filter_only_worst_features(X: pd.DataFrame, 
                            y: pd.Series, 
                            features_ratio: float):

    """
    X: the dataset
    y: the labels
    features_ratio: the ratio of the features that we want to keep, will keep only the features with the worst mutual information
    for example, if features_ratio = 0.25, it will keep only the features with the 25% worst mutual information
    """

    X_out = X.copy()

    binary_features = X.nunique(axis=0)<=2
    mutual_info = mutual_info_classif(X, y, discrete_features=binary_features, n_neighbors=3, copy=True, random_state=41)
    mutual_info = pd.Series(data=mutual_info, index=X.columns)
    q = mutual_info.quantile(features_ratio) #The quantile we want to keep
    X_out = X.loc[:, mutual_info<=q]
    return X_out

def _validate_constraint(constraint: Union[float, int], 
                         size: int) -> int:
    
    if isinstance(constraint, int):
        assert constraint >= 1, "constraint must be a positive integer"
    
    if isinstance(constraint, float):
        assert 0 <= constraint <= 1.0, "constraint must be a number between 0 and 1"
        constraint = int(size * constraint)
    return constraint

def benchmark_algorithm(y_pred: pd.Series,
                        group_ids: pd.Series | None, 
                        constraint: Constraint):
    # y_test: the true labels
    # y_pred: the predict probabilities
    # provider_id: the provider id for each row
    # constraints: the constraints dictionary, should be relative constraints.
    #value counts provider ids

    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    
    #assert unique index in y_pred and group ids
    assert y_pred.index.is_unique, "y_pred must have unique index"
    if group_ids is not None:
        if isinstance(group_ids, np.ndarray):
            group_ids = pd.Series(group_ids)
        assert group_ids.index.is_unique, "group_ids must have unique index"
        assert len(group_ids) == len(y_pred), "group_ids and y_pred must have the same length"
        assert all(group_ids.index == y_pred.index), "group_ids and y_pred must have the same index"

    df = pd.DataFrame({"y_pred": y_pred, "group_ids": group_ids})
    if isinstance(constraint, RelativeConstraint):
        group_counts = df["group_ids"].value_counts(dropna=False).to_dict()
        constraint = constraint.convert_to_absolute(group_counts)

    budget = constraint.as_budget() #We start with the absolute constraints

    df["outputs"] = 0 #We start with all zeros
    
    #iterate the dataframe, from the highest to the lowest probability
    for i, row in df.sort_values(by="y_pred", ascending=False).iterrows():
        group_id = row["group_ids"]
        if budget.has_enough(group_id) and row["y_pred"]>=0.5:
            budget.pay(group_id)
            df.loc[i, "outputs"] = 1 

    return df["outputs"].values

def is_satisfy_constraint(
        y_pred: pd.Series, 
        constraint: Constraint, 
        group_ids: pd.Series | None=None) -> bool:
    """
    y_pred: decisions of the classifier
    group_ids: group ids of the instances
    constraint: the constraint that we want to satisfy
    """

    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    
    #assert unique index in y_pred and group ids
    assert y_pred.index.is_unique, "y_pred must have unique index"
    if group_ids is not None:
        if isinstance(group_ids, np.ndarray):
            group_ids = pd.Series(group_ids)
        assert group_ids.index.is_unique, "group_ids must have unique index"
        assert len(group_ids) == len(y_pred), "group_ids and y_pred must have the same length"
        assert all(group_ids.index == y_pred.index), "group_ids and y_pred must have the same index"
    
    df = pd.DataFrame({"y_pred": y_pred, "group_ids": group_ids})
    
    group_counts = df["group_ids"].value_counts(dropna=False).to_dict()
    constraint = constraint.convert_to_absolute(group_counts)

    #Validating the constraint
    if y_pred.sum() > constraint.global_constraint:
        return False
    if constraint.local_constraints is not None:
        for group in constraint.local_constraints:
            if df[df["group_ids"]==group]["y_pred"].sum() > constraint.local_constraints[group]:
                return False
    return True


def say_hello():
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")    