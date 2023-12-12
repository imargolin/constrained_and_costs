
"""
This module include the cost sensitive decision tree method.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import copy
import sys
import time

from tqdm import trange
import six
import numbers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from .metrics import cost_loss
from .utils import prediction_up_to_constraint, get_dynamic_threshold, prepare_for_cost_cle, find_effective_threshold
from .utils import is_satisfy_constraint

from .classes import Constraint, RelativeConstraint
from .logger import get_logger

from xgboost import XGBClassifier

logger = get_logger(__name__)
logger.setLevel("INFO")


class CostSensitiveDecisionTreeClassifier(BaseEstimator):
    """A example-dependent cost-sensitive binary decision tree classifier.

    Parameters
    ----------
    criterion : string, optional (default="direct_cost")
        The function to measure the quality of a split. Supported criteria are
        "direct_cost" for the Direct Cost impurity measure, "pi_cost", "gini_cost",
        and "entropy_cost".

    criterion_weight : bool, optional (default=False)
        Whenever or not to weight the gain according to the population distribution.

    num_pct : int, optional (default=100)
        Number of percentiles to evaluate the splits for each feature.



    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
          - If int, then consider `max_features` features at each split.
          - If float, then `max_features` is a percentage and
            `int(max_features * n_features)` features are considered at each
            split.
          - If "auto", then `max_features=sqrt(n_features)`.
          - If "sqrt", then `max_features=sqrt(n_features)`.
          - If "log2", then `max_features=log2(n_features)`.
          - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
        Ignored if ``max_samples_leaf`` is not None.

    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    min_samples_leaf : int, optional (default=1)
        The minimum number of samples required to be at a leaf node.

    min_gain : float, optional (default=0.001)
        The minimum gain that a split must produce in order to be taken into account.

    pruned : bool, optional (default=True)
        Whenever or not to prune the decision tree using cost-based pruning

    Attributes
    ----------
    `tree_` : Tree object
        The underlying Tree object.

    See also
    --------
    sklearn.tree.DecisionTreeClassifier

    References
    ----------

    .. [1] Correa Bahnsen, A., Aouada, D., & Ottersten, B.
           `"Example-Dependent Cost-Sensitive Decision Trees. Expert Systems with Applications" <http://albahnsen.com/files/Example-Dependent%20Cost-Sensitive%20Decision%20Trees.pdf>`__,
           Expert Systems with Applications, 42(19), 6609–6619, 2015,
           http://doi.org/10.1016/j.eswa.2015.04.042

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.cross_validation import train_test_split
    >>> from costcla.datasets import load_creditscoring1
    >>> from costcla.models import CostSensitiveDecisionTreeClassifier
    >>> from costcla.metrics import savings_score
    >>> data = load_creditscoring1()
    >>> sets = train_test_split(data.data, data.target, data.cost_mat, test_size=0.33, random_state=0)
    >>> X_train, X_test, y_train, y_test, cost_mat_train, cost_mat_test = sets
    >>> y_pred_test_rf = RandomForestClassifier(random_state=0).fit(X_train, y_train).predict(X_test)
    >>> f = CostSensitiveDecisionTreeClassifier()
    >>> y_pred_test_csdt = f.fit(X_train, y_train, cost_mat_train).predict(X_test)
    >>> # Savings using only RandomForest
    >>> print(savings_score(y_test, y_pred_test_rf, cost_mat_test))
    0.12454256594
    >>> # Savings using CSDecisionTree
    >>> print(savings_score(y_test, y_pred_test_csdt, cost_mat_test))
    0.481916135529
    """
    def __init__(self,
                 criterion='direct_cost',
                 criterion_weight=False,
                 num_pct=100,
                 max_features=None,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_gain=0.001,
                 pruned=True,
                 ):

        self.criterion = criterion
        self.criterion_weight = criterion_weight
        self.num_pct = num_pct
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.pruned = pruned

        self.n_features_ = None
        self.max_features_ = None

        self.tree_ = []

    def set_param(self, attribute, value):
        setattr(self, attribute, value)

    def _node_cost(self, y_true, cost_mat):
        """ Private function to calculate the cost of a node.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        tuple(cost_loss : float, node prediction : int, node predicted probability : float)

        """
        n_samples = len(y_true)

        # Evaluates the cost by predicting the node as positive and negative
        costs = np.zeros(2)
        costs[0] = cost_loss(y_true, np.zeros(y_true.shape), cost_mat)
        costs[1] = cost_loss(y_true, np.ones(y_true.shape), cost_mat)

        pi = np.array([1 - y_true.mean(), y_true.mean()])

        if self.criterion == 'direct_cost':
            costs = costs
        elif self.criterion == 'pi_cost':
            costs *= pi
        elif self.criterion == 'gini_cost':
            costs *= pi ** 2
        elif self.criterion in 'entropy_cost':
            if pi[0] == 0 or pi[1] == 0:
                costs *= 0
            else:
                costs *= -np.log(pi)

        y_pred = np.argmin(costs)

        # Calculate the predicted probability of a node using laplace correction.
        n_positives = y_true.sum()
        y_prob = (n_positives + 1.0) / (n_samples + 2.0)

        return costs[y_pred], y_pred, y_prob

    def _calculate_gain(self, cost_base, y_true, X, cost_mat, split):
        """ Private function to calculate the gain in cost of using split in the
         current node.

        Parameters
        ----------
        cost_base : float
            Cost of the naive prediction

        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        split : tuple of len = 2
            split[0] = feature to split = j
            split[1] = where to split = l

        Returns
        -------
        tuple(gain : float, left node prediction : int)

        """

        # Check if cost_base == 0, then no gain is possible
        #TODO: This must be check in _best_split
        if cost_base == 0.0:
            return 0.0, int(np.sign(y_true.mean() - 0.5) == 1)  # In case cost_b==0 and pi_1!=(0,1)

        j, l = split
        filter_Xl = (X[:, j] <= l)
        filter_Xr = ~filter_Xl
        n_samples, n_features = X.shape

        # Check if one of the leafs is empty
        #TODO: This must be check in _best_split
        if np.nonzero(filter_Xl)[0].shape[0] in [0, n_samples]:  # One leaft is empty
            return 0.0, 0.0

        # Split X in Xl and Xr according to rule split
        Xl_cost, Xl_pred, _ = self._node_cost(y_true[filter_Xl], cost_mat[filter_Xl, :])
        Xr_cost, _, _ = self._node_cost(y_true[filter_Xr], cost_mat[filter_Xr, :])

        if self.criterion_weight:
            n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
            Xl_w = n_samples_Xl * 1.0 / n_samples
            Xr_w = 1 - Xl_w
            gain = round((cost_base - (Xl_w * Xl_cost + Xr_w * Xr_cost)) / cost_base, 6)
        else:
            gain = round((cost_base - (Xl_cost + Xr_cost)) / cost_base, 6)

        return gain, Xl_pred

    def _best_split(self, y_true, X, cost_mat):
        """ Private function to calculate the split that gives the best gain.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        tuple(split : tuple(j, l), gain : float, left node prediction : int,
              y_pred : int, y_prob : float)

        """

        n_samples, n_features = X.shape
        num_pct = self.num_pct

        cost_base, y_pred, y_prob = self._node_cost(y_true, cost_mat)

        # Calculate the gain of all features each split in num_pct
        gains = np.zeros((n_features, num_pct))
        pred = np.zeros((n_features, num_pct))
        splits = np.zeros((n_features, num_pct))

        # Selected features
        selected_features = np.arange(0, self.n_features_)
        # Add random state
        np.random.shuffle(selected_features)
        selected_features = selected_features[:self.max_features_]
        selected_features.sort()

        #TODO:  # Skip the CPU intensive evaluation of the impurity criterion for
                # features that were already detected as constant (hence not suitable
                # for good splitting) by ancestor nodes and save the information on
                # newly discovered constant features to spare computation on descendant
                # nodes.

        # For each feature test all possible splits
        for j in selected_features:
            splits[j, :] = np.percentile(X[:, j], np.arange(0, 100, 100.0 / num_pct).tolist())

            for l in range(num_pct):
                # Avoid repeated values, since np.percentile may return repeated values
                if l == 0 or (l > 0 and splits[j, l] != splits[j, l - 1]):
                    split = (j, splits[j, l])
                    gains[j, l], pred[j, l] = self._calculate_gain(cost_base, y_true, X, cost_mat, split)

        best_split = np.unravel_index(gains.argmax(), gains.shape)

        return (best_split[0], splits[best_split]), gains.max(), pred[best_split], y_pred, y_prob

    def _tree_grow(self, y_true, X, cost_mat, level=0):
        """ Private recursive function to grow the decision tree.

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        Returns
        -------
        Tree : Object
            Container of the decision tree
            NOTE: it is not the same structure as the sklearn.tree.tree object

        """

        #TODO: Find error, add min_samples_split
        if len(X.shape) == 1:
            tree = dict(y_pred=y_true, y_prob=0.5, level=level, split=-1, n_samples=1, gain=0)
            return tree

        # Calculate the best split of the current node
        split, gain, Xl_pred, y_pred, y_prob = self._best_split(y_true, X, cost_mat)

        n_samples, n_features = X.shape

        # Construct the tree object as a dictionary

        #TODO: Convert tree to be equal to sklearn.tree.tree object
        tree = dict(y_pred=y_pred, y_prob=y_prob, level=level, split=-1, n_samples=n_samples, gain=gain)

        # Check the stopping criteria
        if gain < self.min_gain:
            return tree
        if self.max_depth is not None:
            if level >= self.max_depth:
                return tree
        if n_samples <= self.min_samples_split:
            return tree
        
        j, l = split
        filter_Xl = (X[:, j] <= l)
        filter_Xr = ~filter_Xl
        n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
        n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

        if min(n_samples_Xl, n_samples_Xr) <= self.min_samples_leaf:
            return tree

        # No stooping criteria is met
        tree['split'] = split
        tree['node'] = self.tree_.n_nodes
        self.tree_.n_nodes += 1

        tree['sl'] = self._tree_grow(y_true[filter_Xl], X[filter_Xl], cost_mat[filter_Xl], level + 1)
        tree['sr'] = self._tree_grow(y_true[filter_Xr], X[filter_Xr], cost_mat[filter_Xr], level + 1)

        return tree

    class _tree_class():
        def __init__(self):
            self.n_nodes = 0
            self.tree = dict()
            self.tree_pruned = dict()
            self.nodes = []
            self.n_nodes_pruned = 0

    def fit(self, X, y, cost_mat, check_input=False):
        """ Build a example-dependent cost-sensitive decision tree from the training set (X, y, cost_mat)

        Parameters
        ----------
        y_true : array indicator matrix
            Ground truth (correct) labels.

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.


        Returns
        -------
        self : object
            Returns self.
        """

        #TODO: Check input
        #TODO: Add random state
        n_samples, self.n_features_ = X.shape

        self.tree_ = self._tree_class()

        # Maximum number of features to be taken into account per split
        if isinstance(self.max_features, six.string_types):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1, int(self.max_features * self.n_features_))
            else:
                max_features = 1  # On sklearn is 0.
        self.max_features_ = max_features

        self.tree_.tree = self._tree_grow(y, X, cost_mat)

        if self.pruned:
            self.pruning(X, y, cost_mat)

        self.classes_ = np.array([0, 1])

        return self

    def _nodes(self, tree):
        """ Private function that find the number of nodes in a tree.

        Parameters
        ----------
        tree : object

        Returns
        -------
        nodes : array like of shape [n_nodes]
        """
        def recourse(temp_tree_, nodes):
            if isinstance(temp_tree_, dict):
                if temp_tree_['split'] != -1:
                    nodes.append(temp_tree_['node'])
                    if temp_tree_['split'] != -1:
                        for k in ['sl', 'sr']:
                            recourse(temp_tree_[k], nodes)
            return None

        nodes_ = []
        recourse(tree, nodes_)
        return nodes_

    def _classify(self, X, tree, proba=False):
        """ Private function that classify a dataset using tree.

        Parameters
        ----------

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        tree : object

        proba : bool, optional (default=False)
            If True then return probabilities else return class

        Returns
        -------
        prediction : array of shape = [n_samples]
            If proba then return the predicted positive probabilities, else return
            the predicted class for each example in X
        """

        n_samples, n_features = X.shape
        predicted = np.ones(n_samples)

        # Check if final node
        if tree['split'] == -1:
            if not proba:
                predicted = predicted * tree['y_pred']
            else:
                predicted = predicted * tree['y_prob']
        else:
            j, l = tree['split']
            filter_Xl = (X[:, j] <= l)
            filter_Xr = ~filter_Xl
            n_samples_Xl = np.nonzero(filter_Xl)[0].shape[0]
            n_samples_Xr = np.nonzero(filter_Xr)[0].shape[0]

            if n_samples_Xl == 0:  # If left node is empty only continue with right
                predicted[filter_Xr] = self._classify(X[filter_Xr, :], tree['sr'], proba)
            elif n_samples_Xr == 0:  # If right node is empty only continue with left
                predicted[filter_Xl] = self._classify(X[filter_Xl, :], tree['sl'], proba)
            else:
                predicted[filter_Xl] = self._classify(X[filter_Xl, :], tree['sl'], proba)
                predicted[filter_Xr] = self._classify(X[filter_Xr, :], tree['sr'], proba)

        return predicted

    def predict(self, X):
        """ Predict class of X.

        The predicted class for each sample in X is returned.

        Parameters
        ----------

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes,
        """
        #TODO: Check consistency of X
        if self.pruned:
            tree_ = self.tree_.tree_pruned
        else:
            tree_ = self.tree_.tree

        return self._classify(X, tree_, proba=False)

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        prob : array of shape = [n_samples, 2]
            The class probabilities of the input samples.
        """
        #TODO: Check consistency of X
        n_samples, n_features = X.shape
        prob = np.zeros((n_samples, 2))

        if self.pruned:
            tree_ = self.tree_.tree_pruned
        else:
            tree_ = self.tree_.tree

        prob[:, 1] = self._classify(X, tree_, proba=True)
        prob[:, 0] = 1 - prob[:, 1]

        return prob

    def _delete_node(self, tree, node):
        """ Private function that eliminate node from tree.

        Parameters
        ----------

        tree : object

        node : int
            node to be eliminated from tree

        Returns
        -------

        pruned_tree : object
        """
        # Calculate gains
        temp_tree = copy.deepcopy(tree)

        def recourse(temp_tree_, del_node):
            if isinstance(temp_tree_, dict):
                if temp_tree_['split'] != -1:
                    if temp_tree_['node'] == del_node:
                        del temp_tree_['sr']
                        del temp_tree_['sl']
                        del temp_tree_['node']
                        temp_tree_['split'] = -1
                    else:
                        for k in ['sl', 'sr']:
                            recourse(temp_tree_[k], del_node)
            return None

        recourse(temp_tree, node)
        return temp_tree

    def _pruning(self, X, y_true, cost_mat):
        """ Private function that prune the decision tree.

        Parameters
        ----------

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        """
        # Calculate gains
        nodes = self._nodes(self.tree_.tree_pruned)
        n_nodes = len(nodes)
        gains = np.zeros(n_nodes)

        y_pred = self._classify(X, self.tree_.tree_pruned)
        cost_base = cost_loss(y_true, y_pred, cost_mat)

        for m, node in enumerate(nodes):

            # Create temporal tree by eliminating node from tree_pruned
            temp_tree = self._delete_node(self.tree_.tree_pruned, node)
            y_pred = self._classify(X, temp_tree)

            nodes_pruned = self._nodes(temp_tree)

            # Calculate %gain
            gain = (cost_base - cost_loss(y_true, y_pred, cost_mat)) / cost_base

            # Calculate %gain_size
            gain_size = (len(nodes) - len(nodes_pruned)) * 1.0 / len(nodes)

            # Calculate weighted gain
            gains[m] = gain * gain_size

        best_gain = np.max(gains)
        best_node = nodes[int(np.argmax(gains))]

        if best_gain > self.min_gain:
            self.tree_.tree_pruned = self._delete_node(self.tree_.tree_pruned, best_node)

            # If best tree is not root node, then recursively pruning the tree
            if best_node != 0:
                self._pruning(X, y_true, cost_mat)

    def pruning(self, X, y, cost_mat):
        """ Function that prune the decision tree.

        Parameters
        ----------

        X : array-like of shape = [n_samples, n_features]
            The input samples.

        y_true : array indicator matrix
            Ground truth (correct) labels.

        cost_mat : array-like of shape = [n_samples, 4]
            Cost matrix of the classification problem
            Where the columns represents the costs of: false positives, false negatives,
            true positives and true negatives, for each example.

        """
        self.tree_.tree_pruned = copy.deepcopy(self.tree_.tree)
        if self.tree_.n_nodes > 0:
            self._pruning(X, y, cost_mat)
            nodes_pruned = self._nodes(self.tree_.tree_pruned)
            self.tree_.n_nodes_pruned = len(nodes_pruned)

class ConstrainedCSDecisionTree(BaseEstimator):
    def __init__(self, 
                 constraint: float, 
                 num_iterations: int=10, 
                 tolerance: float=1.0,
                 **model_params):

        #assert 0<=constraint<=1, "Constraint must be between 0 and 1"

        self.constraint = constraint
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.model_params = model_params

    def fit(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            cfn: int):

        """
        X: The training features
        y: The training labels
        X_test: test data, it is the dataset we need to give predictions on.
        cfn: cost of false negatives
        constraint: total number of positive predictions allowed (float between 0 and 1)
        """

        #for cfp in trange(self.num_iterations):
        #We need to find the highest cfp that effective threshold is lower than the threshold

        #initialize the threshold
        threshold = 0.0
        effective_threshold = 1.0
        lower_bound = 0.0
        upper_bound = 1000.0 #Should be a big number, I don't know what is big enough
        cfp = 0

        #High CFP increase the threshold and vice versa
        while (effective_threshold > threshold) or (abs(upper_bound - lower_bound) > self.tolerance):

            print(
                f"Threshold: {threshold:.2f}", 
                f"Effective Threshold: {effective_threshold:.2f}", 
                f"Cost False Positive: {cfp:.2f}", 
                f"Lower Bound: {lower_bound:.2f}",
                f"Upper Bound: {upper_bound:.2f}",
                sep = " ", end="\r", flush=True)
            
            threshold = cfp/(cfp+cfn)
            cost_matrix = np.array([[0, cfp], 
                                    [cfn, 0]])
            model_i = CostSensitiveDecisionTreeClassifier() #The base object, we will train it.
            model_i.set_params(**self.model_params)
            model_i.fit(X, y, cost_mat = prepare_for_cost_cle(len(X), cost_matrix)) 
            y_pred_train = model_i.predict_proba(X)[:, 1]
            effective_threshold = find_effective_threshold(y_pred_train, self.constraint, threshold) 

            #The binary search, if we overshot, we need to decrease cfp
            if effective_threshold <= threshold: #It means we have found a good cfp, but might not be the best
                upper_bound = cfp
                cfp = (cfp+lower_bound)/2
                      
            else: #We undershot, we need to increase cfp
                lower_bound = cfp
                cfp = (cfp+upper_bound)/2
            
        self.best_model_ = copy.deepcopy(model_i) #This one is already trained.
        self.cfp_ = cfp
        self.threshold_ = threshold
        
        print("\nDONE")
        print(f"Threshold: {threshold:.2f}")
        print(f"Effective threshold: {effective_threshold:.2f}")
        print("Current cfp: ", cfp)
        print("Lower bound: ", lower_bound)
        print("Upper bound: ", upper_bound)
        return self
            
        # for cfp in [0.0001, 50]: #TODO: Use a better searching mechanism than brute force.
        #     print("Iteration: ", cfp)
        #     threshold = cfp/(cfp+cfn)

        #     cost_matrix = np.array([[0, cfp], 
        #                             [cfn, 0]])
        #     model_i = CostSensitiveDecisionTreeClassifier() #The base object, we will train it.
        #     model_i.set_params(**self.model_params)

        #     #training the model, create_cs_matrix is to make sure we are using the right structure
        #     model_i.fit(X, y, cost_mat = create_cs_matrix(len(X), cost_matrix)) 

        #     #predicting on the training set
        #     y_pred_train = model_i.predict_proba(X)[:, 1]
        #     y_pred_test = model_i.predict_proba(X_test)[:, 1]

        #     #self.trained_models[cfp] = deepcopy(model_i)

        #     #What is the threshold that we need to use to satisfy the constraint?
        #     effective_threshold = find_effective_threshold(y_pred_train, constraint, threshold) 
        #     print(f"Threshold: {threshold:.2f}")
        #     print(f"Effective threshold: {effective_threshold:.2f}")

        #     #y_pred_c_i = prediction_up_to_constraint(y_pred_train, constraint)
        #     #y_pred_c_i_test = prediction_up_to_constraint(y_pred_test, constraint)

        #     if effective_threshold <= threshold:
        #         #The effective threshold is lower than the real threshold, it means that the constraint is effective.
        #         #It means that we have found what we wanted
        #         self.best_model = deepcopy(model_i) #This one is already trained.
        #         print("YAY!!")
        #         return self

        # return self
    
    def predict(self, X):
        y_pred = self.best_model_.predict_proba(X)[:, 1] 
        y_pred = prediction_up_to_constraint(y_pred, self.constraint)
        return y_pred

class Constrained(BaseEstimator):
    def __init__(self, 
                 model:ClassifierMixin) -> None:
        #Composition of ClassifierMixin, including the function fit, predict, predict_proba and predict_constrained.
        self.model = model
    
    def fit(self, X, y=None, **fit_params):
        self.model_ = copy.deepcopy(self.model)
        self.model_.fit(X, y, **fit_params) #The trained object is stored in self.model_
        return self
    
    def predict(self, X):
        return self.model_.predict(X)
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    def predict_constrained(self, 
                            X: np.ndarray, 
                            constraint: float) -> np.ndarray:
        y_pred = self.predict_proba(X)[:, 1]
        return prediction_up_to_constraint(y_pred, constraint)
    
class ConstrainedXGB(BaseEstimator):
    def __init__(self, base_estimator: XGBClassifier, constraint: Constraint, tolerance: float=0.0001):
        self.base_estimator = base_estimator
        self.constraint = constraint
        self.tolerance = tolerance
        self.history = []

    def fit(self, X, y, X_test, **fit_params) -> ConstrainedXGB:
        is_satisfied = False
        constraint = self.constraint.convert_to_absolute({None: len(X_test)})
        scale_pos_weight = 1
        #Checking if the constraint is binding
        is_binding = self.is_binding(X, y, X_test)
        if is_binding:
            #The constraint is binding, we need to find the scale_pos_weight
            upper_bound = 1
            lower_bound = 0
            total_positives = -1
            #The constraint is binding, we need to find the scale_pos_weight

            while total_positives != constraint.global_constraint:
                estimator = copy.deepcopy(self.base_estimator)
                estimator.set_params(scale_pos_weight=scale_pos_weight)
                estimator.fit(X, y, **fit_params)
                y_pred = estimator.predict(X_test)
                total_positives = np.sum(y_pred)
                is_satisfied = is_satisfy_constraint(y_pred, self.constraint)
                self.history.append(
                    {"scale_pos_weight": scale_pos_weight, 
                     "total_positives": total_positives, 
                     "is_satisfied": is_satisfied, 
                     "estimator": estimator})
                
                if total_positives < constraint.global_constraint:
                    lower_bound = scale_pos_weight
                    scale_pos_weight = (scale_pos_weight + upper_bound)/2
                else:
                    upper_bound = scale_pos_weight
                    scale_pos_weight = (scale_pos_weight + lower_bound)/2

                if abs(upper_bound - lower_bound) < self.tolerance:
                    #Taking from history the best scale_pos_weight
                    logger.info("Finding the best estimator from history")
                    df = pd.DataFrame(self.history) #contains scale_pos_weight, total_positives, is_satisfied
                    best_estimator_idx = df[df["is_satisfied"]==True]["total_positives"].idxmax()
                    estimator = df.loc[best_estimator_idx]["estimator"] #highest total_positives while satisfying the constraint
                    break

                if total_positives == constraint.global_constraint:    
                    break

            self.base_estimator = copy.deepcopy(estimator)
        else:
            self.base_estimator.fit(X, y, **fit_params)
        return self

    def is_binding(self, X, y, X_test):
        estimator = copy.deepcopy(self.base_estimator)
        y_pred = estimator.fit(X, y).predict(X_test)
        return not is_satisfy_constraint(y_pred, self.constraint)

    def predict(self, X):
        return self.base_estimator.predict(X)
