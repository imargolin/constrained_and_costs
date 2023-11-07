import numpy as np
from sklearn.utils import column_or_1d

def cost_loss(y_true, y_pred, cost_mat):
    #TODO: update description
    """Cost classification loss.

    This function calculates the cost of using y_pred on y_true with
    cost-matrix cost-mat. It differ from traditional classification evaluation
    measures since measures such as accuracy asing the same cost to different
    errors, but that is not the real case in several real-world classification
    problems as they are example-dependent cost-sensitive in nature, where the
    costs due to misclassification vary between examples.

    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.

    y_pred : array-like or label indicator matrix
        Predicted labels, as returned by a classifier.

    cost_mat : array-like of shape = [n_samples, 4]
        Cost matrix of the classification problem
        Where the columns represents the costs of: false positives, false negatives,
        true positives and true negatives, for each example.

    Returns
    -------
    loss : float
        Cost of a using y_pred on y_true with cost-matrix cost-mat

    References
    ----------
    .. [1] C. Elkan, "The foundations of Cost-Sensitive Learning",
           in Seventeenth International Joint Conference on Artificial Intelligence,
           973-978, 2001.

    .. [2] A. Correa Bahnsen, A. Stojanovic, D.Aouada, B, Ottersten,
           `"Improving Credit Card Fraud Detection with Calibrated Probabilities" <http://albahnsen.com/files/%20Improving%20Credit%20Card%20Fraud%20Detection%20by%20using%20Calibrated%20Probabilities%20-%20Publish.pdf>`__, in Proceedings of the fourteenth SIAM International Conference on Data Mining,
           677-685, 2014.

    See also
    --------
    savings_score

    Examples
    --------
    >>> import numpy as np
    >>> from costcla.metrics import cost_loss
    >>> y_pred = [0, 1, 0, 0]
    >>> y_true = [0, 1, 1, 0]
    >>> cost_mat = np.array([[4, 1, 0, 0], [1, 3, 0, 0], [2, 3, 0, 0], [2, 1, 0, 0]])
    >>> cost_loss(y_true, y_pred, cost_mat)
    3
    """

    #TODO: Check consistency of cost_mat

    y_true = column_or_1d(y_true)
    y_true = (y_true == 1).astype(np.float32)
    y_pred = column_or_1d(y_pred)
    y_pred = (y_pred == 1).astype(np.float32)
    cost = y_true * ((1 - y_pred) * cost_mat[:, 1] + y_pred * cost_mat[:, 2])
    cost += (1 - y_true) * (y_pred * cost_mat[:, 0] + (1 - y_pred) * cost_mat[:, 3])
    return np.sum(cost)
