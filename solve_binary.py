import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
#importing optional for the user

from typing import List, Dict, Union, Optional


def solve_binary(y_hat: np.ndarray, 
          groups: np.ndarray, 
          cost_matrix: np.ndarray, 
          global_constraint: Optional[Dict[str, Union[str, int]]] = None, 
          local_constraint: Optional[Dict[str, Union[str, Dict[str, int]]]] = None):
    
    """
    This function solves the optimization problem for the binary case. 
    The user can insert a global constraint and a local constraint. 

    The global constraint is a dictionary with the following structure:
        global_constraint = {"constraint_type": "absolute", "value": 100}

    The local constraint is a dictionary with the following structure:
        local_constraint = {"constraint_type": "absolute", "contraints": {"group1": 0, "group2": 50}}

    The user can also insert a cost matrix with a shape of (N, 2, 2) or (2, 2)
    In case the cost matrix has a shape of (2, 2), it will be repeated N times to match the number of samples.

    Args:
        y_hat (np.ndarray): The predicted probabilities of the positive class
        groups (np.ndarray): The groups of each sample
        cost_matrix (np.ndarray): The cost matrix
        global_constraint (Optional[Dict[str, Union[str, int]]], optional): The global constraint. Defaults to None.
        local_constraint (Optional[Dict[str, Union[str, Dict[str, int]]]], optional): The local constraint. Defaults to None.
    """

    
    #preprocess stuff
    y_hat = prepare_y_hat(y_hat)
    n_classes = 2
    groups = np.array(groups)

    n_samples = len(y_hat)

    #solver = pywraplp.Solver.CreateSolver('GLOP')

    #The user has option to insert a cost matrix with a shape of (N, 2, 2) or (2, 2)
    if cost_matrix.shape == (2, 2):
        cost_matrix = repeat_cost_matrix(cost_matrix, n_samples)

    #create list of lists with shape nX2
    #R = create_num_vars(n_samples, n_classes)
    R = np.empty((n_samples, n_classes)).tolist()
    #return num_vars
    for i in range(n_samples):
        for j in range(n_classes):
            R[i][j] = solver.NumVar(0, solver.infinity(), 'R{}{}'.format(i, j))
    
    R = np.array(R)

    #case when we predict negative (we predict not to crawl)
    case1 = R[:, 0] * (y_hat[:, 0] * cost_matrix[:, 0, 0] + y_hat[:, 1] * cost_matrix[:, 1, 0])
    #case when we predict positive (we predict to crawl)
    case2 = R[:, 1] * (y_hat[:, 0] * cost_matrix[:, 0, 1] + y_hat[:, 1] * cost_matrix[:, 1, 1])

    
    
    objective_function = case1.sum() + case2.sum()

    [solver.Add(a>=1) for a in R.sum(axis=1)] #We must predict at least one class

    if global_constraint:
        solver.Add(R.sum(axis=0)[1] <= global_constraint["value"]) #We must predict at most global_constraint["value"] of class 1

    if local_constraint:
        for group in local_constraint["contraints"]:
            solver.Add(R[group==groups].sum(axis=0)[1] <= local_constraint["contraints"][group])

    solver.Minimize(objective_function)
    
    status = solver.Solve()

    results = pd.DataFrame(R).map(lambda x: x.solution_value()).values#[:, 1]
    
    return results

local_constraint = {"constraint_type": "absolute", "contraints": {"group1": 0, "group2": 0}}
local_constraint = {"constraint_type": "absolute", "contraints": {"group1": 0, "group2": 0, "group3": 100}}
local_constraint = {"constraint_type": "absolute", "contraints": {"group1": 0, "group2": 0, "group3": 10}}
x = solve_binary(y_hat, groups, cost_matrix, global_constraint, local_constraint=local_constraint)