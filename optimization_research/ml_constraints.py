import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
#importing optional for the user

from typing import List, Dict, Union, Optional
from .preparation_utils import prepare_y_hat, repeat_cost_matrix, create_num_vars

#for binary case
#global_constraint = {"constraint_type": "absolute", "value": 100}
# global_constraint = {"constraint_type": "relative", "value": 2}
# local_constraint = {"constraint_type": "absolute", "contraints": {"group1": 0, "group2": 50}}

# y_hat = [1.0, 1.0, 1.0, 1.0]
# groups = ["group1", "group1", "group2", "group3"]

# #cost mnatrix is N*2*2 where N is the number of samples
# #cost_matrix = np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]], [[0, 1], [1, 0]], [[0, 1], [1, 0]]])
# cost_matrix = np.array([[1, 2], [3, 4]])


#the constraints are optionals

class MulticlassSolver:
    def __init__(self,
                 y_hat: np.ndarray, 
                 groups: np.ndarray,
                 cost_matrix: np.ndarray,
                 global_constraint: Optional[List[Union[int, None]]] = None,
                 local_constraint: Optional[Dict[str, List[Union[int, None]]]] = None):
        
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
        self.y_hat = y_hat #2d array of probabilities (N, C)
        self.groups = groups #1d array of groups (N,)
        self.cost_matrix = cost_matrix #3d array of cost matrix (N, C, C)
        self.global_constraint = global_constraint #list of global constraints (C,)
        self.local_constraint = local_constraint #dict of local constraints with group as key and list of constraints as value

        self.n_samples, self.n_classes = self.y_hat.shape

        self.assert_input()
        

    def assert_input(self):
        n_samples, n_classes = self.y_hat.shape
        assert len(self.y_hat.shape) == 2, "y_hat should be a 2D array"

        #cost matrix must be (N,C,C) or (C,C)
        assert len(self.cost_matrix.shape) == 3 or len(self.cost_matrix.shape) == 2, "The cost matrix should be 2D or 3D"
        if len(self.cost_matrix.shape) == 3:
            assert self.cost_matrix.shape[0] == n_samples, "The cost matrix should have the same number of samples as y_hat"

        assert self.groups.shape[0] == n_samples, "The number of samples in y_hat and groups should be the same"
        assert self.cost_matrix.shape[-1] == self.cost_matrix.shape[-2], "The cost matrix should be square"
        assert self.cost_matrix.shape[1] == n_classes, "The number of classes in y_hat and cost_matrix should be the same"
        assert self.global_constraint is None or len(self.global_constraint) == n_classes, "The number of classes in y_hat and global_constraint should be the same"
        #all local constraints should have the same number of classes
        assert self.local_constraint is None or all([len(a) == n_classes for a in self.local_constraint.values()]), "The number of classes in y_hat and local_constraint should be the same"
        assert np.allclose(self.y_hat.sum(axis=1), 1), "The sum of y_hat should be 1"

    def solve(self):
        

        if self.cost_matrix.ndim == 2:
            self.cost_matrix = repeat_cost_matrix(self.cost_matrix, self.n_samples)

        R = np.empty((self.n_samples, self.n_classes)).tolist()
        #return num_vars
        for i in range(self.n_samples):
            for j in range(self.n_classes):
                R[i][j] = self.solver.NumVar(0, self.solver.infinity(), 'R{}{}'.format(i, j))
        R = np.array(R)
        #print(R.shape)

        #Objective function
        objective_function = ((self.cost_matrix * self.y_hat[:, :, np.newaxis]).transpose(0, 2, 1) * R[:, :, np.newaxis]).sum()

        #Constraints
        [self.solver.Add(a>=1) for a in R.sum(axis=1)] #We must predict at least one class

        #Global constraints
        if self.global_constraint:
            totals = R.sum(axis=0)
            for class_number in range(self.n_classes):
                constraint = self.global_constraint[class_number]
                if constraint is not None:
                    
                    self.solver.Add(totals[class_number] <= constraint)
                    #print(totals[class_number])
        
        if self.local_constraint:
            for group, list_constraints in self.local_constraint.items():
                #list_constraints is with shape of C
                totals = R[self.groups == group].sum(axis=0) #shape of C
                for class_number in range(self.n_classes):
                    constraint = list_constraints[class_number]
                    if constraint is not None:
                        self.solver.Add(totals[class_number] <= constraint)
        
        self.solver.Minimize(objective_function)
        status = self.solver.Solve()
        results = pd.DataFrame(R).map(lambda x: x.solution_value()).values#[:, 1]
        self.objective_function = objective_function

        if status == self.solver.OPTIMAL:
            print("The problem has an optimal solution")
        
        if status == self.solver.INFEASIBLE:
            print("The problem doesn't have a feasible solution")

        print("Status: ", status)
        print("Objective function: ", objective_function.solution_value())

        return results

# #TODO: add relative constraints support
# #TODO: add support for binary case

class BinarySolver:
    def __init__(self,
                 y_hat: np.ndarray, 
                 groups: np.ndarray,
                 cost_matrix: np.ndarray, # (N, 2, 2)
                 global_constraint: Optional[int] = None, #number or None, if None, no global constraint. If number, the constraint of positive class (must be lower)
                 local_constraint: Optional[Dict[str, int]] = None): #dict with group as key and number as value
        
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
        self.y_hat = y_hat #This time it should be a 1d vector of probabilities
        self.groups = groups #This time it should be a 1d vector of groups
        self.cost_matrix = cost_matrix
        self.global_constraint = global_constraint
        self.local_constraint = local_constraint

        self.n_samples = len(self.y_hat)
    
    def solve(self):

        R = np.empty(self.n_samples).tolist() #The results is a vector
        for i in range(self.n_samples):
            R[i] = self.solver.NumVar(0, 1, 'R{}'.format(i))

        R = np.array(R)
        #print(self.cost_matrix[:, 1, 1].shape)#, R.shape, self.y_hat.shape)

        TPC = self.cost_matrix[:, 1, 1] * R * self.y_hat
        FPC = self.cost_matrix[:, 0, 1] * R * (1-self.y_hat) #The price of false positives
        TNC = self.cost_matrix[:, 0, 0] * (1-R) * (1-self.y_hat)
        FNC = self.cost_matrix[:, 1, 0] * (1-R) * self.y_hat

        objective_function = (TPC + FPC + TNC + FNC).sum()

        if self.global_constraint:
            totals = R.sum(axis=0)
            self.solver.Add(totals <= self.global_constraint)

        if self.local_constraint:
            for group, constraint in self.local_constraint.items():
                totals = R[self.groups == group].sum(axis=0)
                self.solver.Add(totals <= constraint)

        self.solver.Minimize(objective_function)
        status = self.solver.Solve()
        self.objective_function_value = objective_function.solution_value()

        results = pd.Series(R).map(lambda x: x.solution_value()).values

        # print ("Status: ", status)
        # print("Objective function: ", self.objective_function_value)
        
        return results
        