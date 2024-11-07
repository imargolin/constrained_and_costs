
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
#importing optional for the user

from typing import List, Dict, Union, Optional
from .preparation_utils import prepare_y_hat, repeat_cost_matrix, create_num_vars
from pulp import LpMaximize , LpProblem , LpVariable , lpSum ,LpBinary, PULP_CBC_CMD

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
    
class ItayChenSolverPreprocessIncluded:
    def __init__(
            self, 
            input_dataframe: pd.DataFrame,
            limits: Dict[str, int],
            prizes: Optional[Dict[str, float]] = {},
            ):
        """
        input_dataframe: pd.DataFrame, with columns user, credential, provider
        limits: dict, with provider as keys and limits as values
        values: optional, dict, with user as keys and values as values, users not in the dict will have a default value of 1
        """
        
        self.effective_users = input_dataframe[input_dataframe["provider"].isin(limits.keys())]["user"].unique()
        self.effective_providers = input_dataframe[input_dataframe["provider"].isin(limits.keys())]["provider"].unique()
        self.limits = limits
        
        #For the solver, not for the finalized solution.
        self.N = len(self.effective_users)
        self.M = len(self.effective_providers)


        self.prizes = pd.Series(prizes, name="values")
        effective_W = input_dataframe[(input_dataframe["user"].isin(self.effective_users)) & (input_dataframe["provider"].isin(self.effective_providers))]
        effective_W = effective_W.groupby(["user", "provider"]).size().unstack(fill_value=0)
        self.effective_W =  effective_W.reindex(self.effective_users, axis=0).reindex(self.effective_providers, axis=1)
        self.output_df = input_dataframe[["user"]].drop_duplicates("user").set_index("user").join(self.prizes).fillna(1)
        self.effective_values = self.output_df["values"].reindex(self.effective_users)

        self.solver = pywraplp.Solver.CreateSolver("SAT")

    def solve(self):
        R = np.empty(self.N).tolist()
        for i in range(self.N):
            R[i] = self.solver.IntVar(0, 1, 'R{}'.format(i))
        R = np.array(R)

        demand = (R[:, None] * self.effective_W.values).sum(axis=0)
        for j in range(self.M):
            self.solver.Add(demand[j] <= self.limits[self.effective_providers[j]])

        objective_function = (R * self.effective_values.values).sum()
        self.solver.Maximize(objective_function)
        status = self.solver.Solve()

        results = pd.Series(R, index = self.effective_users).map(lambda x: x.solution_value())

        if status == self.solver.OPTIMAL:
            print("The problem has an optimal solution")
        
        if status == self.solver.INFEASIBLE:
            print("The problem doesn't have a feasible solution")

        self.output_df["decisions"] = results
        self.output_df["decisions"] = self.output_df["decisions"].fillna(1)
        return self.output_df["decisions"]

class ItayChenSolver2:
    def __init__(
            self, 
            input_dataframe: pd.DataFrame,
            limits: Dict[str, int],
            prizes: Optional[Dict[str, float]] = {},
            ):
        """
        input_dataframe: pd.DataFrame, with columns user, credential, provider
        limits: dict, with provider as keys and limits as values
        prizes: optional, dict, with user as keys and values as values, users not in the dict will have a default value of 1
        """
        
        self.W = input_dataframe.groupby(["user", "provider"]).size().unstack(fill_value=0)

        self.users = self.W.index
        self.providers = self.W.columns
        self.limits = pd.Series(limits, name="limits").reindex(self.providers).fillna(float("inf"))
        self.prizes = pd.Series(prizes, name="prizes").reindex(self.users).fillna(1)
        
        #For the solver, not for the finalized solution.
        self.N, self.M = self.W.shape

        self.solver = pywraplp.Solver.CreateSolver("SAT")

    def solve(self):
        R = np.empty(self.N).tolist() #The results is a vector, number of users
        for i in range(self.N):
            R[i] = self.solver.IntVar(0, 1, 'R{}'.format(i))
        R = np.array(R)

        demand = (R[:, None] * self.W.values).sum(axis=0)
        for j in range(self.M):
            self.solver.Add(demand[j] <= self.limits[self.providers[j]])

        objective_function = (R * self.prizes.values).sum()
        self.solver.Maximize(objective_function)
        status = self.solver.Solve()

        self.results = pd.Series(R, index = self.users).map(lambda x: x.solution_value())

        if status == self.solver.OPTIMAL:
            print("The problem has an optimal solution")
        
        if status == self.solver.INFEASIBLE:
            print("The problem doesn't have a feasible solution")

        return self.results
    
class GonenSolver:
    def __init__(
            self,
            input_dataframe: pd.DataFrame,
            limits: Dict[str, int],
            prizes: Optional[Dict[str, float]] = {},
            ):
        
        input_dataframe = input_dataframe.set_index("credential")
        self.W = input_dataframe.groupby(["user", "provider"])["credential_weight"].sum().unstack(fill_value=0)
        self.users, self.providers = self.W.index, self.W.columns
        self.credentials = input_dataframe.index.tolist()
        self.prizes = {user: prizes.get(user, 1) for user in self.users} # if user is not in prizes, then prize is 1
        self.limits = {provider: limits.get(provider, float("inf")) for provider in self.providers} # filter out providers that are not in the data
        self.creds_weights = input_dataframe["credential_weight"].to_dict()

        self.user_to_credentials = input_dataframe.groupby("user").apply(lambda x: set(x.index)).to_dict()
        self.provider_to_credentials = input_dataframe.groupby("provider").apply(lambda x: set(x.index)).to_dict()

        self.solver = LpProblem("ILP_Problem", LpMaximize)
        
    def solve(self):
        x = LpVariable.dicts("x", self.users , cat = LpBinary) # Binary variable for family selection
        y = LpVariable.dicts("y", [(credential ,provider) for credential in self.credentials for provider in self.providers] ,cat = LpBinary) # Binary variable for item assignment

        self.solver += lpSum(self.prizes[user] * x [user] for user in self.users)

        #Constraints

        # The sum of the weights of the credentials assigned to a provider must be less than or equal to the limit of the provider
        for provider, limit in self.limits.items():
            if limit != float("inf"):
                self.solver += lpSum(self.creds_weights[credential] * y[credential , provider] for credential in self.credentials) <= self.limits[provider]
                
        # The sum of the weights of the credentials assigned to a user must be less than or equal to the limit of the user
        for user, user_credentials in self.user_to_credentials.items():
            for credential in user_credentials:
                self.solver += lpSum (y[credential, provider] for provider in self.providers) == x[user]

        # The credentials that are not allowed to be assigned to a provider must be 0
        set_credentials = set(self.credentials)
        for provider in self.providers:
            misalowed_credentials = set_credentials - self.provider_to_credentials[provider]
            for credential in misalowed_credentials:
                self.solver += y[credential , provider] == 0

        self.solver.solve(PULP_CBC_CMD(msg=True)) #

        self.results = pd.Series(x).map(lambda x: x.varValue)
        return self.results
