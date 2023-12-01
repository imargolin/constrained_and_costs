from abc import ABC, abstractmethod

class Constraint(ABC):
    def __init__(self, global_constraint, local_constraints):
        self.global_constraint = global_constraint
        self.local_constraints = local_constraints
    
    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(global_constraint={self.global_constraint}, local_constraints={self.local_constraints})"

    def to_dict(self):
        out = {}
        out["global_constraint"] = self.global_constraint
        if self.local_constraints is not None:
            out["local_constraints"] = self.local_constraints
        
        return out

class AbsoluteConstraint(Constraint):
    def __init__(self, 
                 global_constraint: int, 
                 local_constraints: dict[str, int] | None = None):
        super().__init__(global_constraint, local_constraints)
        self._validate_constraint()

    def convert_to_relative(self, group_counts):
        """
        Converts an absolute constraint to a relative constraint
        Args:
            group_counts: Total number of elements in each group
        Returns:
            RelativeConstraint
        """
        
        rel_constraints = {}
        total = sum(group_counts.values())
        rel_constraints["global_constraint"] = self.global_constraint/total
        rel_constraints["local_constraints"] = None

        if self.local_constraints:
            rel_constraints["local_constraints"] = {}
            for group in self.local_constraints:
                rel_constraints["local_constraints"][group] = self.local_constraints[group]/group_counts[group]
        
        return RelativeConstraint(**rel_constraints)
    
    def _validate_constraint(self):
        #y_pred_probs: np.array of predicted probabilities
        assert self.global_constraint >= 0
        if self.local_constraints is not None:
            for group in self.local_constraints:
                assert self.local_constraints[group] >= 0, "Local constraints must be non-negative"
    
    def as_budget(self):
        return Budget(**self.to_dict())
        
    
class RelativeConstraint(Constraint):
    def __init__(self, 
                 global_constraint: float, 
                 local_constraints: dict[str, float] | None = None):
        super().__init__(global_constraint, local_constraints)
        self._validate_constraint()
    
    def convert_to_absolute(self, group_counts: dict[str, int]):
        """
        Converts a relative constraint to an absolute constraint
        Args:
            group_counts: Total number of elements in each group
        Returns:
            AbsoluteConstraint
        """
        
        absolute_constraint = {}
        total = sum(group_counts.values()) #We derive the size of the global constraint from the group counts
        
        absolute_constraint["global_constraint"] = int(self.global_constraint*total)

        if self.local_constraints is not None:
            absolute_constraint["local_constraints"] = {}
            for group in self.local_constraints:
                absolute_constraint["local_constraints"][group] = int(self.local_constraints[group]*group_counts[group])
        return AbsoluteConstraint(**absolute_constraint)
    
    def _validate_constraint(self):
        #global constraint must be between 0 and 1
        assert self.global_constraint <= 1.0 and self.global_constraint >= 0, "Global constraint must be between 0 and 1"
        
        if self.local_constraints is not None:
            assert all([self.local_constraints[group] >= 0 for group in self.local_constraints]), "Local constraints must be between 0 and 1"
            assert all([self.local_constraints[group] <= 1.0 for group in self.local_constraints]), "Local constraints must be between 0 and 1"

class Budget:
    def __init__(self, 
                 global_constraint: float, 
                 local_constraints: dict[str, float] | None = None):
        self.global_constraint = global_constraint
        self.local_constraints = local_constraints
    
    def pay(self, group_id):
        self.global_constraint -= 1
        if self.local_constraints:
            if group_id in self.local_constraints:
                self.local_constraints[group_id] -= 1
    
    def has_enough(self, group_id):
        if self.global_constraint<1:
            return False
        if self.local_constraints:
            if group_id in self.local_constraints:
                if self.local_constraints[group_id] <1:
                    return False
        return True