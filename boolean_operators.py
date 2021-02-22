import numpy as np


def names_generator():
    # Generates names for new variables (starting at 1)
    # Each var is represented by a number
    i = 0
    while True:
        i += 1
        yield i


names = {}
counter = names_generator()
DEBUG = False

class BooleanOperator:
    # General class representing a boolean operator
    def __init__(self, name: int):
        self.name = name


class And(BooleanOperator):
    # A class representing the And boolean operator

    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        if DEBUG:
            print(f"{self.name}: {self.left.name} and {self.right.name}")

    def __repr__(self):
        return f"({self.left.__repr__()} and {self.right.__repr__()})"


class Or(BooleanOperator):
    # A class representing the Or boolean operator

    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        if DEBUG:
            print(f"{self.name}: {self.left.name} or {self.right.name}")

    def __repr__(self):
        return f"({self.left.__repr__()} or {self.right.__repr__()})"


class Not(BooleanOperator):
    # A class representing the Not boolean operator

    def __init__(self, param: BooleanOperator):
        if type(param) == Atomic:
            super().__init__(-param.name)
        else:
            super().__init__(next(counter))
        self.param = param
        if DEBUG:
            print(f"{self.name}: not {self.param.name}")

    def __repr__(self):
        return f"not({self.param.__repr__()})"


class Imp(BooleanOperator):
    # A class representing the Implication boolean operator

    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        if DEBUG:
           print(f"{self.name}: {self.left.name} -> {self.right.name}")

    def __repr__(self):
        return f"({self.left.__repr__()} -> {self.right.__repr__()})"

class Equiv(BooleanOperator):
    # A class representing the Equivalence boolean operator

    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        if DEBUG:
            print(f"{self.name}: {self.left.name} <-> {self.right.name}")

    def __repr__(self):
        return f"({self.left.__repr__()} <=> {self.right.__repr__()})"

class Atomic(BooleanOperator):
    # A class representing an Atomic proposition

    def __init__(self, name: str):
        if name not in names.keys():
            names[name] = next(counter)
            super().__init__(names[name])
        else:
            super().__init__(names[name])
        self.val = name
        if DEBUG:
            print(f"{self.name}: {self.val}")

    def __repr__(self):
        return self.val


# A class responsible for converting a general formula to CNF
# CNF is represented as a list of lists. Lower level lists represent the clauses, and elements represent literals
# Atomic variables represented by abs of literals, negative elements are negated atomics
class FormulaProcessor:
    def __init__(self, formula):
        self.formula = formula
        self.cnf = []
        self.atomic_abstractions = {}

    @staticmethod
    def simple_cnf(p, q, r, bool_op):
        # Results of Manual conversion of (p <-> q*r) to CNF, for every boolean operator *
        # According to algorithm learned in class - conversion to NNF and using distribution
        if bool_op == And:
            return [[-p, q], [-p, r], [p, -q, -r]]
        elif bool_op == Or:
            return [[-p, q, r], [p, -q], [p, -r]]
        elif bool_op == Not:
            return [[p, q], [-p, -q]]
        elif bool_op == Imp:
            return [[-p, -q, r], [p, q], [p, -r]]
        elif bool_op == Equiv:
            return [[p, q, r], [p, -q, -r], [-p, q, -r], [-p, -q, r]]
        return

    def tseitin(self):
        # Calls for recursive conversion to Tseitin form
        if type(self.formula) == Atomic:
            inv_name = {v: k for k, v in names.items()}
            self.atomic_abstractions[self.formula.name] = inv_name[self.formula.name]
            return [[self.formula.name]]
        return self.tseitin_helper(self.formula, [[self.formula.name]])

    def tseitin_helper(self, formula, tseitin):
        # Performs the core of Tseitin transformation
        # Formula is given in boolean operators form
        # cnf is returned as a lists of lists (clauses) of numbers (negation represented by minus sign)
        if type(formula) == Atomic:
            inv_name = {v: k for k, v in names.items()}
            self.atomic_abstractions[formula.name] = inv_name[formula.name]
            return
        elif type(formula) == Not:
            tseitin.extend(self.simple_cnf(formula.name, formula.param.name, None, type(formula)))
            self.tseitin_helper(formula.param, tseitin)
        else:
            tseitin.extend(self.simple_cnf(formula.name, formula.left.name, formula.right.name, type(formula)))
            self.tseitin_helper(formula.left, tseitin)
            self.tseitin_helper(formula.right, tseitin)
        return tseitin

    def preprocessing(self):
        # Removes redundant literals and trivial clauses from Tseitin formula
        processed_tseitin = []
        for clause in self.cnf:
            clause = list(set(clause))  # remove redundant literals
            if len(set(np.abs(clause))) == len(np.abs(clause)):  # remove trivial clauses
                processed_tseitin.append(clause)
        return processed_tseitin

    def convert_and_preprocess(self):
        # Converts formula to Tseitin form and preprocesses it
        self.cnf = self.tseitin()
        self.cnf = self.preprocessing()
        return self.cnf
