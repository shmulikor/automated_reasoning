import numpy as np

def names_generator():
    i = 0
    while True:
        i += 1
        yield i


names = {}
counter = names_generator()


class BooleanOperator:
    def __init__(self, name: int):
        self.name = name


class And(BooleanOperator):
    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        print(f"{self.name}: {self.left.name} and {self.right.name}")


class Or(BooleanOperator):
    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        print(f"{self.name}: {self.left.name} or {self.right.name}")


class Not(BooleanOperator):
    def __init__(self, param: BooleanOperator):
        if type(param) == Atomic:
            super().__init__(-param.name)
        else:
            super().__init__(next(counter))
        self.param = param
        print(f"{self.name}: not {self.param.name}")


class Imp(BooleanOperator):
    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        print(f"{self.name}: {self.left.name} -> {self.right.name}")


class Equiv(BooleanOperator):
    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        print(f"{self.name}: {self.left.name} <-> {self.right.name}")


class Atomic(BooleanOperator):
    def __init__(self, name: str):
        if name not in names.keys():
            names[name] = next(counter)
            super().__init__(names[name])
        else:
            super().__init__(names[name])
        self.val = name
        print(f"{self.name}: {self.val}")


# A class responsible for converting a general formula to CNF
# CNF is represented as a list of lists. Lower level list represents the clauses, and elements represent literals
# Atomic variables represented by abs of literals, negative elements are negated atomics
class FormulaToCNF:
    def __init__(self, formula):
        self.formula = formula
        self.cnf = []

    @staticmethod
    def simple_cnf(p, q, r, type):
        # Manual conversion of (p <-> q*r) to CNF, for every connector * # TODO validate
        if type == And:
            return [[-p, q], [-p, r], [p, -q, -r]]
        elif type == Or:
            return [[-p, q, r], [p, -q], [p, -r]]
        elif type == Not:
            return [[p, q], [-p, -q]]
        elif type == Imp:
            return [[-p, -q, r], [p, q], [p, -r]]
        elif type == Equiv:
            return [[p, q, r], [-p, -q, -r]]
        return

    def tseitin(self):
        # Recursively convert to Tseitin form
        return self.tseitin_helper(self.formula, [[self.formula.name]])

    def tseitin_helper(self, formula, list):
        if type(formula) == Atomic:
            return
        elif type(formula) == Not:
            list.extend(self.simple_cnf(formula.name, formula.param.name, None, type(formula)))
            self.tseitin_helper(formula.param, list)
        else:
            list.extend(self.simple_cnf(formula.name, formula.left.name, formula.right.name, type(formula)))
            self.tseitin_helper(formula.left, list)
            self.tseitin_helper(formula.right, list)
        return list

    def preprocessing(self):
        # Removes redundant literals and trivial clauses from Tseitin formula
        processed_tseitin = []
        for clause in self.cnf:
            clause = list(set(clause))  # remove redundant literals
            if len(set(np.abs(clause))) == len(np.abs(clause)):  # remove trivial clauses
                processed_tseitin.append(clause)
        return processed_tseitin

    def run(self):
        self.cnf = self.tseitin()
        self.cnf = self.preprocessing()
