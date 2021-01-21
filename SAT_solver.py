import numpy as np
from collections import defaultdict
from boolean_operators import And, Or, Not, Imp, Equiv, Atomic


class FormulaToCNF:
    def __init__(self, formula):
        self.formula = formula
        self.cnf = []

    @staticmethod
    def simple_cnf(p, q, r, type):
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
        processed_tseitin = []
        for clause in self.cnf:
            clause = list(set(clause))  # remove redundant literals
            if len(set(np.abs(clause))) == len(np.abs(clause)):  # remove trivial clauses
                processed_tseitin.append(clause)
        return processed_tseitin

    def run(self):
        self.cnf = self.tseitin()
        self.cnf = self.preprocessing()


class Assignments:
    def __init__(self, cnf):
        self.cnf = cnf
        self.var_assignment = {}
        self.assigned_clauses = [False] * len(cnf)

    def bcp(self):
        changed = True
        while changed:
            changed = False
            for i, clause in enumerate(self.cnf):
                if not self.assigned_clauses[i]:
                    if all([abs(l) in self.var_assignment.keys() for l in clause]):
                        self.assigned_clauses[i] = True
                    else:
                        unassigned = [l for l in clause if abs(l) not in self.var_assignment.keys()]
                        if len(unassigned) == 1:
                            assigned = [l for l in clause if abs(l) in self.var_assignment.keys()]
                            past_assignments = [self.var_assignment[l] for l in assigned if l > 0] \
                                               + [not self.var_assignment[-l] for l in assigned if l < 0]
                            if not any(past_assignments):
                                self.var_assignment[abs(unassigned[0])] = True if unassigned[0] > 0 else False
                                self.assigned_clauses[i] = True
                                changed = True



def dlis(formula, var_assignment, assigned_clauses):
    appearances_dict = defaultdict(int)
    for i, clause in enumerate(formula):
        if not assigned_clauses[i]:
            for literal in clause:
                if abs(literal) not in var_assignment.keys():
                    appearances_dict[abs(literal)] += 1
    return max(appearances_dict, key=lambda k: appearances_dict[k])



if __name__ == '__main__':
    formula = Not(Imp(Not(And(Atomic('p'), Atomic('q'))), Not(Atomic('r'))))
    # formula = And(Atomic('p'), Atomic('q'))
    f2cnf = FormulaToCNF(formula)
    f2cnf.run()
    cnf = f2cnf.cnf
    print(cnf)
    a = Assignments(cnf)
    a.bcp()
    print(a.var_assignment, a.assigned_clauses)