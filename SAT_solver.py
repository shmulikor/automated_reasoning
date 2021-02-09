import numpy as np
from collections import defaultdict
from boolean_operators import And, Or, Not, Imp, Equiv, Atomic

USE_WATCH_LIT = False


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


class Assignments:
    def __init__(self, cnf):
        self.cnf = cnf
        self.var_assignment = {}
        self.assigned_clauses = [False] * len(cnf)
        self.decision_levels = {}
        self.splits = []
        self.current_decision_level = 0
        self.init_watch_literals()

    def init_watch_literals(self):
        # Upon initialization - all literals are unassigned
        # Choose two random literals from each clause (or one, if there is only one)
        self.watch_literals = []
        for i, clause in enumerate(self.cnf):
            self.watch_literals.append([])
            lit_set = set(clause)
            if len(lit_set) == 1:
                self.watch_literals[i] = [lit_set.pop()]
            else:
                self.watch_literals[i] = [lit_set.pop(), lit_set.pop()]

    def update_watch_literals(self, indices_to_update):
        # TODO possible improvement - update only one watch literal if possible, since assignment changes only one
        # watch literal
        for index in indices_to_update:
            lit_set = set(self.cnf[index])
            clause_watch_literals = []
            while len(lit_set) > 0 and len(clause_watch_literals) < 2:
                lit = lit_set.pop()
                if abs(lit) not in self.var_assignment.keys():
                    clause_watch_literals.append(lit)
            self.watch_literals[index] = clause_watch_literals

    def get_bcp_candidates(self, literal):
        # Returns sublist of all clauses which include the negation of the literal as a watch literal
        # As stated in class, these are only ones relevant for BCP after assignment
        indices = [i for i, watchers in enumerate(self.watch_literals) if -literal in watchers]
        return [self.cnf[j] for j in indices]

    def bcp(self):
        # Performs BCP until saturation, returns True iff formula has a falsified clause
        changed = True
        if USE_WATCH_LIT and len(self.splits) > 0:
            bcp_candidates = self.get_bcp_candidates(self.splits[-1])
        else:
            bcp_candidates = self.cnf

        while changed:
            changed = False
            for i, clause in enumerate(bcp_candidates):
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
                                chosen_var = unassigned[0]
                                self.var_assignment[abs(chosen_var)] = True if chosen_var > 0 else False
                                self.assigned_clauses[i] = True
                                self.decision_levels[abs(chosen_var)] = self.current_decision_level
                                changed = True
                                involved_indices = [j for j, clause in enumerate(self.cnf) if
                                                    chosen_var in self.cnf[j] or -chosen_var in self.cnf[j]]
                                bcp_candidates = self.get_bcp_candidates(chosen_var) # TODO review carefully - should
                                # first choose candidates and then update or other way around?
                                self.update_watch_literals(involved_indices)
        return self.has_false_clause()  # TODO lec2 p39

    def has_false_clause(self):
        # returns true iff the formula has a falsified clause
        for clause in self.cnf:
            # Counts the number of literals assigned to false
            false_literals = [l for l in clause if ((l < 0 and abs(l) in self.var_assignment.keys() and
                                                     self.var_assignment[abs(l)])
                                                    or
                                                    (l > 0 and l in self.var_assignment.keys() and
                                                     not self.var_assignment[l]))]
            if len(false_literals) == len(clause):
                return False
        return True

    def decide_next(self):
        # Splits cases by using DLIS heuristic, and updates all relevant fields
        # to support non-chronological backjumps
        var, assign = dlis(self, self.var_assignment, self.assigned_clauses)
        assert (var > 0)  # TODO maybe erase after debugging
        self.splits.append(var)
        self.var_assignment[var] = assign
        self.decision_levels[var] = self.current_decision_level
        self.current_decision_level += 1

        involved_indices = [i for i, clause in enumerate(self.cnf) if var in self.cnf[i] or -var in self.cnf[i]]
        # Update watch literals for every clause with decided literal
        self.update_watch_literals(involved_indices)
        # Update assigned clauses
        for i in involved_indices:
            if all([abs(l) in self.var_assignment.keys() for l in self.cnf[i]]):
                self.assigned_clauses[i] = True

    def perform_backjump(self, level):
        # Performs non-chronological backjump to a specified decision level
        self.splits = self.splits[:level]
        self.current_decision_level = level
        for var in self.decision_levels.keys():
            if self.decision_levels[var] > level:
                self.decision_levels.pop(var)
                self.var_assignment.pop(var)
        self.update_watch_literals(range(len(self.cnf)))  # Update watch literals for all (TODO improve from naive way?)
        for i, has_assigned in enumerate(self.assigned_clauses):  # Updates assigned clauses, can only make
            # previously assigned clauses to become unassigned
            if has_assigned:
                if any([abs(l) not in self.var_assignment.keys() for l in self.cnf[i]]):
                    self.assigned_clauses[i] = False

    def analyze_conflict(self):
        # Creates implications graph and decides using UIP the backjump level and the learned clause
        return [], 2

    def solve(self):
        while len(self.assigned_clauses) < len(self.cnf):  # TODO consider better condition
            has_conflict = self.bcp()
            if not has_conflict:
                self.decide_next()
                # TODO - should we check again for false clauses and backjump, or could it lead to infinite loop?
            else:
                learned_clause, bj_level = self.analyze_conflict()
                if bj_level < 0:  # TODO convention for 0-level conflicts, meaning formula is UNSAT
                    return False
                self.cnf.append(learned_clause)
                self.assigned_clauses.append(False)
                self.watch_literals.append([])
                self.update_watch_literals([-1])
                self.perform_backjump(bj_level)
        return True


def dlis(formula, var_assignment, assigned_clauses):
    appearances_dict = defaultdict(int)
    pos_appearances = defaultdict(int)  # DLIS also returns a recommendation for assignment
    neg_appearances = defaultdict(int)
    # Count all appearances of non assigned variables in non assigned clauses (as negated and atomic vars)
    # Return the variable with most appearances, and the assignment that will help satisfying all clauses
    # TODO - maybe act differently - consider each literal by its own and not its abs
    # TODO - meaning we should take max from both neg and pos appearances
    for i, clause in enumerate(formula):
        if not assigned_clauses[i]:
            for literal in clause:
                if abs(literal) not in var_assignment.keys():
                    appearances_dict[abs(literal)] += 1
                    if literal < 0:
                        neg_appearances[abs(literal)] += 1
                    else:
                        pos_appearances[abs(literal)] += 1
    chosen_literal = max(appearances_dict, key=lambda k: appearances_dict[k])
    return chosen_literal, pos_appearances[chosen_literal] > neg_appearances[chosen_literal]


if __name__ == '__main__':
    formula = Not(Imp(Not(And(Atomic('p'), Atomic('q'))), Not(Atomic('r'))))
    # formula = And(Atomic('p'), Atomic('q'))
    f2cnf = FormulaToCNF(formula)
    f2cnf.run()
    cnf = f2cnf.cnf
    print(cnf)
    a = Assignments(cnf)
    print(a.watch_literals)
    a.bcp()
    print(a.var_assignment, a.assigned_clauses)
    print(a.watch_literals)
