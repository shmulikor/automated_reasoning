import numpy as np
from collections import defaultdict
from boolean_operators import *
import networkx as nx
import matplotlib.pyplot as plt

CONFLICT_NODE = 0


# TODO make imp garph and UIP fields

class Assignments:
    def __init__(self, cnf):
        self.cnf = cnf
        self.variables_num = max([max(np.abs(clause)) for clause in self.cnf])
        self.var_assignment = {}
        self.assigned_clauses = [False] * len(cnf)
        self.decision_levels = {}
        self.splits = []
        self.current_decision_level = 0
        self.bcp_implications = {}  # Used for implications graph. Keys are literals, values are clauses implied from
        self.watch_literals = []
        self.init_watch_literals()
        self.cur_conflict_clause = 0

    def init_watch_literals(self):
        # Upon initialization - all literals are unassigned
        # Choose two random literals from each clause (or one, if there is only one)
        for i, clause in enumerate(self.cnf):
            self.watch_literals.append([])
            lit_set = set(clause)
            if len(lit_set) <= 2:
                self.watch_literals[i] = list(lit_set)
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
        # TODO change back
        # if len(self.splits) > 0:
        #     bcp_candidates = self.get_bcp_candidates(self.splits[-1] if self.var_assignment[self.splits[-1]]
        #                                              else -self.splits[-1])
        # else:
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
                                self.bcp_implications[abs(chosen_var)] = (i, self.current_decision_level)
                                changed = True
                                involved_indices = [j for j, clause in enumerate(self.cnf) if
                                                    chosen_var in self.cnf[j] or -chosen_var in self.cnf[j]]
                                bcp_candidates = self.get_bcp_candidates(chosen_var)
                                self.update_watch_literals(involved_indices)
                                for i in involved_indices:
                                    if all([abs(l) in self.var_assignment.keys() for l in self.cnf[i]]):
                                        self.assigned_clauses[i] = True
        return self.has_false_clause()

    def has_false_clause(self):
        # returns true iff the formula has a falsified clause
        for clause in self.cnf:
            # Counts the number of literals assigned to false
            if self.false_literals_in_clause(clause) == len(clause):
                return True
        return False

    def false_literals_in_clause(self, clause):
        return len([l for l in clause if ((l < 0 and abs(l) in self.var_assignment.keys() and
                                           self.var_assignment[abs(l)])
                                          or
                                          (l > 0 and l in self.var_assignment.keys() and
                                           not self.var_assignment[l]))])

    def decide_next(self):
        # Splits cases by using DLIS heuristic, and updates all relevant fields
        # to support non-chronological backjumps
        if len(self.var_assignment.keys()) == self.variables_num:
            return

        var, assign = dlis(self.cnf, self.var_assignment, self.assigned_clauses)
        print("dlis decided to put ", assign, "in var: ", var)
        assert (var > 0)
        self.splits.append(var)
        self.var_assignment[var] = assign
        self.current_decision_level += 1
        self.decision_levels[var] = self.current_decision_level
        involved_indices = [i for i, clause in enumerate(self.cnf) if var in self.cnf[i] or -var in self.cnf[i]]
        # Update watch literals for every clause with decided literal
        self.update_watch_literals(involved_indices)  # TODO, maybe create another list of relevant indices
        # Update assigned clauses
        involved_indices = [i for i, clause in enumerate(self.cnf) if var in self.cnf[i] or -var in self.cnf[i]]
        for i in involved_indices:
            if all([abs(l) in self.var_assignment.keys() for l in self.cnf[i]]):
                self.assigned_clauses[i] = True

    def perform_backjump(self, level):
        assert (level >= 0)
        # Performs non-chronological backjump to a specified decision level
        self.splits = self.splits[:level]
        self.current_decision_level = level
        self.var_assignment = {k: v for k, v in self.var_assignment.items() if self.decision_levels[k] <= level}
        self.decision_levels = {k: v for k, v in self.decision_levels.items() if self.decision_levels[k] <= level}
        self.update_watch_literals(range(len(self.cnf)))  # Update watch literals for all (TODO improve from naive way?)
        for i, has_assigned in enumerate(self.assigned_clauses):  # Updates assigned clauses, can only make
            # previously assigned clauses to become unassigned
            if has_assigned:
                if any([abs(l) not in self.var_assignment.keys() for l in self.cnf[i]]):
                    self.assigned_clauses[i] = False
        self.bcp_implications = {k: v for k, v in self.bcp_implications.items() if v[1] <= level}

    def create_implications_graph(self):
        imp_graph = nx.DiGraph()
        imp_graph.add_node(CONFLICT_NODE)  # Conflict node numbered as 0
        newly_added = set()
        # Add nodes for all falsified clauses
        for i, clause in enumerate(self.cnf):
            if self.assigned_clauses[i] and self.false_literals_in_clause(clause) == len(clause):
                for literal in clause:
                    imp_graph.add_node(-literal)
                    imp_graph.add_edge(-literal, CONFLICT_NODE, weight=i)
                    newly_added.add(abs(literal))
                self.cur_conflict_clause = i
                break

        # Repeat until convergence using implications and decision levels (root nodes):
        # Add edges from nodes in the graph implied by bcp steps
        # Weight of edges is the index of the clauses from which the implication is deduced
        # 0 level bcp should also be roots as well
        while len(newly_added) > 0:
            update = set()
            for var in newly_added:
                if var in self.bcp_implications.keys():
                    clause = self.cnf[self.bcp_implications[var][0]]
                    lit = var if self.var_assignment[var] else -var
                    for implier in clause:
                        if (abs(implier) != var):
                            imp_graph.add_node(-implier)
                            imp_graph.add_edge(-implier, lit, weight=self.bcp_implications[var][0])
                            update.add(abs(implier))
            newly_added = update
        nx.draw_networkx(imp_graph)
        print(self.var_assignment)
        print("splits is: ", self.splits)
        #plt.show()
        assert (nx.algorithms.is_directed_acyclic_graph(imp_graph))
        return imp_graph

    def analyze_conflict(self):
        # Creates implications graph and decides using UIP the backjump level and the learned clause
        imp_graph = self.create_implications_graph()
        FUIP = self.find_first_UIP(imp_graph)
        conflict_clause = self.learn_clause(imp_graph, FUIP)
        self.cur_conflict_clause = 0

        cc_decision_levels = []
        for lit in conflict_clause:
            cc_decision_levels.append(self.decision_levels[abs(lit)])
        cc_decision_levels = list(set(cc_decision_levels))
        cc_decision_levels.sort()
        jump_level = cc_decision_levels[-2] if len(cc_decision_levels) > 1 else 0
        print("jump to level: ", jump_level)
        return conflict_clause, jump_level

    def find_first_UIP(self, imp_graph):
        # Returns the last decision point in the graph
        # TODO maybe we should assert it is the last decision?
        # current_decision = CONFLICT_NODE
        #
        # for decision in reversed(self.splits):
        #     assert (decision in self.var_assignment.keys())
        #     lit = decision if self.var_assignment[decision] else -decision
        #     if imp_graph.__contains__(lit):
        #         current_decision = lit
        #         break
        # assert (current_decision != CONFLICT_NODE)
        current_decision = self.splits[-1]
        current_decision = current_decision if self.var_assignment[current_decision] else -current_decision
        assert (imp_graph.__contains__(current_decision))

        # Create a set of all nodes in all paths from current decision to the conflict node
        paths = nx.all_simple_paths(imp_graph, current_decision, CONFLICT_NODE)
        UIPS = set(next(paths))
        for path in paths:
            UIPS = UIPS.intersection(set(path))

        # If conflict node is the only UIP, it is the first
        if CONFLICT_NODE in UIPS and len(UIPS) == 1:
            return CONFLICT_NODE

        # Otherwise pick the UIP closest to conflict node, which is not the conflict node
        UIPS.remove(CONFLICT_NODE)
        distance = float('inf')
        first_UIP = CONFLICT_NODE
        for node in UIPS:
            temp_distance = len(nx.bidirectional_shortest_path(imp_graph, node, CONFLICT_NODE))
            if temp_distance < distance:
                distance = temp_distance
                first_UIP = node
        print("First UIP is : ", first_UIP)  # TODO remove
        return first_UIP

    def learn_clause(self, imp_graph, FUIP):
        cur_clause = self.cnf[self.cur_conflict_clause]
        while (-FUIP not in cur_clause):
            max_level = -1
            last_lit = 0
            for lit in cur_clause:
                if self.decision_levels[abs(lit)] > max_level:
                    max_level = self.decision_levels[abs(lit)]
                    last_lit = lit
            in_edges = imp_graph.in_edges([last_lit], True)
            incoming_clause_index = in_edges[0][2]['weight']
            cur_clause = boolean_res(self.cnf[incoming_clause_index], cur_clause)
            assert (cur_clause is not None)
        print('Learned clause is:', cur_clause)
        return cur_clause

    def solve(self):
        while len(self.var_assignment.keys()) != self.variables_num or self.has_false_clause():  # TODO consider
        # better condition
            has_conflict = self.bcp()
            if not has_conflict:
                print("Current assignment: ", self.var_assignment)
                print("Assigned Clauses: ", self.assigned_clauses)
                print(self.has_false_clause())
                self.decide_next()
            else:
                if self.current_decision_level == 0:
                    return False, []
                learned_clause, jump_level = self.analyze_conflict()
                self.cnf.append(learned_clause)
                self.assigned_clauses.append(False)
                self.watch_literals.append([])
                self.update_watch_literals([-1])
                self.perform_backjump(jump_level)
                print(self.cnf)
        return True, self.var_assignment


def dlis(formula, var_assignment, assigned_clauses):
    appearances_dict = defaultdict(int)
    # Count all appearances of non assigned variables in non assigned clauses (as negated and atomic vars)
    # Return the variable with most appearances, and the assignment that will help satisfying all clauses
    for i, clause in enumerate(formula):
        if not assigned_clauses[i]:
            for literal in clause:
                if abs(literal) not in var_assignment.keys():
                    appearances_dict[literal] += 1
    chosen_literal = max(appearances_dict, key=lambda k: appearances_dict[k])
    print("appearances_dict ", appearances_dict)
    return abs(chosen_literal), chosen_literal > 0


def boolean_res(clause1, clause2):
    # Returns the resolvent of two clauses - if one includes literal l, and the other -l
    # Then the resolvent is a disjunction of all other literals in both clauses
    # Assuming they are both a list of literals (representing a disjunctions of them)
    # Returns None if the clauses don't contain a literal and its negation
    for lit in clause1:
        if -lit in clause2:
            part1 = set(clause1)  # Use set to immediately remove redundant literals
            part1.remove(lit)
            part2 = set(clause2)
            part2.remove(-lit)
            return list(part1.union(part2))
    return None


if __name__ == '__main__':
    # formula = Not(Imp(Not(And(Atomic('p'), Atomic('q'))), Not(Atomic('r'))))
    # formula = And(Atomic('p'), Atomic('q'))
    # f2cnf = FormulaToCNF(formula)
    # f2cnf.run()
    # cnf = f2cnf.cnf
    # print("cnf: ", cnf)
    formula = open("3color_formula.txt")
    cnf = []
    for line in formula.readlines():
        if len(line) > 0:
            cnf.append([int(num) for num in line.split()])
    print(cnf)

    a = Assignments(cnf)
    # print(a.watch_literals)
    # print(a.var_assignment, a.assigned_clauses)
    # print(a.watch_literals)
    print(a.solve())
