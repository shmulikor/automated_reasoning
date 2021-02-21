import sys
from collections import defaultdict
from boolean_operators import *

import networkx as nx
import matplotlib.pyplot as plt
import os
from itertools import chain

CONFLICT_NODE = 0
DRAW_IMP_GRAPH = False
CHECK_SAT = True


# Assumption - var numbers are given in continuous manner

class SATSolver:
    # A class representing the SAT solver for the first part of the project
    # The class handles a formula in cnf form (in our convention - list of lists of numbers)
    # and solves the corresponding SAT problem

    def __init__(self, cnf):
        self.cnf = cnf
        self.variables_num = len(set(chain(*[np.abs(c) for c in self.cnf])))
        self.var_assignment = {}
        self.assigned_clauses = [False] * len(cnf)
        self.decision_levels = {}
        self.splits = []
        self.current_decision_level = 0
        self.bcp_implications = {}  # Used for implications graph. Keys are literals, values are clauses implied from
        self.watch_literals = []
        self.init_watch_literals()
        self.cur_conflict_clause = 0
        self.cur_bcp_candidates = []

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
        # For every index in the indices lists, find at most two unassigned variables
        for index in indices_to_update:
            lit_set = set(self.cnf[index])
            clause_watch_literals = []
            while len(lit_set) > 0 and len(clause_watch_literals) < 2:
                lit = lit_set.pop()
                if abs(lit) not in self.var_assignment.keys():
                    clause_watch_literals.append(lit)
            self.watch_literals[index] = clause_watch_literals

    def solve(self):
        # Runs the main CDCL algorithm
        while self.should_continue():
            # While not everything was assigned, perform BCP (to saturation) and solve conflicts if any has risen
            has_conflict = self.bcp()
            if not has_conflict and len(self.var_assignment.keys()) != self.variables_num:
                self.decide_next()
            elif has_conflict:
                if not self.resolve_conflict():
                    return False, []
        return True, self.var_assignment

    def should_continue(self):
        return (not all(self.assigned_clauses)) or len(self.var_assignment.keys()) != self.variables_num or \
                                                     self.has_false_clause()

    def resolve_conflict(self):
        # Resolves the conflict in the assignment, assuming there is one
        # Returns false iff current decision level is 0
        assert (self.has_false_clause())
        if not self.current_decision_level:
            return False
        learned_clause, jump_level = self.analyze_conflict()
        self.add_clause(learned_clause)
        self.perform_backjump(jump_level)
        return True

    def bcp(self):
        # Performs BCP until saturation, returns True iff formula has a falsified clause
        self.cur_bcp_candidates = [self.cnf[i] for i, watchers in enumerate(self.watch_literals) if len(watchers) ==
                                   1]  # In case there's no last decided literal, pick all possible candidates
        if len(self.splits) > 0:
            last = self.splits[-1]
            self.cur_bcp_candidates = self.get_bcp_candidates(last if self.var_assignment[last] else -last)
            involved_indices = [i for i, clause in enumerate(self.cnf) if last in self.cnf[i] or -last in self.cnf[i]]
            # Update watch literals for every clause with decided literal
            self.update_watch_literals(involved_indices)

        changed = True
        while changed:
            changed = False
            for clause in self.cur_bcp_candidates:
                i = self.cnf.index(clause)
                if not self.assigned_clauses[i]:  # Assign if not assigned previously
                    if all([abs(l) in self.var_assignment.keys() for l in clause]):
                        self.assigned_clauses[i] = True
                    else:
                        unassigned = [l for l in clause if abs(l) not in self.var_assignment.keys()]
                        if len(unassigned) == 1:  # If only single variable is indeed unassigned
                            assigned = [l for l in clause if abs(l) in self.var_assignment.keys()]
                            past_assignments = [self.var_assignment[l] for l in assigned if l > 0] + \
                                               [not self.var_assignment[-l] for l in assigned if l < 0]
                            if not any(past_assignments):  # If all other literals in clause are false
                                self.bcp_updates(clause, unassigned[0])
                                changed = True
        self.cur_bcp_candidates = []
        return self.has_false_clause()

    def get_bcp_candidates(self, literal):
        # Returns sublist of all clauses which include the negation of the literal as a watch literal
        # Add clauses with only one watch literal as well #TODO fix
        # As stated in class, these are only ones relevant for BCP after assignment
        indices = [i for i, watchers in enumerate(self.watch_literals) if -literal in watchers]
        indices.extend([i for i, watchers in enumerate(self.watch_literals) if len(watchers) == 1])
        return [self.cnf[j] for j in indices]

    def bcp_updates(self, clause, chosen_lit):
        # Performs all updates to internal data structures corresponding to BCP action
        i = self.cnf.index(clause)
        self.assign_variable(abs(chosen_lit), True if chosen_lit > 0 else False)
        self.bcp_implications[abs(chosen_lit)] = (i, self.current_decision_level)

    def has_false_clause(self):
        # returns true iff the formula has a falsified clause
        return any([self.false_literals_in_clause(clause) == len(clause) for clause in self.cnf])

    def false_literals_in_clause(self, clause):
        # Count the number of literals that have an assignment and it contradicts the literal
        return len([l for l in clause if abs(l) in self.var_assignment.keys() and
                    ((l < 0 and self.var_assignment[abs(l)])
                     or
                     (l > 0 and not self.var_assignment[l]))])

    def decide_next(self):
        # Splits cases by using DLIS heuristic, and updates all relevant fields
        # to support non-chronological backjumps
        assert (self.variables_num != len(self.var_assignment.keys()))
        var, assign = dlis(self.cnf, self.var_assignment, self.assigned_clauses)
        assert (var > 0)
        self.splits.append(var)
        self.var_assignment[var] = assign
        self.current_decision_level += 1
        self.decision_levels[var] = self.current_decision_level
        involved_indices = [i for i, clause in enumerate(self.cnf) if var in self.cnf[i] or -var in self.cnf[i]]
        for i in involved_indices:
            if all([abs(l) in self.var_assignment.keys() for l in self.cnf[i]]):
                self.assigned_clauses[i] = True

    def perform_backjump(self, level):
        # Performs non-chronological backjump to a specified decision level
        assert (level >= 0)
        self.splits = self.splits[:level]
        self.current_decision_level = level
        self.var_assignment = {k: v for k, v in self.var_assignment.items() if self.decision_levels[k] <= level}
        self.decision_levels = {k: v for k, v in self.decision_levels.items() if self.decision_levels[k] <= level}
        self.update_watch_literals(range(len(self.cnf)))  # Update watch literals for all
        for i, has_assigned in enumerate(self.assigned_clauses):  # Updates assigned clauses, can only make
            # previously assigned clauses to become unassigned
            if has_assigned and any([abs(l) not in self.var_assignment.keys() for l in self.cnf[i]]):
                self.assigned_clauses[i] = False
        self.bcp_implications = {k: v for k, v in self.bcp_implications.items() if v[1] <= level}

    def analyze_conflict(self):
        # Creates implications graph and decides using UIP the backjump level and the learned clause
        assert self.current_decision_level
        imp_graph = self.create_implications_graph()
        FUIP = self.find_first_UIP(imp_graph)
        conflict_clause = self.learn_clause(imp_graph, FUIP)
        self.cur_conflict_clause = -1  # Reverse after solving the conflict
        jump_level = self.clause_jump_level(conflict_clause)
        # Finds second highest decision level in learned clause
        return conflict_clause, jump_level

    def create_implications_graph(self):
        # Create an implication graph for the conflict
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
        # Root nodes are decided literals or 0-level bcp implications
        while len(newly_added):
            update = set()
            for var in newly_added:
                # Check for more nodes if current var is not decided and implied by bcp in level > 0
                if var in self.bcp_implications.keys() and self.bcp_implications[var][1] > 0:
                    clause = self.cnf[self.bcp_implications[var][0]]
                    lit = var if self.var_assignment[var] else -var
                    for implier in clause:
                        if abs(implier) != var:
                            imp_graph.add_node(-implier)
                            imp_graph.add_edge(-implier, lit, weight=self.bcp_implications[var][0])
                            update.add(abs(implier))
            newly_added = update
        if DRAW_IMP_GRAPH:
            nx.draw_networkx(imp_graph, pos=nx.circular_layout(imp_graph))
            plt.show()
        assert (nx.algorithms.is_directed_acyclic_graph(imp_graph))
        return imp_graph

    def find_first_UIP(self, imp_graph):
        # Returns the first UIP in the graph
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
        return first_UIP

    def learn_clause(self, imp_graph, FUIP):
        # Computes the learned clause in the graph according to the first UIP
        # Performs algorithm learned in class
        cur_clause = self.cnf[self.cur_conflict_clause]

        # FUIP should be the only literal in the learned clauses assigned in its level
        while (-FUIP not in cur_clause) or \
                self.num_literals_in_decision_level(self.decision_levels[abs(FUIP)], cur_clause) > 1:
            # Find a literal in the maximal decision level that has an incoming edge
            max_level = -1
            for lit in cur_clause:
                if self.decision_levels[abs(lit)] > max_level:
                    max_level = self.decision_levels[abs(lit)]

            max_level_literals = [-literal for literal in cur_clause if self.decision_levels[abs(literal)] == max_level
                                  and literal != -FUIP]
            in_edges = list(imp_graph.in_edges([max_level_literals[0]], True))
            incoming_clause_index = in_edges[0][2]['weight']
            # Resolve current clause and the clause implying the max-level literal found
            cur_clause = boolean_res(self.cnf[incoming_clause_index], cur_clause)
            assert (cur_clause is not None)
        return cur_clause

    def num_literals_in_decision_level(self, level, clause):
        # Counts the number of literals in a clause decided in the given level
        count = 0
        for lit in clause:
            if abs(lit) in self.decision_levels.keys() and self.decision_levels[abs(lit)] == level:
                count += 1
        return count

    def add_clause(self, clause):
        # Add a clause to the formula and update all relevant data structures
        if clause in self.cnf:
            print(clause)
        assert (clause not in self.cnf)
        self.cnf.append(clause)
        self.assigned_clauses.append(False)
        self.watch_literals.append([])
        self.update_watch_literals([-1])

    def clause_jump_level(self, conflict_clause):
        # computes the jump level of a clause
        cc_decision_levels = []
        for lit in conflict_clause:
            cc_decision_levels.append(self.decision_levels[abs(lit)])
        cc_decision_levels = list(set(cc_decision_levels))
        cc_decision_levels.sort()
        return cc_decision_levels[-2] if len(cc_decision_levels) > 1 else 0

    def assign_variable(self, var, assignment):
        # Assign a variable with given assignment
        # Updates all relevant data structures
        self.var_assignment[var] = assignment
        self.decision_levels[var] = self.current_decision_level
        involved_indices = [j for j, clause in enumerate(self.cnf) if
                            var in self.cnf[j] or -var in self.cnf[j]]
        candidates = self.get_bcp_candidates(var if assignment else -var)
        for candidate in candidates:
            if candidate not in self.cur_bcp_candidates:
                self.cur_bcp_candidates.append(candidate)
        self.update_watch_literals(involved_indices)
        for j in involved_indices:  # New clauses can be assigned
            if all([abs(l) in self.var_assignment.keys() for l in self.cnf[j]]):
                self.assigned_clauses[j] = True


def dlis(formula, var_assignment, assigned_clauses):
    # Decides the next variable assignment according to DLIS heuristic
    appearances_dict = defaultdict(int)
    # Count all appearances of non assigned variables in non assigned clauses (as negated and atomic vars)
    # Return the variable with most appearances, and the assignment that will help satisfying all clauses
    for i, clause in enumerate(formula):
        if not assigned_clauses[i]:
            for literal in clause:
                if abs(literal) not in var_assignment.keys():
                    appearances_dict[literal] += 1
    chosen_literal = max(appearances_dict, key=lambda k: appearances_dict[k])
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


def run_cnf_files(directory):
    # Parses cnf files into our formula convention, and solves the sat problem they describe
    files = os.listdir(directory)
    errors = 0
    completed = 0
    for file in files:
        try:
            with open(os.path.join(directory, file)) as formula_file:
                cnf = []
                for line in formula_file.readlines():
                    if len(line.strip()) and (not line.strip()[0].isalpha() and not line.strip()[0] == '%'):
                        to_add = [int(num) for num in line.split()]
                        if to_add[-1] == 0:
                            to_add.pop()
                        if len(to_add):
                            cnf.append(to_add)
                print("Solving: ", os.path.join(directory, file))
                solution = SATSolver(cnf)
                if CHECK_SAT:
                    assert (solution.solve()[0])
                else:
                    assert (not solution.solve()[0])
                completed += 1
        except UnicodeDecodeError:
            errors += 1
            continue
    print("Checked ", completed, " files\n", errors, " files were not checked due to decoding errors")


if __name__ == '__main__':
    if len(sys.argv) and sys.argv[1] == "0":  # Toggle between hard-coded formula and reading from CNF files
        formula = Not(Imp(Not(And(Atomic('p'), Atomic('q'))), Not(Atomic('r'))))
        f2cnf = FormulaProcessor(formula)
        f2cnf.convert_and_preprocess()
        print(SATSolver(f2cnf.cnf).solve())
    else:
        directory = "SAT_examples" if CHECK_SAT else "UNSAT_examples"
        run_cnf_files(directory)
