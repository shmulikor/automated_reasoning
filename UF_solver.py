from SATSolver import *

LEFT = 0
RIGHT = 1
EQUAL = 2
DRAW_CONG_DAG = False
INVALID_JUMP_LEVEL = -1


def abstractions_generator():
    # Generates names for new variables (starting at 1)
    # Each var is represented by a number
    i = 0
    while True:
        i += 1
        yield i


names = {}
abstractor = abstractions_generator()


class UFVar:
    # A class representing a variable in UF theory
    def __init__(self, var: tuple):
        self.left = var[LEFT]
        self.right = var[RIGHT]
        self.equal = var[EQUAL]

    def __repr__(self):
        return f"{self.left}{'=' if self.equal else '!='}{self.right}"


class UFSolver:
    # Main engine for running the SMT solver for UF theory
    # Assumption - all atomic are of equality form, input is given as BooleanOperator class
    # input is syntactically valid
    # Atomic variables are string representing equality, of form "left=right"

    def __init__(self, raw_formula: BooleanOperator):
        self.boolean_abstraction = {}  # keys are numbers of boolean variables, values are relevant atomics
        self.inv_boolean_abstraction = {}  # inverse of previous dictionary
        self.boolean_formula = []  # The boolean version of the formula
        self.cong_DAG = None
        self.parents = defaultdict(set)
        self.reps = {}
        self.convert_raw_to_cnf(raw_formula)
        self.sat_solver = SATSolver(self.boolean_formula)

    def convert_raw_to_cnf(self, raw_formula):
        # Converts the raw formula to cnf form using Tsietin transformation
        processor = FormulaProcessor(raw_formula)
        processor.tseitin_convert_and_preprocess()
        self.boolean_formula = processor.cnf
        self.boolean_abstraction = processor.atomic_abstractions
        self.inv_boolean_abstraction = {v: k for k, v in self.boolean_abstraction.items()}

    def solve(self):
        # Runs the SMT solver, decides whether the formula is uf-sat
        while self.sat_solver.should_continue() or not self.decide_uf_sat():
            # Continue if the sat solver has not finished, or it has and needs to be corrected by theory
            has_conflict = self.sat_solver.bcp()
            if has_conflict:
                if not self.sat_solver.resolve_conflict():
                    return False, {}  # Logical contradiction (at decision level 0)
            else:
                if self.decide_uf_sat():
                    if self.uf_propagation():  # Perform t-propagation until saturation, and return to BCP
                        continue
                    elif len(self.sat_solver.var_assignment.keys()) != self.sat_solver.variables_num:
                        self.sat_solver.decide_next()
                    else:
                        return True, self.sat_solver.get_atomic_assignments()
                else:  # Not uf-sat, need to learn new clause and backjump
                    clause, jump_level = self.uf_conflict()
                    if jump_level == INVALID_JUMP_LEVEL:
                        return False, {}  # Theory contradiction (at decision level 0)
                    self.sat_solver.add_clause(clause)
                    self.sat_solver.perform_backjump(jump_level)
        return True, self.sat_solver.get_atomic_assignments()

    def decide_uf_sat(self, potential_assignment=None):
        # Decides whether the partial assignment is consistent with the UF theory

        # Compute conjunction of all uf-expression corresponding to assigned variables
        cong_formula = self.compute_partial_assignment_formula(potential_assignment)

        # Get all subterms of the above mentioned expressions
        subterms_set = self.compute_formula_subterms(cong_formula)
        self.create_cong_DAG(subterms_set)
        is_sat = self.process_formula_in_DAG(cong_formula)

        # Revert changes in class fields
        self.cong_DAG = None
        self.parents = defaultdict(set)
        self.reps = {}
        return is_sat

    def uf_conflict(self):
        # Creates the conflict clause from a partial assignment
        # Assumes that the partial assignment indeed leads to a conflict

        # Case of 0 level conflict
        if not self.sat_solver.current_decision_level:
            return [], INVALID_JUMP_LEVEL

        # Create naive clause from partial assignment and decide jump level
        t_learn = [-lit if self.sat_solver.var_assignment[lit] else lit for lit in self.sat_solver.var_assignment.keys()]
        jump_level = self.sat_solver.clause_jump_level(t_learn)
        return t_learn, jump_level

    def compute_partial_assignment_formula(self, potential_assignment=None):
        # Compute conjunction of all uf-expression corresponding to assigned variables
        # Take all variables assigned by internal sat solver, and additional variables as an option

        literals_conjunction = []
        for var in self.sat_solver.var_assignment.keys():
            if var in self.boolean_abstraction:
                literals_conjunction.append(self.parse_atomic(self.boolean_abstraction[var], self.sat_solver.var_assignment[var]))

        if potential_assignment is not None:
            for var in potential_assignment.keys():
                if var in self.boolean_abstraction:
                    literals_conjunction.append(
                        self.parse_atomic(self.boolean_abstraction[var], potential_assignment[var]))

        return literals_conjunction

    def compute_formula_subterms(self, cong_formula):
        # Compute the set of subterms of all parts in the formula (list of UFVars)
        subterms_set = set()
        for var in cong_formula:
            subterms_set.union(self.parse_term_to_subterms(subterms_set, var.left))
            subterms_set.union(self.parse_term_to_subterms(subterms_set, var.right))
        return subterms_set

    def parse_term_to_subterms(self, subterms_set, term):
        # Recursively compute the subterms of a given term

        if not len(term):  # Base case - empty term adds nothing to the set
            return subterms_set

        elif ',' not in term and '(' not in term and ')' not in term:
            # Base case - it is a simple term
            subterms_set.add(term)
            return subterms_set

        elif self.is_func_call(term):  # Step case - function call
            subterms_set.add(term)
            subterm = term[term.index('(') + 1: -1]
            subterms_set = self.parse_term_to_subterms(subterms_set, subterm)
        else:  # Step case - "list" of arguments
            arg_list = self.parse_function_arguments(term)
            for arg in arg_list:
                subterms_set = self.parse_term_to_subterms(subterms_set, arg)
        return subterms_set

    def create_cong_DAG(self, subterms):
        # Create the DAG for the Congruence Closure Algorithm
        cong_DAG = nx.MultiDiGraph()
        cong_DAG.add_nodes_from(subterms)
        for subterm1 in subterms:
            for subterm2 in subterms:
                if subterm1 in subterm2 and subterm1 != subterm2:  # Strictly includes
                    all_idx = [i for i in range(len(subterm2)) if subterm2[i:min(len(subterm2), i + len(subterm1))] ==
                               subterm1]
                    for i in all_idx:  # Indices of all appearances in of subterm 1 in subterm 2
                        if subterm2[:i].count('(') - subterm2[:i].count(')') <= 1:
                            # subterm1 is not an argument of an inner function
                            cong_DAG.add_edge(subterm2, subterm1, kind="subterm")
                            self.parents[subterm1].add(subterm2)
        assert (nx.is_directed_acyclic_graph(cong_DAG))
        for node in cong_DAG:
            # Self loops for representatives
            cong_DAG.add_edge(node, node, kind="rep")
            self.reps[node] = node

        if DRAW_CONG_DAG:
            nx.draw_networkx(cong_DAG)
            plt.show()
        self.cong_DAG = cong_DAG

    def process_formula_in_DAG(self, cong_formula):
        # Processes the conjunction of literals in the DAG, to decide uf-sat
        assert (self.cong_DAG is not None)
        equalities = [var for var in cong_formula if var.equal]
        inequalities = [var for var in cong_formula if not var.equal]
        # Update reps according to all equalities
        for equality in equalities:
            self.process_equality(equality)

        # Check for contradictions in inequalities
        for inequality in inequalities:
            if self.find_rep(inequality.left) == self.find_rep(inequality.right):
                return False
        return True

    def process_equality(self, equality):
        assert (self.cong_DAG is not None)

        # Merge rep classes
        t1 = equality.left
        t2 = equality.right
        rep1 = self.find_rep(t1)
        rep2 = self.find_rep(t2)

        # Get parents lists
        parents1 = self.parents[rep1]
        parents2 = self.parents[rep2]

        # Update parents of representatives
        self.parents[rep2].union(self.parents[rep1])
        self.parents[rep1] = set()
        self.reps[rep1] = rep2

        # Get original parents, recursively parse equalities of congruent parents
        for p1 in parents1:
            for p2 in parents2:
                if self.are_congruent_parents(p1, p2):
                    congruence = UFVar((p1, p2, True))
                    self.process_equality(congruence)

    def find_rep(self, term):
        # Finds representative of a term in the Congruence Closure Algorithm
        # Follows representatives until a self representation
        assert (term in self.reps.keys())
        while self.reps[term] != term:
            term = self.reps[term]
        return term

    def are_congruent_parents(self, p1, p2):
        # Check if two parents in the Congruence Closure Algorithm are congruent

        # Find function name
        f1 = p1[:p1.index('(')]
        f2 = p2[:p2.index('(')]
        if f1 == f2:  # Both parents are call for the same function
            p1 = p1[p1.index('(') + 1: -1]
            p2 = p2[p2.index('(') + 1: -1]
            # Find sets of parameters reps
            # if all are equal - call recursively
            p1_args = self.parse_function_arguments(p1)
            p2_args = self.parse_function_arguments(p2)
            if len(p1_args) == len(p2_args):
                for i in range(len(p1_args)):
                    if self.find_rep(p1_args[i]) != self.find_rep(p2_args[i]):
                        return False
                return True
        return False

    @staticmethod
    def parse_atomic(var, equality):
        # Parse atomic uf-expression to UFVar
        parts = var.split('=')
        left = parts[0]
        right = parts[1]
        return UFVar((left, right, equality))

    @staticmethod
    def is_func_call(term):
        # is the term of the form "func(expr)"?
        return term[-1] == ')' and (not term.count(',') or term.index(',') > term.index('('))

    @staticmethod
    def parse_function_arguments(func_arg):
        # divides arguments "list" into parts, each is a valid term
        arg_list = []
        comma_idx = [i for i, ltr in enumerate(func_arg) if ltr == ','] + [len(func_arg)]
        last_idx = 0
        # Check if each comma is has same number of ( and ) before it. if so, slice the string
        for idx in comma_idx:
            take_index = last_idx if not last_idx else last_idx + 1
            if func_arg[take_index:idx].count('(') == func_arg[take_index:idx].count(')'):
                arg_list.append(func_arg[take_index:idx])
                last_idx = idx
        return arg_list

    def uf_propagation(self):
        # Try assigning every unassigned variable corresponding to an atomic
        # If one assignment results uf-unsat, we can deduce the theory implies the
        # var should be assigned the other option
        # Repeat until saturation
        changed = True
        propagation_counter = 0
        while changed:
            unassigned_vars = [var for var in self.boolean_abstraction.keys() if var not in
                               self.sat_solver.var_assignment.keys()]
            changed = False
            for var in unassigned_vars:
                if not self.decide_uf_sat({var: True}):
                    self.sat_solver.assign_variable(var, False)
                    changed = True
                    propagation_counter += 1
                elif not self.decide_uf_sat({var: False}):
                    self.sat_solver.assign_variable(var, True)
                    changed = True
                    propagation_counter += 1

        return propagation_counter > 0
