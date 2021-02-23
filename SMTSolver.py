import sys
from scipy.optimize import linprog
from FormulaProcessor import *
from UFSolver import UFSolver
from LPSolver import LPSolver
from SATSolver import SATSolver
import os

# The main module for the project, which calls all other modules (and tests them)
# Testing is hardcoded in specified functions

# For all parts - Input is assumed to be a BooleanOperator formula, with atomics carrying a string
# With theory signature (or anything in case of propositional theory).

# SAT Assumption - input can be parsed from a cnf file, without first converted to formula
# (and without applying Tseitin transformation).
# BooleanOperator formula is converted using Tseitin transformation

# UF Assumptions - all atomics are of equality form, inequality is expressed using the Not class
# Atomic string are syntactically correct and accurate, with all function in their right arity

# LP Assumptions:
# Atomics have tq formulas as their strings
# Input adheres the standard form, with correct dimensions of input objects
# (all variables are assumed to be >= 0), so no strict inequalities appear
# No variable named v in the formula - as it is used for conversion
# Variables names don't contain numbers (e.g. no x0)
# All relevant variables are included in all clauses, with the same order
# All coefficients are stated explicitly (including +-1, 0)
# Unless inserted as an argument, objective function is the all-ones vector


supported_theories = ['boolean', 'uf', 'lp']


class SMTSolver:
    # A class representing a SMT solver, which supports theories corresponding to each
    # part of the project - pure boolean, uninterpreted functions, and lp
    # Class assumes correct input, as described above and does not support non-boolean theory combinations

    def __init__(self, raw_formula, theory='boolean', lp_objective=None, boolean_is_cnf=True):
        assert theory in supported_theories
        self.formula = raw_formula
        self.solvers = []  # Might need several solvers in case of LP
        self.processor = FormulaProcessor(raw_formula)
        if theory == 'uf':
            self.solvers.append(UFSolver(self.formula))
        elif theory == 'lp':
            dnf = self.processor.process_q_formula()
            for clause in dnf:
                A, b = self.processor.convert_clause_to_lp(clause)
                c = np.array(A.shape[1] * [1]) if lp_objective is None else lp_objective
                self.solvers.append(LPSolver(A, b, c))
        elif theory == 'boolean':
            self.solvers.append(SATSolver(raw_formula, boolean_is_cnf))

    def solve(self):
        # Decides satisfiability of the formula using the theory solvers
        for solver in self.solvers:
            is_sat, assignment = solver.solve()
            if is_sat:
                return is_sat, assignment
        return False, {}


def SAT_solver_test(check_sat=False, hardcoded=False):
    # Runs tests for SAT solver
    # Using a hardcoded formula or reads from cnf files
    if hardcoded:
        formula = Not(Imp(Not(And(Atomic('p'), Atomic('q'))), Not(Atomic('r'))))
        result, assignment = SMTSolver(formula, "boolean", boolean_is_cnf=False).solve()
        print(assignment)
    else:
        directory = "SAT_examples" if check_sat else "UNSAT_examples"
        run_cnf_files(directory, check_sat)


def run_cnf_files(directory, check_sat):
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
                solution = SMTSolver(cnf, "boolean", boolean_is_cnf=True)
                if check_sat:
                    result, assignment = solution.solve()
                    assert result
                    if DEBUG:
                        print(assignment)
                else:
                    assert (not solution.solve()[0])
                completed += 1
        except UnicodeDecodeError:
            errors += 1
            continue
    print("Checked ", completed, " files\n", errors, " files were not checked due to decoding errors")


def uf_test():
    # Tests several UF formulas, given in a list of hardcoded formulas
    examples = [
        Or(And(Atomic("f(a)=f(b)"), Atomic("b=c")), Atomic('g(a,f(a,k(b),f(c)))=d')),  # True
        Atomic("f(f(a,b),a)=f(c,d)"),  # True
        And(Atomic("a=b"), And(Or(Or(Not(Atomic("a=b")), Not(Atomic("s=t"))), Atomic("b=c")),
                               And(Or(Or(Atomic("s=t"), Not(Atomic("t=r"))), Atomic("f(s)=f(t)")),
                                   And(Or(Or(Not(Atomic("b=c")), Not(Atomic("t=r"))), Atomic("f(s)=f(a)")),
                                       Or(Not(Atomic("f(s)=f(a)")), Not(Atomic("f(a)=f(c)"))))))),  # True
        And(Atomic("f(g(x))=g(f(x))"), And(Atomic("f(g(f(y)))=x"), And(Atomic("f(y)=x"),
                                                                       Not(Atomic("g(f(x))=x"))))),  # False
        And(Atomic("a=b"), And(Atomic("b=c"), Or(Atomic("d=e"), Or(Atomic("a=c"), Atomic("f=g"))))),  # True
        And(Atomic("g(a)=c"), And(Or(Not(Atomic("f(g(a))=f(c)")), Atomic("g(a)=d")), Not(Atomic("c=d")))),  # False
        And(Or(Atomic("g(a)=c"), Atomic("x=y")), And(Or(Not(Atomic("f(g(a))=f(c)")), Atomic("g(a)=d")),
                                                     And(Not(Atomic("c=d")), Not(Atomic("f(x)=f(y)"))))),  # False
        And(Not(Atomic("x=y")), Atomic("f(x)=f(y)")),  # True
        And(Atomic("f(a)=a"), Not(Atomic("f(f(a))=a"))),  # False
        And(Atomic("f(f(f(a)))=a"), And(Atomic("f(f(f(f(f(a)))))=a"), Not(Atomic("f(a)=a")))),  # False
        Or(Not(Atomic("x=g(y,z)")), Atomic("f(x)=f(g(y,z))")),  # True
        And(Atomic("a=b"), And(Atomic("f(c)=c"), Atomic("f(a)=b"))),  # True
        And(Atomic("f(a,b)=a"), Not(Atomic("f(f(a,b),b)=a"))),  # False
        And(Not(Atomic("s=x")), And(Atomic("g(f(z))=s"), And(Atomic("g(f(y))=x"), Atomic("y=z")))),  # False
        And(Or(Atomic("f(x)=f(y)"), Atomic("a=b")), Atomic("x=y")),  # True
        And(Not(Atomic("f(x)=f(y)")), And(Atomic("y=x"), Atomic("a=b"))),  # False
        And(Not(Atomic("f(f(x))=f(f(y))")), And(Atomic("f(x)=f(y)"), Atomic("x=y")))  # False
    ]

    desired_results = [True, True, True, False, True, False, False, True, False, False, True, True, False, False, True,
                       False, False]

    for i in range(len(examples)):
        uf = SMTSolver(examples[i], "uf")
        print(uf.solvers[0].boolean_abstraction)
        assert (uf.solve()[0] == desired_results[i])


def lp_test():
    # Tests several LP instances, given in a list of (A,b,c)
    examples = [
        (np.array([[1, 1, 2],
                   [2, 0, 3],
                   [2, 1, 3]]),
         np.array([4, 5, 7]),
         np.array([3, 2, 4])),

        (np.array([[3, 2, 1, 2],
                   [1, 1, 1, 1],
                   [4, 3, 3, 4]]),
         np.array([225, 117, 420]),
         np.array([19, 13, 12, 17])),

        (np.array([[2, 2, -1],
                   [3, -2, 1],
                   [1, -3, 1]]),
         np.array([10, 10, 10]),
         np.array([1, 3, -1])),

        (np.array([[1, -1],
                   [-1, -1],
                   [2, 1]]),
         np.array([-1, -3, 4]),
         np.array([3, 1])),

        (np.array([[1, -1],
                   [-1, -1],
                   [2, 1]]),
         np.array([-1, -3, 2]),
         np.array([3, 1])),

        (np.array([[1, -1],
                   [-1, -1],
                   [2, -1]]),
         np.array([-1, -3, 2]),
         np.array([3, 1])),

        (np.array([[-1, 1],
                   [-2, -2],
                   [-1, 4]]),
         np.array([-1, -6, 2]),
         np.array([1, 3])
         ),

        (np.array([[1, 2, 3, 1],
                   [1, 1, 2, 3]]),
         np.array([5, 3]),
         np.array([5, 6, 9, 8])),

        (np.array([[2, 3],
                   [1, 5],
                   [2, 1],
                   [4, 1]]),
         np.array([3, 1, 4, 5]),
         np.array([2, 1])),

        (np.array([[1, -2],
                   [1, -1],
                   [2, -1],
                   [1, 0],
                   [2, 1],
                   [1, 1],
                   [1, 2],
                   [0, 1], ]),
         np.array([1, 2, 6, 5, 16, 12, 21, 10]),
         np.array([3, 2])),

        # Klee minty
        (np.array([[1, 0, 0],
                   [20, 1, 0],
                   [200, 20, 1]]),
         np.array([1, 100, 10000]),
         np.array([100, 10, 1])),

        (np.array([[-2, - 1, 3, 1.2],
                   [0, - 0, - 0, - 1]]),
         np.array([-3, 0]),
         np.array([1, 1, 1, 1]))
    ]

    for A, b, c in examples:
        print("Solving: max <{}, x> s.t\n {}x<={}".format(c, A, b))
        # Print scipy-solver results for comparison
        res = linprog(c=-c, A_ub=A, b_ub=b)  # Solves min -c problem, need to negate results
        print("Scipy solver result status: {}\nObjective value {}".format(res['message'],-res['fun']))

        lp_solver = LPSolver(A, b, c)
        lp_solver.solve()
        print("Final assignment:", lp_solver.assignment)
        print("Objective value:", lp_solver.objective)
        print("Sol type: ", lp_solver.solution_type)
        print("####################################")


def tq_test():
    # Tests conversion of a tq_formula to a valid standard form LP
    formula = And(Or(Atomic("-2xx-1y+3z+1.2t<=-3"), Not(Not(Atomic("1xx+1y+1z-1t<=2.7")))), Imp(Not(Atomic(
        "0xx-0y-0z-1t<=0")), Atomic("20xx-40y-80z-13t<=2000")))
    smt = SMTSolver(formula, 'lp')
    print(smt.solve())


if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] not in supported_theories:
        print("USAGE: SMTSolver.py [theory], where theory is one of 'boolean', 'uf', 'lp'")
    elif sys.argv[1] == 'boolean':
        SAT_solver_test(hardcoded=True)
        SAT_solver_test(check_sat=True)
        SAT_solver_test(check_sat=False)
    elif sys.argv[1] == 'uf':
        uf_test()
    elif sys.argv[1] == 'lp':
        lp_test()
        tq_test()

