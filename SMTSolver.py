from boolean_operators import *
from UF_solver import UFSolver
from LP_theory_solver import LPSolver

GEQ = ">="
LEQ = "<="
EQ = "="
NEQ = "!="
LT = "<"
GT = ">"
PLUS = "+"
MINUS = "-"
PLACE_HOLDER = "?"
UNUSED_VAR = "v"


# Assumption - no variable named v in the formula
# TODO document and make pretty

class SMTSolver:
    supported_theories = ['uf', 'lp']

    def __init__(self, raw_formula: BooleanOperator, theory='uf'):
        assert theory in self.supported_theories
        self.formula = raw_formula
        self.solvers = []
        if theory == 'uf':
            self.solvers.append(UFSolver(self.formula))
        elif theory == 'lp':
            # convert to dnf
            # for clause in dnf, init lp solver
            pass

    def solve(self):
        for solver in self.solvers:
            is_sat, assignment = solver.solve()
            if is_sat:
                return is_sat, assignment
        return False, {}

    def process_q_formula(self, formula):
        nnf = self.run_to_saturation(formula, self.nnf_helper)
        rewrited_nnf = self.rewrite_q_atomics(nnf)
        nnf = self.run_to_saturation(rewrited_nnf, self.nnf_helper)
        dnf_bool_op = self.run_to_saturation(nnf, self.distribute)
        dnf = self.convert_bool_dnf(dnf_bool_op)
        return dnf

    def distribute(self, formula):
        if type(formula) == Atomic:
            return formula
        elif type(formula) == Not:
            return Not(self.distribute(formula.param))
        elif type(formula) == Or:
            return Or(self.distribute(formula.left), self.distribute(formula.right))
        elif type(formula) == Imp:
            return Imp(self.distribute(formula.left), self.distribute(formula.right))
        elif type(formula) == Equiv:
            return Equiv(self.distribute(formula.left), self.distribute(formula.right))
        elif type(formula) == And:
            if type(formula.left) == Or:
                a = formula.left.left
                b = formula.left.right
                c = formula.right
                return self.distribute(Or(And(a, c), And(b, c)))
            elif type(formula.right) == Or:
                a = formula.left
                b = formula.right.left
                c = formula.right.right
                return self.distribute(Or(And(a, b), And(a, c)))
            else:
                return And(self.distribute(formula.left), self.distribute(formula.right))
        return formula

    def rewrite_q_atomics(self, formula):
        if type(formula) == Atomic:
            expr = str(formula)
            if GEQ in expr:
                return Atomic(self.invert_geq_atomic(expr))
            elif NEQ in expr:
                left_formula = self.rewrite_q_atomics(Atomic(expr.replace(NEQ, LT)))
                right_expr = self.invert_geq_atomic(expr.replace(NEQ, GEQ))
                i = right_expr.index(LEQ)
                right_expr = right_expr[:i] + PLUS + UNUSED_VAR + LEQ + right_expr[i + 2:]
                return Or(left_formula, And(Atomic(right_expr), Atomic(UNUSED_VAR + GT + "0")))
            elif EQ in expr:
                return And(Atomic(expr.replace(EQ, LEQ)), Atomic(self.invert_geq_atomic(expr.replace(EQ, GEQ))))
            elif LT in expr:
                i = expr.index(LT)
                expr = expr[:i] + PLUS + UNUSED_VAR + LEQ + expr[i + 1:]
                return And(Atomic(expr), Atomic(UNUSED_VAR + GT + "0"))
            return formula
        elif type(formula) == Not:
            return Not(self.rewrite_q_atomics(formula.param))
        elif type(formula) == And:
            return And(self.rewrite_q_atomics(formula.left), self.rewrite_q_atomics(formula.right))
        elif type(formula) == Or:
            return Or(self.rewrite_q_atomics(formula.left), self.rewrite_q_atomics(formula.right))
        elif type(formula) == Imp:
            return Imp(self.rewrite_q_atomics(formula.left), self.rewrite_q_atomics(formula.right))
        elif type(formula) == Equiv:
            return Equiv(self.rewrite_q_atomics(formula.left), self.rewrite_q_atomics(formula.right))
        else:
            return formula

    def invert_geq_atomic(self, expr):
        assert GEQ in expr
        expr = expr.replace(PLUS, PLACE_HOLDER)
        expr = expr.replace(MINUS, PLUS)
        expr = expr.replace(PLACE_HOLDER, MINUS)
        if expr[0] == PLUS:
            expr = expr[1:]
        else:
            expr = MINUS + expr
        i = expr.index(GEQ) + 2
        if expr[i] == PLUS:
            expr = expr[:i] + expr[i + 1:]
        else:
            expr = expr[:i] + MINUS + expr[i:]
        return expr.replace(GEQ, LEQ)

    def run_to_saturation(self, formula, func):
        to_continue = True
        new = formula
        prev = None
        while to_continue:
            prev = new
            new = func(prev)
            if str(new) == str(prev):
                to_continue = False
        return prev

    def nnf_helper(self, formula):
        if type(formula) == Atomic:
            return formula
        elif type(formula) == And:
            return And(self.nnf_helper(formula.left), self.nnf_helper(formula.right))
        elif type(formula) == Or:
            return Or(self.nnf_helper(formula.left), self.nnf_helper(formula.right))
        elif type(formula) == Imp:
            return self.nnf_helper(Or(Not(formula.left), formula.right))
        elif type(formula) == Equiv:
            return self.nnf_helper(And(Or(Not(formula.left), formula.right), Or(Not(formula.right), formula.left)))
        elif type(formula) == Not:
            if type(formula.param) == Not:
                return self.nnf_helper(self.nnf_helper(formula.param.param))
            elif type(formula.param) == And:
                return self.nnf_helper(Or(Not(formula.param.left), Not(formula.param.right)))
            elif type(formula.param) == Or:
                return self.nnf_helper(And(Not(formula.param.left), Not(formula.param.right)))
            else:
                return Not(self.nnf_helper(formula.param))
        return

    def convert_bool_dnf(self, dnf_bool_op):
        temp = []
        dnf = []
        self.converter(dnf_bool_op, temp, Or)
        for conj in temp:
            clause = []
            self.converter(conj, clause, And)
            dnf.append(clause)
        return dnf

    def converter(self, formula, output_list, input_type):
        if type(formula) == input_type:
            self.converter(formula.left, output_list, input_type)
            self.converter(formula.right, output_list, input_type)
        elif type(formula) == Atomic:
            output_list.append(formula.val)
        else:
            output_list.append(formula)


if __name__ == '__main__':
    # formula = Not(Imp(Atomic("-x+y=1"), And(Atomic("z>=-3"), Atomic("-4x+z!=7"))))
    # formula = Atomic("-2x+2y-z=-3")
    formula = And(Or(Atomic("q1"), Not(Not(Atomic("q2")))), Imp(Not(Atomic("r1")), Atomic("r2")))
    smt = SMTSolver(formula)
    print(smt.process_q_formula(formula))
