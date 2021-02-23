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
            print(self.name , " : " , self.left.name , " and " , self.right.name)

    def __repr__(self):
        return "(" + self.left.__repr__() + " and " + self.right.__repr__() + ")"


class Or(BooleanOperator):
    # A class representing the Or boolean operator

    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        if DEBUG:
            print(self.name , " : " , self.left.name , " or " , self.right.name)

    def __repr__(self):
        return "(" + self.left.__repr__() + " or " + self.right.__repr__() + ")"


class Not(BooleanOperator):
    # A class representing the Not boolean operator

    def __init__(self, param: BooleanOperator):
        if type(param) == Atomic:
            super().__init__(-param.name)
        else:
            super().__init__(next(counter))
        self.param = param
        if DEBUG:
            print(self.name, " : not", self.param.name)

    def __repr__(self):
        return "(not " + self.param.__repr__() + ")"


class Imp(BooleanOperator):
    # A class representing the Implication boolean operator

    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        if DEBUG:
            print(self.name , " : " , self.left.name , " -> " , self.right.name)

    def __repr__(self):
        return "(" + self.left.__repr__() + " -> " + self.right.__repr__() + ")"



class Equiv(BooleanOperator):
    # A class representing the Equivalence boolean operator

    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        if DEBUG:
            print(self.name , " : " , self.left.name , " <=> " , self.right.name)

    def __repr__(self):
        return "(" + self.left.__repr__() + " <=> " + self.right.__repr__() + ")"


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
            print (self.name, " : ", self.val)

    def __repr__(self):
        return self.val

class FormulaProcessor:
    # A class responsible for processing a general formula

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
    DOT = "."

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

    def tseitin_preprocessing(self):
        # Removes redundant literals and trivial clauses from Tseitin formula
        processed_tseitin = []
        for clause in self.cnf:
            clause = list(set(clause))  # remove redundant literals
            if len(set(np.abs(clause))) == len(np.abs(clause)):  # remove trivial clauses
                processed_tseitin.append(clause)
        return processed_tseitin

    def tseitin_convert_and_preprocess(self):
        # Converts formula to Tseitin form and preprocesses it
        # CNF is represented as a list of lists. Lower level lists represent the clauses, and elements represent literals
        # Atomic variables represented by abs of literals, negative elements are negated atomics
        self.cnf = self.tseitin()
        self.cnf = self.tseitin_preprocessing()
        return self.cnf

    def process_q_formula(self):
        # Converts formula with atomics over Tq
        # to list of lists of atomics, corresponding to its DNF
        nnf = self.run_to_saturation(self.formula, self.nnf)
        rewritten_nnf = self.rewrite_q_atomics(nnf)
        nnf = self.run_to_saturation(rewritten_nnf, self.nnf)
        dnf_bool_op = self.run_to_saturation(nnf, self.distribute)
        dnf = self.convert_bool_dnf(dnf_bool_op)
        return dnf

    def distribute(self, formula):
        # Recursively applies distribution for a given formula
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
        # Converts atomics over Tq with any order relation to equisat formulas with <=, >0
        if type(formula) == Atomic:
            expr = str(formula)
            if self.LEQ in expr:
                return formula
            elif self.GEQ in expr:
                return Atomic(self.geq_to_leq(expr))
            elif self.NEQ in expr:
                left_formula = self.rewrite_q_atomics(Atomic(expr.replace(self.NEQ, self.LT)))
                right_expr = self.geq_to_leq(expr.replace(self.NEQ, self.GEQ))
                i = right_expr.index(self.LEQ)
                right_expr = right_expr[:i] + self.PLUS + self.UNUSED_VAR + self.LEQ + right_expr[i + 2:]
                return Or(left_formula, And(Atomic(right_expr), Atomic(self.UNUSED_VAR + self.GT + "0")))
            elif self.EQ in expr:
                return And(Atomic(expr.replace(self.EQ, self.LEQ)),
                           Atomic(self.geq_to_leq(expr.replace(self.EQ, self.GEQ))))
            elif self.LT in expr:
                i = expr.index(self.LT)
                expr = expr[:i] + self.PLUS + self.UNUSED_VAR + self.LEQ + expr[i + 1:]
                return And(Atomic(expr), Atomic(self.UNUSED_VAR + self.GT + "0"))
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

    def geq_to_leq(self, expr):
        # Converts >= formula to equivalent <= formula
        assert self.GEQ in expr
        expr = expr.replace(self.PLUS, self.PLACE_HOLDER)
        expr = expr.replace(self.MINUS, self.PLUS)
        expr = expr.replace(self.PLACE_HOLDER, self.MINUS)
        if expr[0] == self.PLUS:
            expr = expr[1:]
        else:
            expr = self.MINUS + expr
        i = expr.index(self.GEQ) + 2
        if expr[i] == self.PLUS:
            expr = expr[:i] + expr[i + 1:]
        else:
            expr = expr[:i] + self.MINUS + expr[i:]
        return expr.replace(self.GEQ, self.LEQ)

    @staticmethod
    def run_to_saturation(formula, func):
        # Runs the given function over the given formula until saturation
        # The given function receives the given formula and returns another one
        to_continue = True
        new = formula
        prev = None
        while to_continue:
            prev = new
            new = func(prev)
            if str(new) == str(prev):
                to_continue = False
        return prev

    def nnf(self, formula):
        # Recursively converts the given formula to NNF
        # For complete transformation needs to be applied until saturation
        if type(formula) == Atomic:
            return formula
        elif type(formula) == And:
            return And(self.nnf(formula.left), self.nnf(formula.right))
        elif type(formula) == Or:
            return Or(self.nnf(formula.left), self.nnf(formula.right))
        elif type(formula) == Imp:
            return self.nnf(Or(Not(formula.left), formula.right))
        elif type(formula) == Equiv:
            return self.nnf(And(Or(Not(formula.left), formula.right), Or(Not(formula.right), formula.left)))
        elif type(formula) == Not:
            if type(formula.param) == Not:
                return self.nnf(self.nnf(formula.param.param))
            elif type(formula.param) == And:
                return self.nnf(Or(Not(formula.param.left), Not(formula.param.right)))
            elif type(formula.param) == Or:
                return self.nnf(And(Not(formula.param.left), Not(formula.param.right)))
            else:
                return Not(self.nnf(formula.param))
        return

    def convert_bool_dnf(self, dnf_bool_op):
        # Convert the given DNF BooleanOperator formula to list of lists of its atomics
        temp = []
        dnf = []
        self.convert_bool_dnf_helper(dnf_bool_op, temp, Or)
        for conj in temp:
            clause = []
            self.convert_bool_dnf_helper(conj, clause, And)
            dnf.append(clause)
        return dnf

    def convert_bool_dnf_helper(self, formula, output_list, input_type):
        # Runs through a formula, made from repeated calls of input_type BooleanOperator
        # i.e. And(x, And(y, And(z)))
        # And creates a list of all its arguments which are not of input_type type
        if type(formula) == input_type:
            self.convert_bool_dnf_helper(formula.left, output_list, input_type)
            self.convert_bool_dnf_helper(formula.right, output_list, input_type)
        elif type(formula) == Atomic:
            output_list.append(formula.val)
        else:
            output_list.append(formula)

    def convert_clause_to_lp(self, clause):
        # Converts atomics strings in a list to LP instance
        # Assuming the atomics adhere the standard form of LP, i.e. contain <=
        A = []
        b = np.zeros(len(clause))
        for i, atomic in enumerate(clause):
            assert self.LEQ in atomic, "Input should adhere the standard form of LP"
            lhs, rhs = atomic.split(self.LEQ)
            b[i] = float(rhs)
            A.append(self.create_LP_row(lhs))
        return np.array(A), b

    def create_LP_row(self, lhs):
        # Parses the left hand side of an equation to a list of coefficients
        # Assuming variables names do not contain numbers
        coefficients = []
        coeff = ''
        for i, char in enumerate(lhs):
            if char.isdigit() or char == self.DOT or (not len(coeff) and char == self.MINUS):
                coeff += char
            else:
                if len(coeff):
                    coefficients.append(float(coeff))
                coeff = ''
        return coefficients
