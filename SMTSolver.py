from SATSolver import *
import re

LEFT = 0
RIGHT = 1
EQUAL = 2

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
    def __init__(self, var : tuple):
        self.left = var[LEFT]
        self.right = var[RIGHT]
        self.equal = var[EQUAL]

    def __repr__(self):
        return (f"{self.left}{'=' if self.equal else '!='}{self.right}")


    # def create_boolean_abstraction(self):
    #     for conjunct in self.cnf:
    #         boolean_conjunct = []
    #         for lit in conjunct:
    #             if (lit.left, lit.right) not in self.boolean_abstraction.keys():
    #                 self.boolean_abstraction[(lit.left, lit.right)] = next(abstractor)
    #             boolean_var = self.boolean_abstraction[(lit.left, lit.right)]
    #             boolean_conjunct.append(boolean_var if lit.isEqual else -boolean_var)
    #         self.boolean_formula.append(boolean_conjunct)
    #     self.inv_boolean_abstraction = {v: k for k, v in self.boolean_abstraction.items()}




class SMTSolver:
    def __init__(self, raw_formula : BooleanOperator):
        self.boolean_abstraction = {}
        self.inv_boolean_abstraction = {}
        self.boolean_formula = []
        self.convert_raw_to_cnf(raw_formula)
        self.sat_solver = SATSolver(self.boolean_formula)

    def convert_raw_to_cnf(self, raw_formula):
        # Assumption - all atomic are of equality form
        processor = FormulaProcessor(raw_formula)
        processor.convert_and_preprocess()
        self.boolean_formula = processor.cnf
        self.boolean_abstraction = processor.atomic_abstractions
        self.inv_boolean_abstraction = {v: k for k, v in self.boolean_abstraction.items()}


    def decide_uf_sat(self, partial_assignment):
        cong_formula = self.compute_partial_assignment_formula(partial_assignment)
        subterms_set = self.compute_formula_subterms(cong_formula)
        cong_DAG = self.create_cong_DAG(subterms_set)
        return self.process_formula_in_DAG(cong_DAG, cong_formula)

    def compute_partial_assignment_formula(self, partial_assignment):
        literals_conjunction = []
        for var in partial_assignment.keys():
            if var in self.boolean_abstraction:
                literals_conjunction.append(self.parse_atomic(self.boolean_abstraction[var], partial_assignment[var]))
        return literals_conjunction

    def parse_atomic(self, var, equality):
        parts = var.split('=')
        left = parts[0]
        right = parts[1] if len(parts) > 0 else ''
        return UFVar((left,right,equality))


    def compute_formula_subterms(self, cong_formula):
        subterms_set = set()
        for var in cong_formula:
            subterms_set.union(self.parse_term_to_subterms(subterms_set, var.left))
            subterms_set.union(self.parse_term_to_subterms(subterms_set, var.right))
        return subterms_set

    def parse_term_to_subterms(self, subterms_set, term):
        if not len(term):
            return subterms_set

        elif ',' not in term and '(' not in term and ')' not in term:
            subterms_set.add(term)
            return subterms_set

        elif term[-1] == ')' and (not term.count(',') or term.index(',') > term.index('(')):
            subterms_set.add(term)
            subterm = term[term.index('(') + 1 : -1]
            subterms_set = self.parse_term_to_subterms(subterms_set, subterm)
        else:
            comma_idx = [i for i, ltr in enumerate(term) if ltr == ','] + [len(term)]
            last_idx = 0
            for idx in comma_idx:
                take_index = last_idx if not last_idx else last_idx+1
                if term[take_index:idx].count('(') == term[take_index:idx].count(')'):
                    subterms_set = self.parse_term_to_subterms(subterms_set, term[take_index:idx])
                    last_idx = idx
        return subterms_set

    def create_cong_DAG(self, subterms):
        cong_DAG = nx.MultiDiGraph()
        cong_DAG.add_nodes_from(subterms)
        for subterm1 in subterms:
            for subterm2 in subterms:
                if subterm1 in subterm2 and subterm1 != subterm2:
                    all_idx = [i for i in range(len(subterm2)) if subterm2[i:min(len(subterm2), i+len(subterm1))] ==
                           subterm1]
                    for i in all_idx:
                        if subterm2[:i].count('(') - subterm2[:i].count(')') <= 1:
                            cong_DAG.add_edge(subterm2, subterm1)
        assert(nx.is_directed_acyclic_graph(cong_DAG))
        print(subterms)
        nx.draw_networkx(cong_DAG, pos=nx.planar_layout(cong_DAG))
        plt.show()
        return cong_DAG



if __name__ == '__main__':
    raw_formula = Or(And(Atomic("f(a)=f(b)"), Atomic("b=c")), Atomic('g(a,f(a,k(b),h(c)))=d'))
    #raw_formula = Atomic("f(f(a,b),a)=g(c)")

    smt = SMTSolver(raw_formula)
    print(smt.boolean_abstraction)
    smt.decide_uf_sat({1: True, 2:False, 3:True, 4:False})
