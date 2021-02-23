import numpy as np
from scipy.linalg import lu

UNBOUNDED = "Unbounded"
INFEASIBLE = "Infeasible"
BOUNDED = "Bounded"
UNDECIDED = "Undecided"
BLAND = "bland"
DANTZIG = "dantzig"
DEBUG_LP = True


# Assumptions - A,b,c correspond to a correct standard form LP instance

class LPSolver:
    # A class representing an LP solver engine
    # Handles LP problems of the standard form
    # Wraps the RevisedSimplexAlgorithm class, in order to manage two instances,
    # in case an auxiliary problem is needed

    def __init__(self, A, b, c, rule=BLAND, eta_threshold=10, epsilon=0.001):
        self.A = A
        self.b = b
        self.c = c
        self.rule = rule
        self.eta_threshold = eta_threshold
        self.epsilon = epsilon
        self.objective = 0
        self.assignment = np.zeros(len(c))
        self.solution_type = UNDECIDED

        # If needed, construct an auxiliary problem for the given one
        if any(b < 0):
            aux_A = np.hstack((np.array([-1] * len(b))[:, None], A))
            aux_c = np.array([-1] + [0] * len(c))
            new_c = np.concatenate(([0], c))
            self.auxiliary = RevisedSimplexAlgorithm(aux_A.copy(), b.copy(), aux_c, rule, eta_threshold, epsilon, True)
            self.mainLP = RevisedSimplexAlgorithm(aux_A.copy(), b.copy(), new_c, rule, eta_threshold, epsilon, False)
        else:
            self.auxiliary = None
            self.mainLP = RevisedSimplexAlgorithm(A, b, c, rule, eta_threshold, epsilon, False)

    def solve(self):
        # Runs the auxiliary problem first (in case it is needed), otherwise runs the main LP
        # If auxiliary problem solution is not bounded with x0=0, original problem is infeasible
        # If not so, use the pivot history of the auxiliary for the main LP as well
        if self.auxiliary:
            b = self.auxiliary.b
            n = self.auxiliary.m
            min_idx = np.argmin(b)
            self.auxiliary.perform_pivot(0, min_idx, abs(min(b)), np.array([-1] * n))
            self.auxiliary.aux_pivot_history.append((0, min_idx, abs(min(b)), np.array([-1] * n)))
            self.auxiliary.solve()
            if self.auxiliary.solution_type == BOUNDED and self.auxiliary.cur_assignment[0]:
                self.solution_type = INFEASIBLE
                return False, {}

            history = self.auxiliary.aux_pivot_history
            for enter_idx, leaving_idx, t, d in history:
                self.mainLP.perform_pivot(enter_idx, leaving_idx, t, d)
            self.mainLP.eraseX0()

        self.mainLP.solve()
        self.objective = self.mainLP.cur_objective
        self.assignment = self.mainLP.cur_assignment
        self.solution_type = self.mainLP.solution_type
        return True, {i: val for i, val in enumerate(self.assignment)}


class RevisedSimplexAlgorithm:
    # A class responsible for implementation of the revised simplex algorithm
    NO_INDEX = -1
    NUM_STAB_PERIOD = 10

    def __init__(self, A, b, c, rule=BLAND, eta_threshold=100, epsilon=0.01, is_aux=True):
        assert rule == BLAND or rule == DANTZIG
        self.rule = rule
        self.epsilon = epsilon
        self.is_aux = is_aux
        self.iterations_counter = 0
        self.aux_pivot_history = []

        self.b = b
        self.m = len(b)
        self.A_n = A

        # init B and etas
        self.B = np.identity(self.m)
        self.etas = []
        self.eta_threshold = eta_threshold

        # For lu factorization
        self.p = np.identity(self.m)

        # init x
        self.x_n = np.arange(A.shape[1])
        self.x_b = np.arange(A.shape[1], A.shape[1] + self.m)
        self.x_b_star = b

        # init c
        self.c_n = c
        self.c_b = np.zeros(self.m)

        self.cur_assignment = np.zeros(A.shape[1] + self.m)
        self.cur_objective = 0
        self.solution_type = UNDECIDED

    def solve(self):
        # Runs the algorithm
        while True:
            self.iterations_counter += 1
            B_inv = self.inv_B_by_etas()

            # For numerical stability reasons if B_inv is not accurate - refactor the basis
            if self.iterations_counter == self.NUM_STAB_PERIOD:
                self.iterations_counter = 0
                if not np.allclose(B_inv @ self.b, self.x_b_star, self.epsilon, self.epsilon):
                    self.lu_factorization()

            # Pick the entering variable
            entering_column_index = self.pick_entering_index(B_inv)
            # If no variable found, conclude optimal solution
            if entering_column_index == self.NO_INDEX:
                self.solution_type = BOUNDED
                self.calc_solution()
                return

            # Pick the leaving variable
            leaving_column_index, t, d = self.pick_leaving_index(B_inv, entering_column_index)

            # Record history for auxiliary problems, so the main problem can mimic it
            if self.is_aux:
                self.aux_pivot_history.append((entering_column_index, leaving_column_index, t, d))

            # If no t found for x*_b - td >= 0
            if leaving_column_index == self.NO_INDEX:
                self.solution_type = UNBOUNDED
                self.calc_solution()
                return

            # Performs the pivot according to the indices, non-basic column d, and maximizer t
            self.perform_pivot(entering_column_index, leaving_column_index, t, d)

    def perform_pivot(self, enter_idx, leave_idx, t, d):
        # Update all parameters of the algorithm according to the pivot specified

        # Retrieve variables indices
        enter_var = self.x_n[enter_idx]
        leave_var = self.x_b[leave_idx]

        # Swap A_n, B columns
        tmp = self.A_n[:, enter_idx].copy()
        self.A_n[:, enter_idx] = self.B[:, leave_idx]
        self.B[:, leave_idx] = tmp

        # Update x
        self.x_b_star = self.x_b_star - t * d
        self.x_b_star[leave_idx] = t
        self.x_b[leave_idx] = enter_var
        self.x_n[enter_idx] = leave_var

        # Swap c vals
        tmp = self.c_n[enter_idx].copy()
        self.c_n[enter_idx] = self.c_b[leave_idx]
        self.c_b[leave_idx] = tmp

        # Add eta
        eta = np.identity(self.m)
        eta[:, leave_idx] = d
        if not np.all(eta == np.identity(self.m)):
            self.etas.append(eta)

        # Refactor if needed
        if len(self.etas) >= self.eta_threshold:
            self.lu_factorization()

    def calc_solution(self):
        # Calculate the solution of the optimization problem, according to current assignment

        self.cur_assignment[self.x_b] = self.x_b_star  # update the current assignment of all vars
        self.cur_objective = self.c_b @ self.x_b_star  # calc the current objective value

    def inv_B_by_etas(self):
        # Use Eta matrices for inverting B

        B_inv = np.identity(self.m)
        for eta in self.etas[::-1]:
            eta_inv = self.inv_eta(eta)
            B_inv = B_inv @ eta_inv
        B_inv = B_inv @ self.p.T  # Permutation matrix is orthogonal
        return B_inv

    def inv_eta(self, eta):
        # Invert a given Eta matrix, based on the algorithm we saw in class
        if np.all(eta == np.identity(self.m)):
            return eta

        inv_eta = eta.astype(float)

        # Find the column where eta is different from the identity matrix
        special_column_idx = np.where(eta - np.identity(self.m))[1][0]

        # Mark all rows, except of the special one
        mask = np.ones(self.m, bool)
        mask[special_column_idx] = False

        # Invert, negate and divide the special column
        inv_eta[special_column_idx, special_column_idx] = 1 / inv_eta[special_column_idx, special_column_idx]
        inv_eta[mask, special_column_idx] = -inv_eta[mask, special_column_idx]
        inv_eta[mask, special_column_idx] *= inv_eta[special_column_idx, special_column_idx]

        return inv_eta

    def pick_entering_index(self, B_inv):
        # Picks the index of the variable entering the basis
        # according to the rule specified by self.rule

        y = self.c_b @ B_inv
        z_coeff = self.c_n - y @ self.A_n

        # For numerical stability - don't pick a rounded positive
        if np.max(z_coeff) > self.epsilon:
            return np.where(z_coeff > self.epsilon)[0][0] if self.rule == BLAND else np.argmax(z_coeff)
        return self.NO_INDEX

    def pick_leaving_index(self, B_inv, enter_idx):
        # Picks the index of the variable leaving the basis

        column = self.A_n[:, enter_idx]
        d = B_inv @ column

        # Pick maximal t such that x*_b - td >= 0, if exists
        ts = []
        for i in range(len(self.x_b_star)):
            if d[i] > 0:
                ts.append(self.x_b_star[i] / d[i])
            else:
                ts.append(np.inf)

        # No such t exists, can conclude problem is unbounded
        if np.min(ts) == np.inf:
            return self.NO_INDEX, 0, d

        # Finds the maximal t (minimal -t)
        t = min(ts)
        leave_idx = np.argmin(ts)
        return leave_idx, t, d

    def eraseX0(self):
        # After creating the dictionary of the auxiliary problem
        # Remove auxiliary variable x0 from all parameters
        assert 0 in self.x_n and not self.is_aux
        zero_idx = np.where(self.x_n == 0)[0][0]
        self.x_n = np.delete(self.x_n, zero_idx)
        self.c_n = np.delete(self.c_n, zero_idx)
        self.A_n = np.delete(self.A_n, zero_idx, axis=1)
        self.x_n -= 1  # Update other variables indices
        self.x_b -= 1
        self.cur_assignment = self.cur_assignment[1:]

    def lu_factorization(self):
        # Factorises the basis

        self.etas = []

        if DEBUG_LP:
            prev_B = self.B.copy()

        # l, u are triangular matrices
        # keep p for further inverse calculations (p is orthogonal, thus inv = transpose)
        self.p, l, u = lu(self.B, permute_l=False)

        # Decompose l, u to eta matrices
        for i in range(l.shape[0]):
            temp_eta = np.identity(self.m)
            temp_eta[:, i] = l[:, i]
            self.etas.append(temp_eta)

        for i in range(u.shape[0])[::-1]:
            temp_eta = np.identity(self.m)
            temp_eta[:, i] = u[:, i]
            self.etas.append(temp_eta)

        if DEBUG_LP:
            check = np.identity(self.m)
            for eta in self.etas:
                check = check @ eta

            assert np.allclose(check, l @ u)
            assert np.allclose(self.p @ check, prev_B)
