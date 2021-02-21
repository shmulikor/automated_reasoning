import numpy as np
from scipy.linalg import lu, solve_triangular
from scipy.optimize import linprog

# TODO delete d,t from pivot function
# TODO lu factorization
# TODO num stab
# TODO merge to SMT


UNBOUNDED = 2
INFEASIBLE = 1
BOUNDED = 0


class LP_SOLVER:


    def __init__(self, A, b, c, rule, eta_threshold=10, epsilon=0.01):
        self.A = A
        self.b = b
        self.c = c
        self.rule = rule
        self.eta_threshold = eta_threshold
        self.epsilon = epsilon

        if any(b < 0):
            aux_A = np.hstack((np.array([-1] * len(b))[:, None], A))
            aux_c = np.array([-1] + [0] * len(c))
            new_c = np.concatenate(([0], c))
            self.auxiliary = RevisedSimplexAlgorithm(aux_A.copy(), b.copy(), aux_c, rule, eta_threshold, epsilon, True)
            self.mainLP = RevisedSimplexAlgorithm(aux_A.copy(), b.copy(), new_c, rule, eta_threshold, epsilon, False)
        else:
            self.auxiliary = None
            self.mainLP = RevisedSimplexAlgorithm(A, b, c, rule, eta_threshold, epsilon, False)

    def run(self):
        if self.auxiliary:
            b = self.auxiliary.b
            n = self.auxiliary.n
            min_idx = np.argmin(b)
            self.auxiliary.perform_pivot(0, min_idx, abs(min(b)), np.array([-1] * n))
            self.auxiliary.aux_pivot_history.append((0, min_idx, abs(min(b)), np.array([-1] * n)))
            self.auxiliary.run()
            if self.auxiliary.solution_type == BOUNDED and self.auxiliary.cur_assignment[0]:
                self.solution_type = INFEASIBLE
                return

            history = self.auxiliary.aux_pivot_history
            for enter_idx, leaving_idx, t, d in history:
                self.mainLP.perform_pivot(enter_idx, leaving_idx, t, d)
            self.mainLP.eraseX0()

        self.mainLP.run()

class RevisedSimplexAlgorithm():
    BLAND = "bland"
    DANTZIG = "dantzig"
    NO_INDEX = -1

    def __init__(self, A, b, c, rule=DANTZIG, eta_threshold=100, epsilon=0.01, is_aux=True):
        assert rule == self.BLAND or rule == self.DANTZIG
        self.rule = rule
        self.epsilon = epsilon
        self.is_aux = is_aux

        self.b = b
        self.n = len(b)
        self.A_n = A

        # init B and etas
        self.B = np.identity(self.n)
        self.etas = []
        self.eta_threshold = eta_threshold

        # For lu factorization
        self.p = np.identity(self.n)
        self.l, self.u = None, None

        # init x
        self.x_n = np.arange(A.shape[1])
        self.x_b = np.arange(A.shape[1], A.shape[1] + self.n)
        self.x_b_star = b

        # init c
        self.c_n = c
        self.c_b = np.zeros(self.n)

        self.cur_assignment = np.zeros(A.shape[1] + self.n)
        self.cur_objective = 0
        self.solution_type = -1

        self.aux_pivot_history = []

    def run(self):
       while True:
            B_inv = self.inv_B_by_etas()
            # B_inv[np.abs(B_inv) < self.epsilon] = 0 # TODO delete
            entering_column_index = self.pick_entering_index(B_inv)
            if entering_column_index == self.NO_INDEX:
                self.solution_type = BOUNDED
                self.calc_solution()
                return
            leaving_column_index, t, d = self.pick_leaving_index(B_inv, entering_column_index)
            if self.is_aux:
                self.aux_pivot_history.append((entering_column_index, leaving_column_index, t, d))
            if leaving_column_index == self.NO_INDEX:
                self.solution_type = UNBOUNDED
                return
            self.perform_pivot(entering_column_index, leaving_column_index, t, d)

    def perform_pivot(self, enter_idx, leave_idx, t, d):
        """
        update all the parameters of the algorithm
        :param enter_idx: index of the enter var
        :param leave_idx: index of the leave var
        :param t: t of this iteration
        :param d: d of this iteration
        :return:
        """

        enter_var = self.x_n[enter_idx]
        leave_var = self.x_b[leave_idx]

        # swap A_n, B columns
        tmp = self.A_n[:, enter_idx].copy()
        self.A_n[:, enter_idx] = self.B[:, leave_idx]
        self.B[:, leave_idx] = tmp

        # update x
        self.x_b_star = self.x_b_star - t * d
        self.x_b_star[leave_idx] = t
        self.x_b[leave_idx] = enter_var
        self.x_n[enter_idx] = leave_var

        # swap c vals
        tmp = self.c_n[enter_idx].copy()
        self.c_n[enter_idx] = self.c_b[leave_idx]
        self.c_b[leave_idx] = tmp

        # add eta
        eta = np.identity(self.n)
        if len(self.etas) >= self.eta_threshold:  # check if lu factorization is needed
            self.p, self.l, self.u = lu(self.B)
            self.etas = []
        eta[:, leave_idx] = d
        if not np.all(eta == np.eye(self.n)):
            self.etas.append(eta)


    def calc_solution(self):
        """
        calc the solution of the optimization problem
        :return:
        """
        self.cur_assignment[self.x_b] = self.x_b_star  # update the current assignment of all vars
        self.cur_objective = self.c_b @ self.x_b_star  # calc the current objective value

    def inv_B_by_etas(self):
        """
        use Eta matrices for inverting B
        :return: the inverse of B
        """
        use_lu = False if self.l is None else True  # if there is a saved l matrix - need to use lu factorization
        B_inv = np.identity(self.n)
        for eta in self.etas[::-1]:
            eta_inv = self.inv_eta(eta)
            B_inv = B_inv @ eta_inv
        if use_lu:
            B_inv = B_inv @ solve_triangular(self.u, np.identity(self.n)) @ \
                    solve_triangular(self.l, np.identity(self.n), lower=True) @ self.p.T
            """
            In contrary to what is written in the powerpoint file, triangular matrices are not eta matrices. It is 
            possible to fix the inv_eta function for this case, but I used the solve_triangular instead, which I believe
            is more efficient than the simple inv function. In addition, I used inv for the p matrix, which is only a 
            permutation matrix. Need to make some decisions on this part. 
            However, algorithmically, the code returns the expected results.  
            """
        return B_inv

    def inv_eta(self, eta):
        """
        inv the given Eta matrix, based on the algorithm we saw in class
        :param eta:
        :return: the inverse of eta
        """
        eta = eta.astype(float)

        # find the column where eta is different from the identity matrix
        special_column_idx = np.where(eta - np.identity(self.n))[1][0]

        # mark all rows, except of the special one
        mask = np.ones(self.n, bool)
        mask[special_column_idx] = False

        # invert, negate and divide the special column
        eta[special_column_idx, special_column_idx] = 1 / eta[special_column_idx, special_column_idx]
        eta[mask, special_column_idx] = -eta[mask, special_column_idx]
        eta[mask, special_column_idx] *= eta[special_column_idx, special_column_idx]
        return eta

    def pick_entering_index(self, B_inv):
        y = self.c_b @ B_inv
        z_coeff = self.c_n - y @ self.A_n

        if np.max(z_coeff) > self.epsilon:
            return np.where(z_coeff > self.epsilon)[0][0] if self.rule == self.BLAND else np.argmax(z_coeff)
        return self.NO_INDEX

    def pick_leaving_index(self, B_inv, enter_idx):
        column = self.A_n[:, enter_idx]
        d = B_inv @ column
        ts = []
        for i in range(len(self.x_b_star)):
            if d[i] > 0:
                ts.append(self.x_b_star[i] / d[i])
            else:
                ts.append(np.inf)
        if np.min(ts) == np.inf:
            return self.NO_INDEX, 0, d
        t = min(ts)
        leave_idx = np.argmin(ts)
        return leave_idx, t, d

    def basis_dict(self):
        return - np.linalg.inv(self.B) @ self.A_n

    def eraseX0(self):
        assert 0 in self.x_n
        zero_idx = np.where(self.x_n==0)[0][0]
        self.x_n = np.delete(self.x_n, zero_idx)
        self.c_n = np.delete(self.c_n, zero_idx)
        self.A_n = np.delete(self.A_n, zero_idx, axis=1)
        self.x_n -= 1
        self.x_b -= 1
        self.cur_assignment = self.cur_assignment[1:]
        # A_n, x_n, c_n, cur_assignment

        # TODO update b with lu factorization, and empty etas


if __name__ == '__main__':

    # ex3 q2 example
    # A = np.array([[1, 1, 2],
    #               [2, 0, 3],
    #               [2, 1, 3]])
    # b = np.array([4, 5, 7])
    # c = np.array([3, 2, 4])
    #
    # lecture example
    # A = np.array([[3, 2, 1, 2],
    #               [1, 1, 1, 1],
    #               [4, 3, 3, 4]])
    # b = np.array([225, 117, 420])
    # c = np.array([19, 13, 12, 17])

    # unbounded
    # A = np.array([[2, 2, -1],
    #               [3, -2, 1],
    #               [1, -3, 1]])
    # b = np.array([10, 10, 10])
    # c = np.array([1, 3, -1])

    # A = np.array([[1, -1],
    #               [-1, -1],
    #               [2, 1]])
    # b = np.array([-1, -3, 2])
    # c = np.array([3, 1])

    A = np.array([[-1, 1],
                  [-2, -2],
                  [-1, 4]])
    b = np.array([-1, -6, 2])
    c = np.array([1, 3])


    # print scipy-solver results
    res = linprog(c=-c, A_ub=A, b_ub=b)
    print(res)

    rsa = LP_SOLVER(A, b, c, RevisedSimplexAlgorithm.BLAND)
    rsa.run()
    print("Final assignment:", rsa.mainLP.cur_assignment)
    print("Objective value:", rsa.mainLP.cur_objective)
    print("Sol type: ",rsa.mainLP.solution_type)
