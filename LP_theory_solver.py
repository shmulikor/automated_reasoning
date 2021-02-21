import numpy as np
from scipy.linalg import lu, solve_triangular
from scipy.optimize import linprog


class RevisedSimplexAlgorithm:
    BLAND = "bland"
    DANTZIG = "dantzig"
    UNBOUNDED = 2
    INFEASIBLE = 1
    BOUNDED = 0
    NO_INDEX = -1

    def __init__(self, A, b, c, rule=DANTZIG, eta_threshold=10, epsilon=0.01, needs_aux=True):
        assert rule == self.BLAND or rule == self.DANTZIG
        self.rule = rule
        self.epsilon = epsilon
        self.needs_aux = needs_aux

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

    def run(self):
        if any(self.b < 0) and self.needs_aux:
            c = np.array([-1] + [0] * len(self.c_n))
            A = np.hstack((np.array([-1] * len(b))[:, None], self.A_n))
            auxiliary = RevisedSimplexAlgorithm(A, b ,c, self.rule, self.eta_threshold, self.epsilon, False)
            min_idx = np.argmin(self.b)
            auxiliary.perform_pivot(0,min_idx,abs(min(self.b)), np.array([-1]*self.n))
            auxiliary.run()
            print("Aux sol: ", auxiliary.solution_type)
            print("Aux assignment: ", auxiliary.cur_assignment)
            self.calc_solution()
            if auxiliary.solution_type == self.BOUNDED and auxiliary.cur_assignment[0]:
                self.solution_type = self.INFEASIBLE
                return
            # TODO postprocessing

        while True:
            B_inv = self.inv_B_by_etas()
            entering_column_index = self.pick_entering_index(B_inv)
            if entering_column_index == self.NO_INDEX:
                self.solution_type = self.BOUNDED
                self.calc_solution()
                return
            leaving_column_index, t, d = self.pick_leaving_index(B_inv, entering_column_index)
            if leaving_column_index == self.NO_INDEX:
                self.solution_type = self.UNBOUNDED
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
            if not d[i]:
                ts.append(np.inf)
            else:
                num = self.x_b_star[i] / d[i]
                if num > 0 or not self.needs_aux:
                    ts.append(num)
        if not len(ts) or min(ts) == np.inf: # unbounded
            return self.NO_INDEX, 0, d
        t = np.min(ts)
        leave_idx = np.argmin(ts)
        return leave_idx, t, d

if __name__ == '__main__':

    # ex3 q2 example
    # A = np.array([[1, 1, 2],
    #               [2, 0, 3],
    #               [2, 1, 3]])
    # b = np.array([4, 5, 7])
    # c = np.array([3, 2, 4])

    # lecture example
    # A = np.array([[3, 2, 1, 2],
    #               [1, 1, 1, 1],
    #               [4, 3, 3, 4]])
    # b = np.array([225, 117, 420])
    # c = np.array([19, 13, 12, 17])

    # A = np.array([[2, 2, -1],
    #               [3, -2, 1],
    #               [1, -3, 1]])
    # b = np.array([10, 10, 10])
    # c = np.array([1, 3, -1])

    A = np.array([[1, -1],
                  [-1, -1],
                  [2, 1]])
    b = np.array([-1, -3, 4])
    c = np.array([3, 1])

    # A = np.array([[-1, +1],
    #               [-2, -2],
    #               [-1, 4]])
    # b = np.array([-1, -6, 2])
    # c = np.array([1, 3])


    # print scipy-solver results
    res = linprog(c=-c, A_ub=A, b_ub=b)
    print(res)

    rsa = RevisedSimplexAlgorithm(A, b, c, RevisedSimplexAlgorithm.BLAND)
    rsa.run()
    print(rsa.cur_assignment)
    print(rsa.cur_objective)
    print(rsa.solution_type)
