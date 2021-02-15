import numpy as np
from scipy.linalg import lu, solve_triangular
from scipy.optimize import linprog


class RevisedSimplexAlgorithm:
    BLAND = "bland"
    DANTZIG = "dantzig"

    def __init__(self, A, b, c, rule, eta_threshold=1, epsilon=0.01):
        assert rule == self.BLAND or rule == self.DANTZIG
        self.rule = rule
        self.epsilon = epsilon

        self.n = len(b)
        self.A_n = A[:, :A.shape[1] - self.n]

        # init Bs and etas
        self.Bs = []
        self.etas = []
        self.Bs.append(np.identity(self.n))
        self.eta_threshold = eta_threshold
        self.p = np.identity(self.n)
        self.l, self.u = None, None

        # init x
        self.x_n = np.arange(A.shape[1] - self.n)
        self.x_b = np.arange(A.shape[1] - self.n, A.shape[1])
        self.x_b_star = b

        # init c
        self.c_n = c[:A.shape[1] - self.n]
        self.c_b = np.zeros(self.n)

        self.solution_vars = np.zeros(A.shape[1])
        self.objective = 0

    def run(self):
        self.main_algo()    # run the algo
        self.calc_solution()    # calc the solution

    def calc_enter_and_leave(self, B_inv, z, use_eta):
        """

        :param B_inv: inverse of B matrix
        :param z: z vector
        :param use_eta: if True - need to calc the current Eta matrix
        :return:
            True if there is an enter var, otherwise - False,
            enter_idx - index of the enter var,
            leave_idx - index of the leave var,
            t - t of this iteration,
            d - d of this iteration,
            eta - current Eta matrix if use_eta, otherwise - identity matrix
        """
        if np.max(z) > self.epsilon:
            enter_idx = np.where(z > self.epsilon)[0][0] if self.rule == self.BLAND else np.argmax(z)
        else:
            return False, 0, 0, 0, 0, None

        column = self.A_n[:, enter_idx]
        d = B_inv @ column
        d[d == 0] = np.finfo(float).eps
        ts = [num for num in self.x_b_star / d if num > 0]
        t = np.min(ts)
        leave_idx = np.argmin(ts)

        eta = np.identity(self.n)
        if use_eta:
            if len(self.etas) >= self.eta_threshold:    # check if lu factorization is needed
                self.p, self.l, self.u = lu(self.Bs[-1])
                self.etas = []
            eta[:, leave_idx] = d
            if np.min(np.abs(np.diag(eta))) < self.epsilon: # for numerical stability
                z[enter_idx] = 0

        return True, enter_idx, leave_idx, t, d, eta

    def update_parameters(self, enter_idx, leave_idx, t, d):
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
        self.A_n[:, enter_idx] = self.Bs[-1][:, leave_idx]
        B = self.Bs[-1].copy()
        B[:, leave_idx] = tmp
        self.Bs.append(B)

        # update x
        self.x_b_star = self.x_b_star - t * d
        self.x_b_star[leave_idx] = t
        self.x_b[leave_idx] = enter_var
        self.x_n[enter_idx] = leave_var

        # swap c vals
        tmp = self.c_n[enter_idx].copy()
        self.c_n[enter_idx] = self.c_b[leave_idx]
        self.c_b[leave_idx] = tmp

    def main_algo(self, use_eta=False):
        """
        main flow of the algorithm
        :param use_eta: if True - use Eta factorization for inverting B matrix
        :return:
        """
        enter_idx, leave_idx, t, d = 0, 0, 0, 0
        while True:
            """the loop stops if calc_enter_and_leave returns False, i.e. if there is no enter and leave vars for a 
            specific iteration"""
            B_inv = self.inv_B_by_etas() if use_eta else np.linalg.inv(self.Bs[-1])
            y = self.c_b @ B_inv
            z = self.c_n - y @ self.A_n

            eta = np.zeros(self.n)
            while np.min(np.abs(np.diag(eta))) < self.epsilon:  # this loop is needed for numerical stability
                found, enter_idx, leave_idx, t, d, eta = self.calc_enter_and_leave(B_inv, z, use_eta)
                if not found:
                    return

            if use_eta:
                self.etas.append(eta)

            # use the enter and leave vars for update A, B, x, c etc.
            self.update_parameters(enter_idx, leave_idx, t, d)

            # TODO - when |b - Bs[-1] @ self.x_b_star| > epsilon, need to refactor the basis immediately

    def calc_solution(self):
        """
        calc the solution of the optimization problem
        :return:
        """
        self.solution_vars[self.x_b] = self.x_b_star # update the values of all the vars
        self.objective = self.c_b @ self.x_b_star # calc the solution

    def inv_B_by_etas(self):
        """
        use Eta matrices for inverting B
        :return: the inverse of B
        """
        use_lu = False if self.l is None else True # if there is a saved l matrix - need to use lu factorization
        B_inv = np.identity(self.n)
        for eta in self.etas[::-1]:
            eta_inv = self.inv_eta(eta)
            B_inv = B_inv @ eta_inv
        if use_lu:
            B_inv = B_inv @ solve_triangular(self.u, np.identity(self.n)) @ \
                    solve_triangular(self.l, np.identity(self.n), lower=True) @ np.linalg.inv(self.p)
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


if __name__ == '__main__':
    # todo - auxiliary?

    # ex3 q2 example
    # A = np.array([[1, 1, 2, 1, 0, 0],
    #               [2, 0, 3, 0, 1, 0],
    #               [2, 1, 3, 0, 0, 1]])
    # b = np.array([4, 5, 7])
    # c = np.array([3, 2, 4, 0, 0, 0])

    # lecture example
    A = np.array([[3, 2, 1, 2, 1, 0, 0],
                  [1, 1, 1, 1, 0, 1, 0],
                  [4, 3, 3, 4, 0, 0, 1]])
    b = np.array([225, 117, 420])
    c = np.array([19, 13, 12, 17, 0, 0, 0])

    # print scipy-solver results
    res = linprog(c=-c, A_ub=A, b_ub=b)
    print(res)

    # TODO - without the assumption that the matrix includes slack variables
    rsa = RevisedSimplexAlgorithm(A, b, c, RevisedSimplexAlgorithm.BLAND)
    rsa.run()
    print(rsa.solution_vars)
    print(rsa.objective)
