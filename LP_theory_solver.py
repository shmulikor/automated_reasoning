import numpy as np
from scipy.linalg import lu, solve_triangular

class RevisedSimplexAlgorithm:
    BLAND = "bland"
    DANTZIG = "dantzig"

    def __init__(self, A, b, c, rule, eta_threshold=1, epsilon=0.01):
        assert rule == self.BLAND or rule == self.DANTZIG
        self.rule = rule
        self.epsilon = epsilon

        self.n = len(b)
        self.A_n = A[:, :A.shape[1] - self.n]

        self.Bs = []
        self.etas = []
        self.Bs.append(np.identity(self.n))
        self.eta_threshold = eta_threshold
        self.p = np.identity(self.n)
        self.l, self.u = None, None

        self.x_n = np.arange(A.shape[1] - self.n)
        self.x_b = np.arange(A.shape[1] - self.n, A.shape[1])
        self.x_b_star = b

        self.c_n = c[:A.shape[1] - self.n]
        self.c_b = np.zeros(self.n)

        self.solution_vars = np.zeros(A.shape[1])
        self.objective = 0

    def run(self):
        self.main_algo()
        self.calc_solution()

    def calc_enter_and_leave(self, B_inv, z, use_eta):
        if np.max(z) > self.epsilon:
            enter_idx = np.where(z > self.epsilon)[0][0] if self.rule == self.BLAND else np.argmax(z)
        else:
            return False, 0, 0, 0, 0, None

        column = self.A_n[:, enter_idx]
        d = B_inv @ column
        d[d == 0] = np.finfo(float).eps
        t = np.min([num for num in self.x_b_star / d if num > 0])
        leave_idx = np.where(self.x_b_star / d == t)[0][0]

        if use_eta:
            if len(self.etas) >= self.eta_threshold:
                self.p, self.l, self.u = lu(self.Bs[-1])
                self.etas = []
            eta = np.identity(self.n)
            eta[:, leave_idx] = d
            if np.min(np.diag(eta)) < self.epsilon:
                z[enter_idx] = 0

        return True, enter_idx, leave_idx, t, d, eta if use_eta else None

    def update_parameters(self, enter_idx, leave_idx, t, d):
        enter_var = self.x_n[enter_idx]
        leave_var = self.x_b[leave_idx]

        tmp = self.A_n[:, enter_idx].copy()
        self.A_n[:, enter_idx] = self.Bs[-1][:, leave_idx]
        B = self.Bs[-1].copy()
        B[:, leave_idx] = tmp
        self.Bs.append(B)

        self.x_b_star = self.x_b_star - t * d
        self.x_b_star[leave_idx] = t
        self.x_b[leave_idx] = enter_var
        self.x_n[enter_idx] = leave_var

        tmp = self.c_n[enter_idx].copy()
        self.c_n[enter_idx] = self.c_b[leave_idx]
        self.c_b[leave_idx] = tmp

    def main_algo(self, use_eta=True):
        enter_idx, leave_idx, t, d = 0, 0, 0, 0
        while True:
            B_inv = self.inv_B_by_etas() if use_eta else np.linalg.inv(self.Bs[-1])
            y = self.c_b @ B_inv
            z = self.c_n - y @ self.A_n

            eta = np.zeros(self.n)
            while np.min(np.diag(eta)) < self.epsilon:
                found, enter_idx, leave_idx, t, d, eta = self.calc_enter_and_leave(B_inv, z, use_eta)
                if not found:
                    return

            if use_eta:
                self.etas.append(eta)

            self.update_parameters(enter_idx, leave_idx, t, d)

            # TODO - when |b - Bs[-1] @ self.x_b_star| > epsilon, need to refactor the basis immediately

    def calc_solution(self):
        self.solution_vars[self.x_b] = self.x_b_star
        self.objective = self.c_b @ self.x_b_star

    def inv_B_by_etas(self):
        use_lu = False if self.l is None else True
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
        eta = eta.astype(float)
        special_column_idx = np.where(eta - np.identity(self.n))[1][0]
        mask = np.ones(self.n, bool)
        mask[special_column_idx] = False
        eta[special_column_idx, special_column_idx] = 1 / eta[special_column_idx, special_column_idx]
        eta[mask, special_column_idx] = -eta[mask, special_column_idx]
        eta[mask, special_column_idx] *= eta[special_column_idx, special_column_idx]
        return eta


if __name__ == '__main__':
    # todo - auxiliary?

    # ex3 q2 example
    # A = np.array([[1, 1, 2, 1, 0, 0], [2, 0, 3, 0, 1, 0], [2, 1, 3, 0, 0, 1]])
    # b = np.array([4, 5, 7])
    # c = np.array([3, 2, 4, 0, 0, 0])

    # lecture example
    A = np.array([[3, 2, 1, 2, 1, 0, 0], [1, 1, 1, 1, 0, 1, 0], [4, 3, 3, 4, 0, 0, 1]])
    b = np.array([225, 117, 420])
    c = np.array([19, 13, 12, 17, 0, 0, 0])

    # print scipy-solver results
    from scipy.optimize import linprog
    res = linprog(c=-c, A_ub=A, b_ub=b)
    print(res)

    rsa = RevisedSimplexAlgorithm(A, b, c, RevisedSimplexAlgorithm.DANTZIG)
    rsa.run()
    print(rsa.solution_vars)
    print(rsa.objective)
