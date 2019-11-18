# -*- coding: utf-8 -*-
"""
A simple implementation of Levenberg Marquardt Algorithm
"""

import numpy as np
from matplotlib import pyplot as plt


class ProblemSolver(object):
    def __init__(self, x, y):
        self.theta = 0
        self.last_theta = 0
        self.delta_theta = 0
        self.jacobian_ = 0
        self.object_ = 0
        self.lamda = 0
        self.residue_ = 0
        self.last_residue_ = 0
        self.chi = 0
        self.last_chi = 0
        self.ni_ = 2
        self.rho = 0
        self.x = x
        self.y = y
        self.H = 0
        self.b = 0

    def set_estimate(self, theta_):
        self.theta = theta_

    def build_normal_equation(self):
        self.compute_jacobian()
        self.H = self.jacobian_.T.dot(self.jacobian_)
        self.b = -self.jacobian_.T.dot(self.residue_)

    def solve_normal_equation(self):
        H = self.H + self.lamda * np.identity(self.H.shape[0])
        self.delta_theta = np.linalg.inv(H).dot(self.b)
        # TODO: Sparse solver stragety

    def initial_solver(self):
        tau = 1e-5
        self.lamda = np.max(np.diag(self.H)) * tau

    def roll_back(self):
        pass


    def in_trusted_region(self):
        scale = self.delta_theta.T @ (self.lamda * self.delta_theta + self.b)
        scale += 1e-3
        self.update_parameter()
        self.compute_residue()
        self.rho = (self.last_chi - self.chi)/scale
        return self.rho[0][0] > 0

    def update_lamda_and_trusted_region(self):
        if self.rho > 0:
            alpha = 1 - np.power(2 * self.rho - 1, 3)
            alpha = min(alpha, 2.0 / 3.0)
            scaleFactor = max(1.0 / 3.0, alpha)
            self.lamda *= scaleFactor
            self.ni_ = 2
        else:
            self.lamda *= self.ni_
            self.ni_ *= 2

    def retract(self):
        self.residue_ = self.last_residue_
        self.chi = self.last_chi
        self.theta = self.last_theta

    def solve_problem(self):
        self.compute_residue()
        self.build_normal_equation()
        self.initial_solver()
        i = 0
        while i < 100:
            success = False
            while not success:
                self.solve_normal_equation()

                if np.linalg.norm(self.delta_theta) < 1e-8:
                    return
                success = self.in_trusted_region()
                print("iteration {iteration}: {lamda} {theta}, {chi}".format(iteration=i, theta=self.theta.reshape([-1]),
                                                                             lamda=self.lamda, chi=self.chi[0][0]))
                if success:
                    self.update_lamda_and_trusted_region()
                    self.build_normal_equation()
                else:
                    self.update_lamda_and_trusted_region()
                    self.retract()
            i = i + 1

            print("accepted: iteration {iteration}: {lamda} {theta}, {chi}".format(iteration=i, theta=self.theta.reshape([-1]),
                                                                 lamda=self.lamda, chi=self.chi[0][0]))

    def compute_residue(self):
        self.compute_object()
        self.last_residue_ = self.residue_
        self.last_chi = self.chi
        self.residue_ = self.object_ - self.y
        self.chi = self.residue_.T @ self.residue_

    def update_parameter(self):
        self.last_theta = self.theta
        self.theta = self.theta + self.delta_theta

    def compute_object(self):
        pass

    def compute_jacobian(self):
        pass

    def get_object(self):
        return self.object_

    def get_residue(self):
        return self.residue_, self.chi

    def get_theta(self):
        return self.theta


class SquareProblemSolver(ProblemSolver):
    def __init__(self, x, y):
        ProblemSolver.__init__(self, x, y)

    def compute_object(self):
        a, b, c = self.theta[0, 0], self.theta[1, 0], self.theta[2, 0]
        self.object_ = a * self.x * self.x + b * self.x + c

    def compute_jacobian(self):
        temp_ = np.zeros_like(x) + 1
        self.jacobian_ = np.hstack([x * x, x, temp_])


class ExpSquareProblemSolver(ProblemSolver):
    def __init__(self, x, y):
        ProblemSolver.__init__(self, x, y)

    def compute_object(self):
        a, b, c = self.theta[0, 0], self.theta[1, 0], self.theta[2, 0]
        self.object_ = np.exp(a * self.x * self.x + b * self.x + c)

    def compute_jacobian(self):
        temp_ = np.zeros_like(x) + 1
        self.jacobian_ = self.object_ * np.hstack([x * x, x, temp_])

if __name__ == "__main__":
    begin, end, sample = 0, 1, 1000
    sigma = 0.2
    x = np.arange(begin, end, (end - begin) / sample).reshape([sample, 1])
    y_ = np.exp(x * x + 2 * x + 1)
    y = y_ + np.random.normal(0, sigma, [sample, 1])
    solver = ExpSquareProblemSolver(x, y)

    initial_guess = np.array([0, 0, 0]).reshape([3, 1])
    solver.set_estimate(initial_guess)
    solver.solve_problem()

    y_predict = solver.get_object()
    plt.plot(x, y_predict, "r")# x, y_, "g")
    plt.scatter(x, y)
    plt.show()




