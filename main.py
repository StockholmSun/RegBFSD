# -*- coding: utf-8 -*-
"""
GenAlg-based Regularization Fine-Tuning for RegBFSD

Yutai Sun @ Southeast University

Acknowledge: The code is rewritten from Sample Codes of Pymoo and Dr. Yingmeng Ge.
"""

import time
import numpy as np
from GAQ import myProblem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair=

N = 3

class MyOutput(Output):
    def __init__(self):
        super().__init__()
        self.x_mean = Column("L_mean", width=13)
    def update(self, algorithm):
        super().update(algorithm)
        self.x_mean.set(np.mean(algorithm.pop.get("X")))
        
class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []
    def notify(self, algorithm):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)


if __name__ == '__main__':
    hyperparameters = {'N': N}
    myPro = myProblem(hyperparameters)
    pop_size = 10
    n_gen = 2000

    good_init = np.array([110,100,110]).astype(int)
    all_pop = np.zeros([pop_size,good_init.shape[0]],dtype=int)
    idx = 0
    for n in range(good_init.shape[0]):
        all_pop[:, n] = np.clip(good_init[n]+np.round(2*np.random.randn(pop_size)), 1, 200)
    all_pop[0,:] = good_init
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=all_pop,
        crossover=SBX(prob=1.0, eta=15, vtype=float, repair=RoundingRepair()),
        mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True
    )

    callback = MyCallback()
    start = time.time()
    res = minimize(myPro, algorithm, ('n_gen', n_gen), callback=callback, seed=1, verbose=True)