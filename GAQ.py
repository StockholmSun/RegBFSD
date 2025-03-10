# -*- coding: utf-8 -*-

import autograd.numpy as anp
from pymoo.core.problem import Problem
from multiprocessing import Pool
from Cal import CalculateAlpha # For RegBFSD calculation

class myProblem(Problem):
    
    def __init__(self, hyperparameters):
        
        # Parameters Setting
        self.N = hyperparameters['N']

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))
        
        self.Nobj = 1
        self.Nconstr = 1
        
        xl = anp.ones(self.N)*0
        xu = anp.ones(self.N)*200
        
        super().__init__(n_var=self.N,
                         n_obj=self.Nobj,
                         n_constr=self.Nconstr,
                         xl=xl,
                         xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        
        # Quantization scheme
        pop_size = x.shape[0]
        f = anp.zeros([pop_size])
        g = anp.zeros([pop_size])
        for n in range(pop_size):
            f[n] = CalculateAlpha(x[n]) # For RegBFSD calculation here.
            g[n]=0
            print(x[n])
            print(f[n])
            with open('log.txt', 'a') as ff:
                ff.write(str(x[n])+'\n')
                ff.write(str(f[n])+'\n')
            
        
        out['F'] = f
        out['G'] = g
