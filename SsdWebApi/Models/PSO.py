# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:59:59 2020

@author: Federico
"""
import numpy as np
import random

def paraboloid(xvec):
    sum = 0
    for i in range(len(xvec)):
        sum += np.power(xvec[i], 2)
    return sum;

def compute_fitness(position, portfolio):
    #return paraboloid(position)
    investment_per_asset = portfolio.investment * position
    return

class Portfolio:
    def __init__(self, indices_variations, investment, risk_weight, return__weight):
        self.indices_variations = indices_variations
        self.investment = investment
        self.risk_weight = risk_weight
        self.return__weight = return__weight
    
class Particle:
    def __init__(self, dimensionality, neigh_size):
        self.position = np.zeros(dimensionality, dtype=np.float)
        self.velocity = np.zeros(dimensionality, dtype=np.float)
        self.pbest = np.zeros(dimensionality, dtype=np.float) # best local solution (position)
        self.lbest = np.zeros(dimensionality, dtype=np.float) # best neighborhood solution (position)
        self.neighborhood = np.zeros(neigh_size, dtype=np.int) # neighbors indexes (used as IDs)
        self.fit = 0 # actual particle fitness
        self.neigh_fitbest = 0 # best fitness of the neighborhood
        self.fitbest = 0 # best fitness in the history of this particle
    
class PSO:
    def __init__(self, portfolio, c0, c1, c2, pos_min, pos_max, dimensionality, num_particles, num_neighbors):
        self.portfolio = portfolio
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.dimensionality = dimensionality
        self.num_neighbors = num_neighbors
        self.particles = self._init_particles(num_particles)
        self.gfitbest = np.inf
        self.gbest = np.zeros(dimensionality, dtype=np.float)
        
    # --> particles init
    def _init_particles(self, num_particles):
        particles = []
        for i in range(num_particles):
            p = Particle(self.dimensionality, self.num_neighbors)
            for j in range(self.dimensionality):
                p.position[j] = random.random() * (self.pos_max-self.pos_min)-self.pos_min
                p.velocity[j] = (random.random()-random.random()) * 0.5 * (self.pos_max-self.pos_min)-self.pos_min
                p.pbest[j] = p.position[j]
                p.lbest[j] = p.position[j]
            p.fit = compute_fitness(p.position, self.portfolio)
            p.fitbest = p.fit
            for j in range(self.num_neighbors): # neighbors selection (by index)
                # each particle neighborhood is made of X particles randomly selected between all particles
                while True:
                    id = random.randrange(num_particles)
                    if id not in p.neighborhood:
                        p.neighborhood[j] = id
                        break
            particles.append(p)
        return particles
    
    def run(self, maxiter):    
        # ---> main loop
        for iter in range(maxiter):
            print("iter {0} - gfitbest {1}".format(iter, self.gfitbest))
            for i in range(len(self.particles)): # for each particles
                actual_particle = self.particles[i]
                for d in range(self.dimensionality): # for each dimension
                    # stochastic coefficients
                    w1 = self.c1*random.random()
                    w2 = self.c2*random.random()
                    actual_particle.velocity[d] = self.c0 * actual_particle.velocity[d] + \
                        w1 * (actual_particle.pbest[d] - actual_particle.position[d]) + \
                        w2 * (actual_particle.lbest[d] - actual_particle.position[d])
                    actual_particle.position[d] += actual_particle.velocity[d] # update position with the new velocity
                    
                    # check bounds
                    if actual_particle.position[d] < self.pos_min:
                        actual_particle.position[d] = self.pos_min
                        actual_particle.velocity[d] = -actual_particle.velocity[d]
                    elif actual_particle.position[d] > self.pos_max:
                        actual_particle.position[d] = self.pos_max
                        actual_particle.velocity[d] = -actual_particle.velocity[d]
                    
                    # update particle fitness
                    actual_particle.fit = compute_fitness(actual_particle.position, self.portfolio)
                    if actual_particle.fit < actual_particle.fitbest: # update pbest
                        actual_particle.fitbest = actual_particle.fit
                        for j in range(self.dimensionality):
                            actual_particle.pbest[j] = actual_particle.position[j]
                    
                    # update lbest
                    actual_particle.neigh_fitbest = np.inf
                    for j in range(self.num_neighbors):
                        if self.particles[actual_particle.neighborhood[j]].fit < actual_particle.neigh_fitbest:
                            actual_particle.neigh_fitbest = self.particles[actual_particle.neighborhood[j]].fit
                            for k in range(self.dimensionality):
                                actual_particle.lbest[k] = self.particles[actual_particle.neighborhood[j]].position[k]
                                
                    # update gbest
                    if actual_particle.fit < self.gfitbest:
                        self.gfitbest = actual_particle.fit
                        for j in range(self.dimensionality):
                            self.gbest[j] = actual_particle.position[j]
        return self.gfitbest