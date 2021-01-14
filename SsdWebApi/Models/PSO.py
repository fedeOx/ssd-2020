# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:59:59 2020

@author: Federico
"""
import numpy as np
import random
import math

def paraboloid(xvec):
    sum = 0
    for i in range(len(xvec)):
        sum += np.power(xvec[i], 2)
    return sum;

def compute_capital_variations(init_capital, asset_daily_variations):
    res = [init_capital] # first day capital
    for i in range(1, len(asset_daily_variations)):
        res.append((1 + asset_daily_variations[i]) * res[i-1]) # capital_t'' = (1 + var_t'') * capital_t'
    return res

def compute_ret_risk(position, portfolio):
    #position = np.array([0.05, 0.05, 0.2, 0.1, 0.05, 0.3, 0.25]) # SOLO PER TEST
    
    # compute assets capital daily variations
    capital_per_asset = portfolio.capital * position # initial capital distribution per index
    capital_variations = []
    for i in range(portfolio.assets_variations.shape[1]): # for each asset
        asset_init_capital = capital_per_asset[i]
        asset_variations = portfolio.assets_variations[:,i]
        v = compute_capital_variations(asset_init_capital, asset_variations)
        capital_variations.append(v)
    capital_variations = np.array(capital_variations).transpose()
    
    # compute portfolio daily values and variations (%)
    portfolio_daily_values = []
    portfolio_variations_perc = []
    for i in range(len(capital_variations)):
        portfolio_daily_values.append(capital_variations[i].sum())
        if i>0:
            portfolio_variations_perc.append((portfolio_daily_values[i] - portfolio_daily_values[i-1]) / portfolio_daily_values[i-1])
    
    one_month = 22
    # return is the average of the portfolio values in the last month
    ret = np.average(portfolio_daily_values[-one_month:])

    # compute risk (std.dev.)
    maa = []
    sqee = []
    for i in range(one_month, len(portfolio_daily_values)+1):
        # mobile average on one month of portfolio values
        mobile_average = np.average(portfolio_daily_values[i-one_month:i])
        maa.append(mobile_average)
        # squared error on mobile average
        squared_error = pow((portfolio_daily_values[i-1] - maa[i-one_month]), 2)
        sqee.append(squared_error)
    risk = math.sqrt(sum(sqee)/len(sqee))
    
    return ret, risk

def compute_fitness(ret, risk, portfolio, avg_ret, avg_risk):
    print("return: {0} - risk: {1}".format(ret, risk))
    # linear combination between return and risk
    return (1-portfolio.alpha) * ret/avg_ret - portfolio.alpha * risk/avg_risk

class Portfolio:
    def __init__(self, assets_variations, capital, alpha):
        self.assets_variations = assets_variations
        self.capital = capital
        self.alpha = alpha
    
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
        self.ret = 0.0 # return associated to the position of this particle
        self.risk = 0.0 # risk associated to the position of this particle
    
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
        self.avgRet, self.avgRisk = self._compute_avg_ret_risk(10) # for compatibility in ret and risk dimensions (obj function maximization)
        self.particles = self._init_particles(num_particles)
        self.gfitbest = -np.inf # best global fitness
        self.gbest = np.zeros(dimensionality, dtype=np.float)
        self.bestRet = 0.0
        self.bestRisk = 0.0
            
    def _rand_position(self):
        position = []
        max_threshold = 1.0
        for d in range(self.dimensionality-1): 
            p = random.uniform(self.pos_min, self.pos_max)
            max_threshold -= p
            residual_dims = self.dimensionality-1-d
            min_threshold = self.pos_min * (residual_dims)
            if max_threshold < min_threshold:
                surplus = min_threshold - max_threshold # surplus used to correct p
                position.append(p-surplus) # p correction
                for k in range(d+1, self.dimensionality-1):
                    position.append(self.pos_min)
                max_threshold = self.pos_min
                break
            position.append(p)
        
        position.append(max_threshold) # the last dpos is the residual (ensures sum to 1.0)
            
        return np.array(position)

    def _ensure_sum_1(self, pos):
        res = np.array([])
        for p in pos: # ensures sum to 1.0
            res = np.append(res, p/sum(pos))
        return res
    
    def _ensure_constraints(self, pos, vel):
        # checks each dimension of pos is in [0.05,0.7]
        fixed_pos = np.array([])
        fixed_vel = np.array([])
        for d in range(len(pos)):
            if pos[d] < self.pos_min:
                fixed_pos = np.append(fixed_pos, self.pos_min)
                fixed_vel = np.append(fixed_vel, -vel[d])
            elif pos[d] > self.pos_max:
                fixed_pos = np.append(fixed_pos, self.pos_max)
                fixed_vel = np.append(fixed_vel, -vel[d])
            else:
                fixed_pos = np.append(fixed_pos, pos[d])
                fixed_vel = np.append(fixed_vel, vel[d])
        return fixed_pos, fixed_vel
    
    def _position_fixing(self, pos, vel):
        # check each dimension of pos is in [0.05,0.7]
        fixed_pos, fixed_vel = self._ensure_constraints(pos, vel)
        # check that dimensions of pos sums to 1.0
        fixed_pos = self._ensure_sum_1(fixed_pos)
        # correct eventual underflows
        for i in range(len(fixed_pos)):
            if fixed_pos[i] < self.pos_min:
                delta = self.pos_min - fixed_pos[i]
                fixed_pos[i] = fixed_pos[i] + delta # correct fixed_pos[i] to 0.05
                # select a random dpos that can be reduced by delta
                targetId = random.randrange(0, self.dimensionality)
                while fixed_pos[targetId]-delta < self.pos_min:
                    targetId = random.randrange(0, self.dimensionality)
                fixed_pos[targetId] -= delta # reduce the randomly selected dpos by delta
        return fixed_pos, fixed_vel
    
    def _compute_avg_ret_risk(self, n_port):
        rets = []
        risks = []
        for i in range(n_port):
            position = self._rand_position()
            ret, risk = compute_ret_risk(position, self.portfolio)
            rets.append(ret)
            risks.append(risk)
        return np.average(rets), np.average(risks)
    
    # --> particles init
    def _init_particles(self, num_particles):
        particles = []
        for i in range(num_particles):
            p = Particle(self.dimensionality, self.num_neighbors)
            for j in range(self.dimensionality):
                p.velocity[j] = (random.random()-random.random()) * 0.5 * (self.pos_max-self.pos_min)-self.pos_min
            p.position = self._rand_position()
            p.pbest = p.position
            p.lbest = p.position
            p.ret, p.risk = compute_ret_risk(p.position, self.portfolio)
            p.fit = compute_fitness(p.ret, p.risk, self.portfolio, self.avgRet, self.avgRisk)
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

                actual_particle.position, actual_particle.velocity = self._position_fixing(
                                                                        actual_particle.position,
                                                                        actual_particle.velocity)
                    
                # update particle ret and risk
                actual_particle.ret, actual_particle.risk = compute_ret_risk(actual_particle.position, self.portfolio)
                
                # update particle fitness
                actual_particle.fit = compute_fitness(actual_particle.ret, actual_particle.risk,
                                                      self.portfolio, self.avgRet, self.avgRisk)
                
                # update pbest
                if actual_particle.fit > actual_particle.fitbest:
                    actual_particle.fitbest = actual_particle.fit
                    for j in range(self.dimensionality):
                        actual_particle.pbest[j] = actual_particle.position[j]
                    
                # update lbest
                actual_particle.neigh_fitbest = -np.inf
                for j in range(self.num_neighbors):
                    if self.particles[actual_particle.neighborhood[j]].fit > actual_particle.neigh_fitbest:
                        actual_particle.neigh_fitbest = self.particles[actual_particle.neighborhood[j]].fit
                        for k in range(self.dimensionality):
                            actual_particle.lbest[k] = self.particles[actual_particle.neighborhood[j]].position[k]
                                
                # update gbest
                if actual_particle.fit > self.gfitbest:
                    self.gfitbest = actual_particle.fit
                    self.bestRet = actual_particle.ret
                    self.bestRisk = actual_particle.risk
                    for j in range(self.dimensionality):
                        self.gbest[j] = actual_particle.position[j]
        
        return self.gbest, self.gfitbest, self.bestRet, self.bestRisk 