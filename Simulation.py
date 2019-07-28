# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
from numpy import sqrt, exp, log
import math

def merton_jump_diffusion_simulate(params):
    '''
    Merton Jump Diffusion simulation
    S0 = 100    # spot price
    K  = 100    # strike price
    r  = 0.05   # risk free rate
    q  = 0.02   # dividend yield
    σ  = 0.2    # volatility
    t  = 1      # maturity
    λ  = 3      # jump rate
    γ  = 0.1    # jump mean
    δ  = 0.2    # jump volatility
    Δt = 0.001  # discretization step
    s  = 123456 # seed for pseudorandom number generation 
    '''
    S0 = params['S0']
    K  = params['K']
    r  = params['r']
    q  = params['q']
    σ  = params['σ']
    t  = params['t']
    λ  = params['λ']
    γ  = params['γ']
    δ  = params['δ']
    Δt = params['Δt']
    s  = params['seed']
    np.random.seed(np.random.randint(0,65535))

    #random.normal(loc=0.0, scale=1.0, size=None)
    #print(f"μ={μ}, σ={σ}, Δt={Δt}, N={N}\n")

    M = np.int(1/Δt)
    S = np.zeros(M + 1)
    N = np.zeros(M + 1, dtype=np.int32)
    S[0] = S0

    k = exp(γ + δ*δ/2) - 1
    µ = (r - q - λ*k)

    for t in range(1, M + 1):
        Z = norm.rvs(0, 1)
        N[t] = poisson.rvs(λ * Δt)
        X = sum([norm.rvs(γ, δ*δ) for i in range(0, N[t])])
        S[t] = S[t - 1] + (μ - σ*σ/2) * Δt + σ * sqrt(Δt) * Z + X
    return exp(S), N

def heston_stochastic_volatility_simulate(params):
    '''
    Heston stochastic volatility model
    S0 = 100    # spot price
    K  = 100    # strike price
    r  = 0.05   # risk risk-free interest rate for the stock
    q  = 0.02   # dividend yield
    κ  =        # mean reversion speed for the variance
    θ  =        # the mean reversion level for the variance
    σ  = 0.2    # volatility of variance
    ν0 = 0.2    # initial level of the variance
    ρ  =        # correlation between the two Brownian motions W1 and W2

    t  = 1      # maturity
    Δt = 0.001  # discretization step
    s  = 123456 # seed for pseudorandom number generation 
    '''
    S0 = params['S0']
    K  = params['K']
    r  = params['r']
    q  = params['q']
    κ  = params['κ']
    θ  = params['θ']
    σ  = params['σ']
    v0 = 0.35 # params['v0']
    ρ  = params['ρ']

    #t  = params['t']
    Δt = params['Δt']
    s  = params['seed']
    #np.random.seed(s)

    np.random.seed(np.random.randint(0,65535))

    M = np.int(1/Δt)
    S = np.zeros(M + 1)
    S[0] = S0
    v = np.zeros(M + 1)
    v[0] = v0

    for t in range(1, M + 1):

        # Generate two independent random variables $Z_1$ and $Z_2$
        Z1, Z2 = norm.rvs(0, 1), norm.rvs(0, 1)

        # define Z_V = Z_1 and Z_S = ρZ V + √(1 − ρ^2) * Z_2
        ZV = Z1
        ZS = ρ * ZV + sqrt(1-ρ*ρ) * Z2

        # Stochastic variance
        v[t] = ( sqrt(v[t-1]) + σ*sqrt(Δt)*ZV/2 )**2 + κ*(θ - v[t-1])*Δt - σ*σ*Δt/4

        # log-stock price
        S[t] = S[t - 1] + (r - q - v[t-1]/2) * Δt + sqrt(v[t-1] * Δt) * ZS

    return exp(S), v



def binomial_randomwalk_multiplicative_simulate(params):
    '''
    Multiplicative binomial random walk simulation
    S0 = 100   # initial price
    u  = 1.1   # "up" factor
    d  = 0.9   # "down" factor, currently set to the multiplicative inverse of u
    p  = 0.5   # probability of "up"
    T  = 30    # time-step size
    N  = 50000 # sample size (no. of simulations)
    '''
    S0 = params['S0']
    p  = params['p']
    u  = params['u']
    d  = 1/u
    T  = params['T']
    N  = params['N']
    P  = params['P']
    s  = params['seed']
    np.random.seed(s)
    # Simulating N paths with T time steps
    S = np.zeros((T + 1, N))
    S[0] = S0
    for t in range(1, T + 1):
        z = np.random.rand(N)  # pseudorandom numbers
        S[t] = S[t - 1] * ( (z<=p)*u + (z>p)*d )
          # vectorized operation per time step over all paths
    return S

def returnSampleDescriptiveStatistics(S, K, decimalPlaces=2):
    '''
    Returns mean, variance, skewness, kurtosis, etc.
    '''
    samp_mean = f"{S[-1, :].mean():.2f}"
    samp_var = f"{S[-1, :].var():.2f}"
    samp_skew = f"{skew(S[-1, :]):.2f}"
    samp_kurt = f"{kurtosis(S[-1, :]):.2f}"
    
    #samp_X_minus_K = f"{S[-1, :].mean():.2f}"
    #samp_K_minus_X = 


#def random_walk

def generate_path(S0, r, sigma, T, M):
	''' 
	Source: Hilpisch 2017, p. 236
	
	Function to simulate a geometric Brownian motion.
	Parameters
	==========
		S0: float
		initial index level
		r: float
		constant risk-less short rate
		sigma: float
		instantaneous volatility
		T: float
		date of maturity (in year fractions)
		M: int
		number of time intervals
	Returns
	=======
		path: pandas DataFrame object simulated path
	'''
	# length of time interval

	dt = float(T) / M
	# random numbers
	np.random.seed(100000)
	rn = np.random.standard_normal(M + 1)
	rn[0] = 0 # to keep the initial value
	# simulation of path
	path = S0 * np.exp(np.cumsum((r - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * rn))
	# setting initial value
	path = pd.DataFrame(path, columns=['index'])
	return path


