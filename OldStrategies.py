#! /usr/bin/env python

"""
SimStrategies
Dependency for matingsim.py
Contains the mating strategies for male and female birds
By convention male strategies are even, female strategies are odd
written by Ammon Perkes (perkes.ammon@gmail.com) at University of Pennsylvania
2016
"""

import sys, os
import numpy as np
from matplotlib import pyplot as plt

global M_SHIFT, F_SHIFT
M_SHIFT = .001
F_SHIFT = .001
ALPHA = 1.0
KAPPA = 1.0
## Branching function to select the correct function and return response. 
# In the case of males, it returns a new investment matrix 
# In the case of females, it returns a new reward matrix 

def choose(strategy, resources, history, num, alpha = ALPHA, kappa = KAPPA):
## Male strategies (by convention, even)
    if strategy == 0:
        return m_evasive(resources, history, num)
    elif strategy == 2:
        return m_greedy(resources, history, num)
    elif strategy == 4:
        return m_greedy_theft(resources, history, num)
    elif strategy == 6:
        return m_relative_benefit_adjacency(resources, history, num)
    elif strategy == 'M1':
        return m_new_strategy(resources, history, num) 
    elif strategy == 'M0':
        return m_classic_strategy(resources, history, num)
    
## Female strategies (by convention, odd)
    elif strategy == 'F1':
        return f_new_strategy(resources, history, num)
    elif strategy == 'F0':
        return f_classic_strategy(resources, history, num)
    elif strategy == 1:
        return f_high_investors(resources, history, num)
    elif strategy == 3: 
        return f_investment(resources, history, num)
    elif strategy == 5:
        return f_relative_investment(resources, history, num)
    elif strategy == 7:
        return f_relative_investment_a(resources, history, num, alpha)
    elif strategy == 9:
        return f_investment_a(resources, history, num, alpha)
    elif strategy == 11:
        return f_adjacent_quality(resources, history, num, alpha, kappa)
    elif strategy == 13:
        return f_adjacent_quality_relative(resources, history, num, alpha, kappa)
    else:
        print "No strategy found, quitting"
        sys.exit()

## Some basic functions used in various strategies. NOTE: due to python's structure, changing output within a function will change the output itself, since it's merely a pointer towards the object itself. With that in mind, all the returns are sort of more for good habit, just remember that you can't make an identity and then change one of the lists (that goes for arrays and lists)
def normalize(output,resources):
    # first set any negative to 0
    for i in range(len(output)): 
        if output[i] < 0:
            output[i] = 0.0
    # Then normalize to the available resources
    n_output = output * resources / sum(output)
    # Option to allow for negative male investment, if wanted
    #n_output = output * resources / sum(abs(output))
    return n_output

# Function to normalize females, because of indexing they don't work the same way
# This creates a ation matrix (really a vector: [1 1 1 total 1]
def f_normalize(current_reward,resources,num):
    # This allows for negative reward, but actually counts that as investment
    total = abs(current_reward).sum(0)[num]
    #total = current_reward.sum(0)[num]
    transform = np.ones(current_reward[0].size)
    transform[num] = total
    current_reward = current_reward * resources / (transform + .0001) 
    return current_reward

def relocate(output,amount,source,sink):
    output[source] = output[source] - amount
    output[sink] = output[sink] + amount
    return output

## These functions are dumb, just write it in line, it makes it more straight forward
def add_output(output,amount,sink):
    output[sink] = output[sink] + amount
    return output

def subtract_output(output,amount,sink):
    output[source] = output[source] - amount
    return output

######################
## Male strategies: ##
######################

# Inputs: Resources, history, num of male
# Output: A New investment matrix, edited for that one male. 

def m_new_strategy(resources, history, num):
    shift_scaler = .1
    
    current_turn = history.current_turn
    
    previous_invest = history.invest_matrix2[current_turn]

    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest[:]
    
    my_previous_reward = history.reward_matrix2[current_turn - 1,num,:]
    my_previous_invest = history.invest_matrix2[current_turn - 1,num,:]
    my_current_invest = np.zeros(np.shape(my_previous_invest))
    
    profit = my_previous_reward[:] / (my_previous_invest[:] + .0001)
    #profit[0:history.n_males] = -1000
    #female_profit = profit[history.n_males:]
    avg_profit = profit[my_previous_invest > 0.0].mean()
    #avg_profit = profit[history.n_males:].mean()
    max_profit = profit[history.n_males:].max()
    min_profit = profit[history.n_males:].min()
    profit_thresh = max_profit - (max_profit - avg_profit) * .25
    
    shift_matrix = (profit - avg_profit) / (max_profit - min_profit + .0001) * shift_scaler
    

    my_current_invest = my_previous_invest + shift_matrix
    my_current_invest[my_current_invest < 0] = 0
    #my_current_invest[profit > profit_thresh] = profit[profit > profit_thresh] - avg_profit
    my_current_invest[0:history.n_males] = 0
    #my_current_invest[profit <= profit_thresh] = 0
    
    my_current_invest = my_current_invest / (sum(abs(my_current_invest)) +.0001) * resources ## this normalizes it to resources...
    current_invest[num,:] = my_current_invest

    return current_invest

# Inputs: Resources, history, num of male
# Output: A New investment matrix, edited for that one male.

def m_classic_strategy(resources, history, num):
    shift_scaler = .01
    
    current_turn = history.current_turn
    
    previous_invest = history.invest_matrix2[current_turn]
    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest[:]
    
    my_previous_reward = history.reward_matrix2[current_turn - 1,num,:]
    my_previous_invest = history.invest_matrix2[current_turn - 1,num,:]
    my_current_invest = np.zeros(np.shape(my_previous_invest))
    
    profit = my_previous_reward[:] / (my_previous_invest[:] + .0001)

    avg_profit = profit[my_previous_invest > 0.0].mean()
    
    shift_matrix = np.zeros(np.shape(profit))    
    shift_matrix[profit > avg_profit] = shift_scaler
    shift_matrix[profit < avg_profit] = -shift_scaler

    my_current_invest = my_previous_invest + shift_matrix
    my_current_invest[my_current_invest < 0] = 0
    my_current_invest[0:history.n_males] = 0
    
    my_current_invest = my_current_invest / (sum(abs(my_current_invest)) +.0001) * resources ## this normalizes it to resources...
    current_invest[num,:] = my_current_invest    
    return current_invest
                
# Strategy to avoid crowding: 0
def m_evasive(resources, history, num):    
    current_turn = history.current_turn    
    previous_reward = history.reward_matrix[current_turn-1]
    previous_invest = history.invest_matrix[current_turn-1]
    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest
    f_sums = previous_invest.sum(0)
    avg_invest = f_sums / float(history.n_males)
# For each female, add or subtract based on whether investment is above or below average
# Since avg_invest doesn't change, this can happen without confounding itself
    for f in range(history.n_females):
        if previous_invest[num,f] >= avg_invest[f]:
            #print num
            #print f
            #print M_SHIFT
            print "Current turn: " + str(history.current_turn)
            print "Investing by Male: " + str(num) + "in female: " + str(f) + ": " + str(M_SHIFT)
            print "Current Investment:"
            print current_invest
            #print current_invest[num,f]
            #current_invest[num,f] = add_output(current_invest[num,f],M_SHIFT,f)
            current_invest[num,f] = current_invest[num,f] + M_SHIFT
        else:
            print "Divesting by " + str(num) + "in f: " + str(f)
            print current_invest
            #current_invest[num,f] = subtract_output(current_invest[num,f],M_SHIFT,f)
            current_invest[num,f] = current_invest[num,f] - M_SHIFT
# Normalize output
    current_invest[num] = normalize(current_invest[num],resources)
    #print "normalizing across male " + str(num)
    #print current_invest
    return current_invest

# Strategy to maximize rate of return: 2
def m_greedy(resources, history, num):
    #print history.current_turn
    current_turn = history.current_turn    
    previous_reward = history.reward_matrix[current_turn-1]
    previous_invest = history.invest_matrix[current_turn-1]
    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest
    benefit = np.zeros(history.n_females)
    for f in range(history.n_females):
        benefit[f] = previous_reward[num,f] / (previous_invest[num,f] + .001)
    avg_benefit = benefit[np.nonzero(benefit)].mean()   # This is critical for monogamy to function
    #avg_benefit = benefit.mean()
# Invest more if you're beating your avg rate of return, invest less otherwise
    for f in range(history.n_females):
        if benefit[f] > avg_benefit:

            current_invest[num,f] = current_invest[num,f] + M_SHIFT
        else:

            current_invest[num,f] = current_invest[num,f] - M_SHIFT
# Normalize output
    current_invest[num] = normalize(current_invest[num],resources)
    return current_invest

def m_greedy_theft(resources, history, num):
    #print history.current_turn
    current_turn = history.current_turn    
    previous_reward = history.reward_matrix[current_turn-1]
    previous_invest = history.invest_matrix[current_turn-1]
    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest
    benefit = np.zeros(history.n_females)
    for f in range(history.n_females):
        benefit[f] = previous_reward[num,f] / (previous_invest[num,f] + .001)
    avg_benefit = benefit[np.nonzero(benefit)].mean()   
    #avg_benefit = benefit.mean()
# Invest more if you're beating your avg rate of return, invest less otherwise
    count = 0
    aces = []
    for f in range(history.n_females):
        if benefit[f] <= avg_benefit:
            current_invest[num,f] = previous_invest[num,f] - M_SHIFT
            if current_invest[num,f] < 0:
                current_invest[num,f] = 0.0
        else:
            aces.append(f)
    divestment = sum(previous_invest[num,:] - current_invest[num,:])
    for a in aces:
        current_invest[num,a] = current_invest[num,a] + divestment / len(aces)
# Normalize output
    current_invest[num] = normalize(current_invest[num],resources)
    return current_invest  

def m_relative_benefit_adjacency(resources, history, num):
    current_turn = history.current_turn
    previous_reward = history.reward_matrix[current_turn-1]
    previous_invest = history.invest_matrix[current_turn-1]
    current_invest = np.empty_like(previous_invest)
    benefit = np.zeros(history.n_females)
    for f in range(history.n_females):
        benefit[f] = previous_reward[num,f] / (previous_invest[num,f] + .001)
    avg_benefit = benefit[np.nonzero(benefit)].mean()
    count = 0
    aces = []
    for f in range(history.n_females):
        relative_benefit = benefit[f] / (sum(benefit) + .0001) * resources  
        current_invest[num,f] = relative_benefit
    if sum(current_invest[num,:]) > 1.0:
        current_invest = current_invest / sum(current_invest) * resources
    return current_invest

def m_relative_benefit_adjacency2(resources, history, num):
    current_turn = history.current_turn
    previous_reward = history.reward_matrix2[current_turn-1]
    previous_invest = history.invest_matrix[current_turn-1]
    current_invest = np.empty_like(previous_invest)
    benefit = np.zeros(history.n_females)
    for f in range(history.n_females):
        benefit[f] = previous_reward[num,f] / (previous_invest[num,f] + .001)
    avg_benefit = benefit[np.nonzero(benefit)].mean()
    count = 0
    aces = []
    for f in range(history.n_females):
        relative_benefit = benefit[f] / (sum(benefit) + .0001) * resources  
        current_invest[num,f] = relative_benefit
    if sum(current_invest[num,:]) > 1.0:
        current_invest = current_invest / sum(current_invest) * resources
    return current_invest

########################
## Female Strategies: ##
########################

def f_new_strategy(resources, history, num):
    shift_scaler = .1
    current_turn = history.current_turn
    
    previous_invest = history.invest_matrix2[current_turn]
    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest[:]
    
    my_previous_reward = history.reward_matrix2[current_turn - 1,history.n_males + num,:]
    my_previous_invest = history.invest_matrix2[current_turn - 1,history.n_males + num,:]
    my_current_invest = np.zeros(np.shape(my_previous_invest))
    
    profit = my_previous_reward[:] / (my_previous_invest[:] + .0001)
    profit[history.n_males:] = 0
    #avg_profit = profit[:history.n_males].mean()
    male_profit = profit[:history.n_males]
    male_invest = my_previous_invest[:history.n_males]
    avg_profit = male_profit[male_invest > 0.0].mean() 
    
    #avg_profit = profit[my_previous_invest > 0.0].mean()
    max_profit = profit[:history.n_males].max()
    min_profit = profit[:history.n_males].min()
    profit_thresh = max_profit - (max_profit - avg_profit) * .25
    
    shift_matrix = (profit - avg_profit) / (max_profit - min_profit + .0001) * shift_scaler
    my_current_invest = my_previous_invest + shift_matrix
    
    #my_current_invest[profit > profit_thresh] = profit[profit > profit_thresh] - avg_profit
    my_current_invest[history.n_males:] = 0
    #my_current_invest[profit <= profit_thresh] = 0
    my_current_invest[my_current_invest < 0] = 0
        
    my_current_invest = my_current_invest / (sum(abs(my_current_invest)) + .0001) * resources ## this normalizes it to resources...
    current_invest[history.n_males + num,:] = my_current_invest
    
    return current_invest

def f_classic_strategy(resources, history, num):
    shift_scaler = .01
    
    current_turn = history.current_turn
    
    previous_invest = history.invest_matrix2[current_turn]
    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest[:]
    
    my_previous_reward = history.reward_matrix2[current_turn - 1,history.n_males + num,:]
    my_previous_invest = history.invest_matrix2[current_turn - 1,history.n_males + num,:]
    my_current_invest = np.zeros(np.shape(my_previous_invest))
    
    profit = my_previous_reward[:] / (my_previous_invest[:] + .0001)

    avg_profit = profit[my_previous_invest > 0.0].mean()
    
    shift_matrix = np.zeros(np.shape(profit))    
    shift_matrix[profit > avg_profit] = shift_scaler
    shift_matrix[profit < avg_profit] = -shift_scaler

    my_current_invest = my_previous_invest + shift_matrix
    my_current_invest[my_current_invest < 0] = 0
    my_current_invest[history.n_males:] = 0
    
    my_current_invest = my_current_invest / (sum(abs(my_current_invest)) +.0001) * resources 
    current_invest[history.n_males + num,:] = my_current_invest    
    return current_invest
                
# Strategy to reward more investment: 1
def f_high_investors(resources, history, num):
    current_turn = history.current_turn
    previous_reward = history.reward_matrix[current_turn-1]
    current_reward = np.empty_like(previous_reward)
    current_reward[:] = previous_reward
    current_invest = history.invest_matrix[current_turn] 
    m_invest = current_invest[:,num]
    #total_invest = m_invest.sum()
    avg_invest = m_invest.mean()
# For each male, add or subtract based on if it's above average
    for m in range(history.n_males):
        if m_invest[m] > avg_invest:
            current_reward[m,num] = previous_reward[m,num] + F_SHIFT
        else:
            current_reward[m,num] = previous_reward[m,num] - F_SHIFT
    current_reward = f_normalize(current_reward,resources,num)
    return current_reward

# Strategy to reward more investment linearly: 3
def f_investment(resources, history, num):
    current_turn = history.current_turn
    previous_reward = history.reward_matrix[current_turn-1]
    current_reward = np.empty_like(previous_reward)
    current_reward[:] = previous_reward
    previous_invest = history.invest_matrix[current_turn-1] 
    m_invest = previous_invest[:,num]
    #total_invest = m_invest.sum()
    avg_invest = m_invest.mean()
# For each male, reward is a function of investment
    for m in range(history.n_males):
        current_reward[m,num] = previous_invest[m,num]
    current_reward = f_normalize(current_reward,resources,num)
    return current_reward

def f_investment_a(resources, history, num, alpha):
    a = alpha
    current_turn = history.current_turn
    previous_reward = history.reward_matrix[current_turn-1]
    current_reward = np.empty_like(previous_reward)
    current_reward[:] = previous_reward
    previous_invest = history.invest_matrix[current_turn-1] 
    m_invest = previous_invest[:,num]
    #total_invest = m_invest.sum()
    avg_invest = m_invest.mean()
# For each male, reward is a function of investment
    current_reward[:,num] = (previous_invest[:,num]) ** a
    #current_reward = f_normalize(current_reward,resources,num)
    return current_reward

# Strategy to reward more investment: 5
def f_relative_investment(resources, history, num):
    current_turn = history.current_turn
    previous_reward = history.reward_matrix[current_turn-1]
    current_reward = np.empty_like(previous_reward)
    current_reward[:] = previous_reward
    previous_invest = history.invest_matrix[current_turn-1] 
    m_invest = previous_invest[:,num]
    #total_invest = m_invest.sum()
    avg_invest = m_invest.mean()
# For each male, reward is a function of investment
    for m in range(history.n_males):
        current_reward[m,num] = previous_invest[m,num] / previous_invest[:,num].sum()
    current_reward = f_normalize(current_reward,resources,num)
    return current_reward

# Strategy to reward more investment: 5
def f_relative_investment_a(resources, history, num, alpha):
    a = alpha
    current_turn = history.current_turn
    previous_reward = history.reward_matrix[current_turn-1]
    current_reward = np.empty_like(previous_reward)
    current_reward[:] = previous_reward
    previous_invest = history.invest_matrix[current_turn-1] 
    m_invest = previous_invest[:,num]
    #total_invest = m_invest.sum()
    avg_invest = m_invest.mean()
# For each male, reward is a function of investment
    for m in range(history.n_males):
        current_reward[m,num] = (previous_invest[m,num] / previous_invest[:,num].sum()) ** a
    #current_reward = f_normalize(current_reward,resources,num)
    return current_reward

# Stretegy which judges quality & adjacency
def f_adjacent_quality(resources, history, num, alpha, kappa):
    a = alpha
    q = kappa
    current_turn = history.current_turn
    current_reward = np.empty_like(history.reward_matrix[current_turn-1])
    adjacency = history.adjacency_matrix[current_turn]
    m_quality = history.quality_males
    for m in range(history.n_males):
        current_reward[m,num] = (1 + adjacency[m,history.n_males + num])**a * m_quality[m]**q
    current_reward = f_normalize(current_reward,resources,num)
    return current_reward

# Stretegy which judges quality & relative adjacency
def f_adjacent_quality_relative(resources, history, num, alpha, kappa):
    a = alpha
    q = kappa
    current_turn = history.current_turn
    current_reward = np.empty_like(history.reward_matrix[current_turn-1])
    adjacency = history.adjacency_matrix[current_turn-1]
    m_quality = history.quality_males
    for m in range(history.n_males):
        relative_adjacency = adjacency[m,history.n_males + num] / (sum(adjacency[:history.n_males,history.n_males + num]) + .0001)
        current_reward[m,num] = adjacency[m,history.n_males + num]**0 * (relative_adjacency)**a * m_quality[m]**q
    #current_reward = f_normalize(current_reward,resources,num)
    return current_reward
