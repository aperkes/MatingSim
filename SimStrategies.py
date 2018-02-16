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
## Male strategies
    if strategy == 'M1':
        return m_new_strategy(resources, history, num) 
    elif strategy == 'M0':
        return m_classic_strategy(resources, history, num)
    
## Female strategies (by convention, odd)
    elif strategy == 'F1':
        return f_new_strategy(resources, history, num)
    elif strategy == 'F0':
        return f_classic_strategy(resources, history, num)
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
    
    previous_invest = history.invest_matrix[current_turn]

    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest[:]
    
    my_previous_reward = history.reward_matrix[current_turn - 1,num,:]
    my_previous_invest = history.invest_matrix[current_turn - 1,num,:]
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
    
    previous_invest = history.invest_matrix[current_turn]
    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest[:]
    
    my_previous_reward = history.reward_matrix[current_turn - 1,num,:]
    my_previous_invest = history.invest_matrix[current_turn - 1,num,:]
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
                

########################
## Female Strategies: ##
########################

def f_new_strategy(resources, history, num):
    shift_scaler = .1
    current_turn = history.current_turn
    
    previous_invest = history.invest_matrix[current_turn]
    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest[:]
    
    my_previous_reward = history.reward_matrix[current_turn - 1,history.n_males + num,:]
    my_previous_invest = history.invest_matrix[current_turn - 1,history.n_males + num,:]
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
    
    previous_invest = history.invest_matrix[current_turn]
    current_invest = np.empty_like(previous_invest)
    current_invest[:] = previous_invest[:]
    
    my_previous_reward = history.reward_matrix[current_turn - 1,history.n_males + num,:]
    my_previous_invest = history.invest_matrix[current_turn - 1,history.n_males + num,:]
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
                
