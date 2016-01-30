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

## Branching function to select the correct function and return response. 
# In the case of males, it returns a new investment matrix 
# In the case of females, it returns a new reward matrix 

def choose(strategy, resources, history, num):
## Male strategies (by convention, even)
    if strategy == 0:
        return m_evasive(resources, history, num)

## Female strategies (by convention, odd)
    elif strategy == 1:
        return f_high_investors(resources, history, num)

## Some basic functions used in various strategies. NOTE: due to python's structure, changing output within a function will change the output itself, since it's merely a pointer towards the object itself. With that in mind, all the returns are sort of more for good habit, just remember that you can't make an identity and then change one of the lists (that goes for arrays and lists)
def normalize(output,resources):
    n_output = output * resources / sum(output)
    return n_output

# Function to normalize females, because of indexing they don't work the same way
# This creates a transformation matrix (really a vector: [1 1 1 total 1]
def f_normalize(current_reward,resources,num):
    total = current_reward.sum(0)[num]
    transform = np.ones(current_reward[0].size)
    transform[num] = total
    current_reward = current_reward * resources / transform
    return current_reward

def relocate(output,amount,source,sink):
    output[source] = output[source] - amount
    output[sink] = output[sink] + amount
    return output

def add_output(output,amount,sink):
    output[sink] = output[sink] + amount
    return output

def subtract_output(output,amount,sink):
    output[source] = output[source] - amount
    return output

######################
## Male strategies: ##
######################

# Strategy to avoid crowding: 0
def m_evasive(resources, history, num):    
    current_turn = history.current_turn    
    current_reward = history.reward_matrix[current_turn]
    current_invest = history.invest_matrix[current_turn]
    f_sums = current_invest.sum(0)
    avg_invest = f_sums.mean()
# For each female, add or subtract based on whether it's above or below average
# Since avg_invest doesn't change, this can happen without confounding itself
    for f in f_sums:
        if f >= avg_invest:
            current_invest[num,f] = add_output(current_invest[num,f],M_SHIFT,f)
        else:
            current_invest[num,f] = subtract_output(current_invest[num,f],M_SHIFT,f)
# Normalize output
    current_invest[num] = normalize(current_invest[num],resources)
    return current_invest
     
########################
## Female Strategies: ##
########################

# Strategy to reward more investment: 1
def f_high_investors(resources, history, num):
    current_turn = history.current_turn
    current_invest = history.invest_matrix[current_turn] 
    total_invest = current_invest.sum(0)[num]
    avg_invest = total_invest / history.n_males
# For each male, add or subtract based on if it's above average
    for m in m_sums:
        if m >= avg_invest:
            current_reward[m,num] = current_reward[m,num] + F_SHIFT
        else:
            current_reward[m,num] = current_reward[m,num] - F_SHIFT
    current_reward = f_normalize(current_reward,num)
    return current_reward
