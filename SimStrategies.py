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
M_SHIFT = .02
F_SHIFT = .02

## Branching function to select the correct function and return response. 
# In the case of males, it returns a new investment matrix 
# In the case of females, it returns a new reward matrix 

def choose(strategy, resources, history, num):
## Male strategies (by convention, even)
    if strategy == 0:
        return m_evasive(resources, history, num)
    elif strategy == 2:
        return m_greedy(resources, history, num)

## Female strategies (by convention, odd)
    elif strategy == 1:
        return f_high_investors(resources, history, num)
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
# This creates a transformation matrix (really a vector: [1 1 1 total 1]
def f_normalize(current_reward,resources,num):
    # This allows for negative reward, but actually counts that as investment
    total = abs(current_reward).sum(0)[num]
    #total = current_reward.sum(0)[num]
    transform = np.ones(current_reward[0].size)
    transform[num] = total
    current_reward = current_reward * resources / transform
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
    #print 'benefit: ' + str(benefit)
    avg_benefit = benefit.mean()
# Invest more if you're beating your avg rate of return, invest less otherwise
    for f in range(history.n_females):
        if benefit[f] >= avg_benefit:
            #print num
            #print f
            #print M_SHIFT
            #print "Current turn: " + str(history.current_turn)
            #print "Investing by Male: " + str(num) + "in female: " + str(f) + ": " + str(M_SHIFT)
            #print "Current Investment:"
            #print current_invest
            #print current_invest[num,f]
            #current_invest[num,f] = add_output(current_invest[num,f],M_SHIFT,f)
            current_invest[num,f] = current_invest[num,f] + M_SHIFT
        else:
            #print "Divesting by " + str(num) + "in f: " + str(f)
            #print current_invest
            #current_invest[num,f] = subtract_output(current_invest[num,f],M_SHIFT,f)
            current_invest[num,f] = current_invest[num,f] - M_SHIFT
# Normalize output
    current_invest[num] = normalize(current_invest[num],resources)
    #print "normalizing across male " + str(num)
    #print current_invest
    return current_invest


########################
## Female Strategies: ##
########################

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
        if m_invest[m] >= avg_invest:
            current_reward[m,num] = previous_reward[m,num] + F_SHIFT
        else:
            current_reward[m,num] = previous_reward[m,num] - F_SHIFT
    current_reward = f_normalize(current_reward,resources,num)
    return current_reward
