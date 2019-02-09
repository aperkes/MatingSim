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
import pdb

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
    #pdb.set_trace()
    if strategy == 2:
        return m_new_strategy(resources, history, num) 
    elif strategy == 0:
        return m_classic_strategy(resources, history, num)
    elif strategy == 4:
        return m_optimal_strategy(resources, history, num)
    elif strategy == 6:
        return m_single_strategy(resources, history, num)    
## Female strategies (by convention, odd)
    elif strategy == 3:
        return f_flight_strategy(resources, history, num)
        #return f_new_strategy(resources, history, num)
    elif strategy == 1:
        return f_classic_strategy(resources, history, num)
    elif strategy == 5:
        return f_optimal_strategy(resources, history, num)
    elif strategy == 7:
        return f_nash_strategy(resources, history, num)
    elif strategy == 9:
        return f_single_strategy(resources, history, num)
    else:   
        print("No strategy found, quitting")
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

def binary_delta(input_array):
    output_array = np.empty_like(input_array)
    output_array[input_array > 0] = 1
    output_array[input_array < 0] = -1
    output_array[input_array == 0] = 0
    return output_array
    
######################
## Male strategies: ##
######################

## Strategy which enforces each male to only sing to one female at a time: 
def m_single_strategy(resources,history, num):
    bird_index = num
    resources = history.params.ress_m[num]
    n_birds = history.n_males + history.n_females

    #NOTE: Because these are pointers, any change to future_invest will change the invest matrix simultaneously.
#   This could be fixed as below:
#   previous_invest = np.array(history.invest_matrix[history.current_turn - 2]) 
    previous_invest = history.invest_matrix[history.current_turn - 2]
    current_invest = history.invest_matrix[history.current_turn - 1]
    future_invest = history.invest_matrix[history.current_turn]
    #NOTE: Unhack this: 
    previous_reward = history.reward_matrix[history.current_turn - 2] - previous_invest * .1
    current_reward = history.reward_matrix[history.current_turn - 1] - current_invest * .1
    
    previous_adjacency = history.adjacency_matrix[history.current_turn - 2]
    current_adjacency = history.adjacency_matrix[history.current_turn - 1]

    previous_rel_adj = previous_adjacency[bird_index,history.n_males:] / (np.sum(previous_adjacency[:history.n_males,history.n_males:],1) + .0001)
    current_rel_adj = current_adjacency[bird_index,history.n_males:] / (np.sum(current_adjacency[:history.n_males,history.n_males:],1) + .0001)
    delta_inv = current_invest[bird_index,:] - previous_invest[bird_index,:]
    delta_rew = current_reward[bird_index,:] - previous_reward[bird_index,:]
    delta_adj = current_adjacency[bird_index,:] - previous_adjacency[bird_index,:]
    delta_rel_adj = current_rel_adj - previous_rel_adj

    delta_inv = binary_delta(delta_inv)
    delta_rew = binary_delta(delta_rew)
    delta_rel_adj = binary_delta(delta_rel_adj)
    
## If all adjacency is 0, just pick someone at random
    #pdb.set_trace()
    if sum(current_rel_adj) == 0: 
        f_index = np.random.randint(history.n_females) + history.n_males
        future_invest[num,f_index] = 1
        return future_invest
## Otherwise, choose wisely:         
    else: 
        priority = np.array(current_rel_adj)
        for f in range(history.n_females):
            f_max = np.argmax(priority)
            f_index = f_max + history.n_males
            if delta_rel_adj[f_max] > 0:
                future_invest[num,f_index] = previous_invest[num,f_index]
            elif delta_rel_adj[f_max] < 0:
                future_invest[num,f_index] = abs(previous_invest[num,f_index] - 1)
            elif delta_rel_adj[f_max] == 0:
                if np.random.random() > .5:
                    future_invest[num,f_index] = abs(previous_invest[num,f_index] - 1)
                else:
                    future_invest[num,f_index] = previous_invest[num,f_index]
            if future_invest[num,f_index] > 0:
                return future_invest
            else:
                priority[f_max] = -1
    return future_invest
     


## Strategy to optimize through random walk of sorts
def m_optimal_strategy(resources, history, num):
    bird_index = num
    resources = history.params.ress_m[num]
    n_birds = history.n_males + history.n_females

    #NOTE: Because these are pointers, any change to future_invest will change the invest matrix simultaneously.
    previous_invest = history.invest_matrix[history.current_turn - 2]
    current_invest = history.invest_matrix[history.current_turn - 1]
    future_invest = history.invest_matrix[history.current_turn]
     
    #NOTE: Unhack this: 
    previous_reward = history.reward_matrix[history.current_turn - 2] - previous_invest * .1
    current_reward = history.reward_matrix[history.current_turn - 1] - current_invest * .1
    
    previous_adjacency = history.adjacency_matrix[history.current_turn - 2]
    current_adjacency = history.adjacency_matrix[history.current_turn - 1]

    delta_inv = current_invest[bird_index,:] - previous_invest[bird_index,:]
    delta_rew = current_reward[bird_index,:] - previous_reward[bird_index,:]
    delta_adj = current_adjacency[bird_index,:] - previous_adjacency[bird_index,:]

## Try using inv/rew just to get the sign right, and then base the step on...reward
## This turns out to be a pretty key feature, it gets rid of a lot of feedback issues from multiplying delta_inv by delta_reward
    delta_inv[delta_inv > 0] = 1
    delta_inv[delta_inv < 0] = -1
    delta_rew[delta_rew > 0] = 1
    delta_rew[delta_rew < 0] = -1

    if max(np.abs(delta_inv)) == 0: # and np.random.randint(10) > 5:
        random_shift = (.5 - np.random.rand(n_birds)) * (1 - current_reward[bird_index,:]) # << This decides how much random wiggle there should be.
        #pdb.set_trace()
    else:
        random_shift = 0
    if np.random.randint(10) > 8:
        random_encounter = np.random.rand(n_birds) * .1
    else:
        random_encounter = 0
    future_invest[bird_index,:] = current_invest[bird_index,:] + delta_inv * delta_rew * (1 - current_reward[bird_index,:]) * .1 + random_shift + random_encounter

    future_invest[future_invest < 0] = 0

## Set same-sex investment to 0
    if True:
        future_invest[bird_index,:history.n_males] = 0 
## Deal with resource consumption...
    if sum(future_invest[bird_index,:]) > resources:
        #print('We have to scale it somehow...')
        future_invest[bird_index,:] = future_invest[bird_index,:] / np.sum(future_invest[bird_index,:])

    if num == 0:
        pass
        #pdb.set_trace()
    return future_invest


# Inputs: Resources, history, num of male
# Output: A New investment matrix, edited for that one male. 

def m_new_strategy(resources, history, num):
    shift_scaler = .3
    
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

## Strategy for when males can only invest in one female:
def f_single_strategy(resources, history, num):
    iMax = .2 ## Investment of maximal reward, this should actually probably be dynamic somehow...
    bird_index = history.n_males + num
    
    n_birds = history.n_males + history.n_females
    resources = history.params.ress_f[num]

    current_invest = history.invest_matrix[history.current_turn]
    future_invest = np.array(current_invest)

    current_reward = history.reward_matrix[history.current_turn - 1]
    current_adjacency = history.adjacency_matrix[history.current_turn -1]
    
    if history.current_turn >= 20:
        mean_invest = np.mean(history.invest_matrix[history.current_turn-9:history.current_turn+1],0)
    else:
        mean_invest = np.mean(history.invest_matrix[:history.current_turn + 1],0)
    for m in range(history.n_males):
        #pdb.set_trace()
        if m != num:
            future_invest[bird_index,m] = current_invest[m,bird_index] - (0.06 - mean_invest[m,bird_index]) # let them get close enough to learn to stay at a distance. 
            #future_invest[bird_index,m] = current_invest[m,bird_index] #just run from everyone else
        elif m == num:
            future_invest[bird_index,m] = (mean_invest[m,bird_index] - iMax) * .5
    if True:
        future_invest[bird_index,history.n_males:] = 0 
        pass
    if sum(future_invest[bird_index,:]) > resources:
        future_invest[bird_index,:] = future_invest[bird_index,:] / np.sum(future_invest[bird_index,:])
    return future_invest

## Strategy to optimize through random walk of sorts
def f_nash_strategy(resources, history, num):
    bird_index = history.n_males + num
    
    n_birds = history.n_males + history.n_females
    resources = history.params.ress_f[num]

    current_invest = history.invest_matrix[history.current_turn]
    future_invest = np.array(current_invest)
     
    for m in range(history.n_males):
        #pdb.set_trace()
        if m != num:
            future_invest[bird_index,m] = current_invest[m,bird_index]
        elif m == num:
            future_invest[bird_index,m] = 0
    if True:
        future_invest[bird_index,history.n_males:] = 0 
        pass
    if sum(future_invest[bird_index,:]) > resources:
        future_invest[bird_index,:] = future_invest[bird_index,:] / np.sum(future_invest[bird_index,:])
    return future_invest

def f_optimal_strategy(resources, history, num):
    bird_index = history.n_males + num
    resources = history.params.ress_f[num]
    n_birds = history.n_males + history.n_females

    #NOTE: Because these are pointers, any change to future_invest will change the invest matrix simultaneously.
    previous_invest = history.invest_matrix[history.current_turn - 2]
    current_invest = history.invest_matrix[history.current_turn - 1]
    future_invest = history.invest_matrix[history.current_turn]
     
    #NOTE: Unhack this: 
    previous_reward = history.reward_matrix[history.current_turn - 2] - previous_invest * .1
    current_reward = history.reward_matrix[history.current_turn - 1] - current_invest * .1
    
    previous_adjacency = history.adjacency_matrix[history.current_turn - 2]
    current_adjacency = history.adjacency_matrix[history.current_turn - 1]

    delta_inv = current_invest[bird_index,:] - previous_invest[bird_index,:]
    delta_rew = current_reward[bird_index,:] - previous_reward[bird_index,:]
    delta_adj = current_adjacency[bird_index,:] - previous_adjacency[bird_index,:]

## Try using inv/rew just to get the sign right, and then base the step on...reward
## This turns out to be a pretty key feature, it gets rid of a lot of feedback issues from multiplying delta_inv by delta_reward
    delta_inv[delta_inv > 0] = 1
    delta_inv[delta_inv < 0] = -1
    delta_rew[delta_rew > 0] = 1
    delta_rew[delta_rew < 0] = -1

    if max(np.abs(delta_inv)) == 0: # and np.random.randint(10) > 5:
        random_shift = (.5 - np.random.rand(n_birds)) * (1 - current_reward[bird_index,:]) # << This decides how much random wiggle there should be.
        #pdb.set_trace()
    else:
        random_shift = 0
    random_shift = 0
    future_invest[bird_index,:] = current_invest[bird_index,:] + delta_inv * delta_rew * (1 - current_reward[bird_index,:]) * .1 + random_shift 

    future_invest[future_invest < 0] = 0

## Set same-sex investment to 0
    if True:
        future_invest[bird_index,history.n_males:] = 0 
## Deal with resource consumption...
    if sum(future_invest[bird_index,:]) > resources:
        #print('We have to scale it somehow...')
        future_invest[bird_index,:] = future_invest[bird_index,:] / np.sum(future_invest[bird_index,:])
    return future_invest

## Same as classic, but female invests more in poor investors (given the new paradigm of adjacency=male - female investment
def f_flight_strategy(resources, history, num):
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
    shift_matrix[profit > avg_profit] = -shift_scaler
    shift_matrix[profit < avg_profit] = shift_scaler

    my_current_invest = my_previous_invest + shift_matrix
    my_current_invest[my_current_invest < 0] = 0
    my_current_invest[history.n_males:] = 0
    
    my_current_invest = my_current_invest / (sum(abs(my_current_invest)) +.0001) * resources 
    current_invest[history.n_males + num,:] = my_current_invest    
    return current_invest

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
                
