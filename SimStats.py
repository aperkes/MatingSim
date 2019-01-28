#! /usr/bin/env python

"""
SimStats
Dependencies for matingsim.py
Stores the statistical fuctions, as well as a function "get_stats" which will receive a history and return all interesting stats. 
written by Ammon Perkes (perkes.ammon@gmail.com) at University of Pennsylvania
2016
"""
import sys, os
import numpy as np
from  matplotlib import pyplot as plt
from matplotlib.widgets import Slider
#import plotly as py
import networkx as nx
#from plotly.graph_objs import *
import pdb


## Function compile stats for single trial, based on functions below
def get_stats(history):
    stats = {}
    #stats['monogamy'] = get_monogamy_final(history)
    stats['fitness_m'],stats['fitness_f'] = get_new_strat_counts(history)
    #stats['deviation'] = get_deviation_final(history)
    return stats
    
## Functions for calculating stats: ##

## Spits out sum of reward (which means reward has to track precisely to fitness)
def get_fitness(history):
    strats_m = history.params.strats_m
    strats_f = history.params.strats_f
    params = history.params
    fitness = np.sum(np.sum(history.reward_matrix,0),1)
    #fitness = np.sum(np.sum(history.adjacency_matrix,0),1)
    male_fitness = fitness[:params.n_males]
    female_fitness = fitness[params.n_males:]
    
    strat_fitness_m = dict(zip(np.unique(strats_m),np.zeros(len(np.unique(strats_m)))))
    strat_fitness_f = dict(zip(np.unique(strats_f),np.zeros(len(np.unique(strats_f)))))
    for m in range(len(strats_m)):
        strat = strats_m[m]
        success = male_fitness[m]
        strat_fitness_m[strat] += success
    for f in range(len(strats_f)):
        strat = strats_f[f]
        success = female_fitness[f]
        strat_fitness_f[strat] += success
    return strat_fitness_m, strat_fitness_f

# Gives you the vectors of new strategies so you can seed your next trial
def get_new_strat_counts(history):
    n_males = history.params.n_males
    n_females = history.params.n_females
    params = history.params
    strat_fitness_m,strat_fitness_f = get_fitness(history)
    strats_m_values = np.array(list(strat_fitness_m.values()))
    strats_f_values = np.array(list(strat_fitness_f.values()))
    
    normed_m = strats_m_values / np.sum(strats_m_values)
    normed_f = strats_f_values / np.sum(strats_f_values)

    new_values_m = np.round(normed_m * n_males)
    new_values_f = np.round(normed_f * n_females)
    
    ## For the tricky edge case(s) where it rounds to a different number than it starts:
    if np.sum(new_values_m) != n_males:
        difference = n_males - np.sum(new_values_m)
        max_loc, = np.where(new_values_m == np.max(new_values_m))
        new_values_m[np.random.choice(max_loc)] += difference
    if np.sum(new_values_f) != n_females:
        difference = n_females - np.sum(new_values_f)
        max_loc, = np.where(new_values_f == np.max(new_values_f))
        new_values_f[np.random.choice(max_loc)] += difference
        
    new_counts_m = dict(zip(strat_fitness_m.keys(),new_values_m))
    new_counts_f = dict(zip(strat_fitness_f.keys(),new_values_f))
    return new_counts_m,new_counts_f

def get_all_degrees(history):
    all_degrees = []
    for a in history.adjacency_matrix:        
        G = nx.from_numpy_matrix(a,parallel_edges=False)
        all_degrees.append(get_degrees(G))
    return all_degrees

def get_degrees_final(history):
    adjacency = history.adjacency_matrix[-1]
    G = nx.from_numpy_matrix(adjacency,parallel_edges=False)
    return get_degrees(G)
    
def get_deviation_final(history):
    final_deviation = []
    adjacency = history.adjacency_matrix[-1]
    G = nx.from_numpy_matrix(adjacency,parallel_edges=False)
    return get_deviation(G)

def get_degrees(G):
    degrees = []
    for n in G:
        degrees.append(G.degree(n))
    return degrees

def get_deviation(G):
    degrees = get_degrees(G)
    deviation = 0
    for d in degrees:
        deviation += abs(d-1)
    return deviation

def get_deviations(history):
    deviations = []
    for a in history.adjacency_matrix:
        G = nx.from_numpy_matrix(a,parallel_edges=False)
        deviations.append(deviation(G))
    return deviations

def final_fornication(history):
    final_fornication = []
    adjacency = history.adjacency_matrix[-1]
    G = nx.from_numpy_matrix(adjacency,parallel_edges=False)
    final_fornication = final_fornication.append(deviation(G))
    return final_fornication

def get_monogamy_final(history):
    pair_bonds = 0
    monos = 0
    pair_thresh = .3
    for m in range(history.n_males):
        for f in range(history.n_females):
            bond_strength = history.adjacency_matrix[-1,m,f+history.n_males]
            if bond_strength > pair_thresh:
                pair_bonds += 1
                suitors = history.adjacency_matrix[-1,:,f+history.n_males]
                if len(suitors[suitors > pair_thresh]) == 1:
                    monos += 1
    monogamy_ratio = monos / (float(pair_bonds) + .0001)
    return monogamy_ratio
