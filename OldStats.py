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
import plotly as py
import networkx as nx
from plotly.graph_objs import *

## Statistic object which will allow for useful processing of data
class Statistic(object):
    def __init__(self, name, data, plot_type = None, x_axis = None, y_axis = None):
        self.name = name
        self.data = data
        self.plot_type = plot_type
        self.x_axis = x_axis
        self.y_axis = y_axis
    def print_stat(self):
        print self.name
        print self.data
    def plot_stat(self):
        if self.plot_type == 'plot':
            plt.plot(x_axis,y_axis)
        else:
            pass
    def save_stat(self):
        #Write output somehow...
        pass

## function compiling stats for a whole simulation, based on functions
def rec_stats(record):
    stats = []
    stats.append(rec_crowding(record))

## Function compile stats for single trial, based on functions below
def get_stats(history):
    stats = {}

    #stats['waste'] = final_waste(history)
    stats['deviation'] = get_deviation_final(history)
    #stats['crowding'] = final_crowding(history)
    #stats['monogamy'] = get_monogamy_ratio(history)
    return stats

### Record Stats: ###
# These are a little tricky, since record comes as a list of stats...
def rec_crowding(record):
    for h_stats in record:
        for stat in h_stats:
            if stat.name == 'crowding':
                crowd_list.append(stat.data)
                break
            else:
                pass
    avg_crowd = crowd_list.mean()
    std_crowd = crowd_list.std()
    rec_crowd_stat = Statistic('Crowding: (avg & std)',[avg_crowd, std_crowd])
     
### History Stats: ###
def final_crowding(history):
    final_invest = history.invest_matrix[-1]
    final_crowding = crowding(final_invest)
    fin_crowd_stat = Statistic('Final Crowding',final_crowding)
    return fin_crowd_stat

def hist_crowding(history):
    c_history = np.zeros(history.n_turns)
    #c_history = [0 for n in range(history.n_turns)]
    for t in range(history.n_turns):
        c_history[t] = crowding(history.invest_matrix[t])
    his_crowd_stat = Statistic('Crowding History',c_history,plot_type = 'plot', x_axis = range(history.n_turns), y_axis = c_history)
    return c_history
    
## Functions for calculating stats: ##
def crowding(invest):
    fsums = invest.sum(0) ## Take the sum along the vertical axis
    norm_fsums = fsums - 1   #Normalize to 1 (assumes equal resources for females...)
    norm_sum = norm_fsums.sum()     #sum of norms
    n_norm_sum = norm_sum / len(fsums)  # Normalize again for number of females
    return n_norm_sum

def get_waste(history):
    waste = []
    n_females = history.n_females
    for r in history.reward_matrix:
        lost_reward = n_females - sum(sum(r))
        waste.append(lost_reward)
    #plt.plot(waste)
    #plt.show()
    return waste

def final_waste(history):
    n_females = history.n_females
    lost_reward = n_females - sum(sum(history.reward_matrix[-1]))
    return lost_reward

def get_all_degrees(history):
    all_degrees = []
    for i in history.invest_matrix():
        adjacency = get_adjacency(history.invest_matrix[-1])
        adjacency.dtype = [('weight','float')]
        G = nx.from_numpy_matrix(adjacency,parallel_edges=False)
        all_degrees.append(get_degrees(G))
    return all_degrees

def get_degrees_final(history):
    adjacency = get_adjacency(history.invest_matrix[-1])
    adjacency.dtype = [('weight','float')]
    G = nx.from_numpy_matrix(adjacency,parallel_edges=False)
    return final_degrees
    
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
    for i in history.invest_matrix():
        adjacency = get_adjacency(i)
        adjacency.dtype = [('weight','float')]
        G = nx.from_numpy_matrix(adjacency,parallel_edges=False)
        deviations.append(deviation(G))
    return deviations

def final_fornication(history):
    final_fornication = 0
    adjacency = get_adjacency(history.invest_matrix[-1])
    adjacency.dtype = [('weight','float')]
    G = nx.from_numpy_matrix(adjacency,parallel_edges=False)
    final_fornication = fornication.append(deviation(G))
    return final_fornication
    
def get_adjacency(investment):
    n_males,n_females = np.shape(investment)
    size = n_males + n_females
    adjacency = np.zeros([size,size])
    adjacency[0:n_males,n_males:] = investment
    adjacency[n_males:,:n_females] = np.transpose(investment)
    return adjacency

def count_pairs(investment):
    monogamy_thresh = .3
    round_add = .5 - monogamy_thresh
    pairs = np.count_nonzero(np.round(history.invest_matrix[-1] + round_add))
    return pairs

def get_monogamy_ratio(history):
    monogamy_thresh = .3
    investment = history.invest_matrix[-1]
    mono_pairs,poly_pairs = (0,0)
    for m in range(history.n_males):
        for f in range(history.n_females):
            pair_strength = investment[m,f]
            if pair_strength > monogamy_thresh:
                if (sum(investment[m]) == pair_strength) and (sum(investment[:,f]) == pair_strength):
                    mono_pairs += 1
                else:
                    poly_pairs += 1
    monogamy_ratio = mono_pairs / float(mono_pairs + poly_pairs)
    return monogamy_ratio