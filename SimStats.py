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
#from plotly.graph_objs import *

## Function compile stats for single trial, based on functions below
def get_stats(history):
    stats = {}
    stats['monogamy'] = get_monogamy_final(history)
    #stats['deviation'] = get_deviation_final(history)
    return stats
    
## Functions for calculating stats: ##
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
    if pair_bonds != 0:
        monogamy_ratio = monos / float(pair_bonds)
        return monogamy_ratio
    else:
        monogamy_ratio = None
        return monogamy_ratio
