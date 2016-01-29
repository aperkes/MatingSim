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
from matplotlib import pyplot as pt

## Statistic object which will allow for useful processing of data
class Statistic(object):
    def __init__(self, name, data, plot_type = None, x_axis = None, y_axis = None):
        self.name = name
        self.data = data
        self.plot_type = plot_type
        self.x_axis = x_axis
        self.y_axis = y_axis
    def print_stat():
        print self.name
        print data
    def plot_stat():
        if self.plot_type == 'plot':
            plt.plot(x_axis,y_axis)
        else:
            pass
    def save_stat():
        #Write output somehow...
        pass

def get_stats(history):
    stats = []
    stats.append(final_crowding(history))
    stats.append(hist_crowding(history))
    return stats

def final_crowding(history):
    final_invest = history.invest_matrix[-1]
    final_crowding = crowding(final_invest)
    fin_crowd_stat = Statistic('Final Crowding',final_crowding)
    return fin_crowd_stat

def hist_crowding(history):
    c_history = [0 for n in range(history.turns)]
    for t in range(history.turns):
        c_history[t] = crowding(history.invest_matrix[t])
    his_crowd_stat = Statistic('Crowding History',c_history,plot_type = 'plot', x_axis = range(history.turns, y_axis = c_history)
    return c_history

def crowding(invest):
    fsums = invest.sum(0) ## Take the sum along the vertical axis
    norm_fsums = fsums - 1   #Normalize to 1 (assumes equal resources for females...)
    norm_sum = norm_fsums.sum()     #sum of norms
    n_norm_sum = norm_sum / len(fsums)  # Normalize again for number of females
    return n_norm_sum
