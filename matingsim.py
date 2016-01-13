#! /usr/bin/env python

""" Simulation to analyze mating dynamics of cowbirds (or more generally, monogamy in the absence of parental care) 
written by Ammon Perkes (perkes.ammon@gmail.com) at University of Pennsylvania
2016
""" 

import sys, os
import numpy as np
from  matplotlib as import pyplot as plt

#Default starting parameters: 
default_res_m = 1    # Resource limitation for a male
default_res_f = 1    # Resource limitations for a female (NOTE: This might be unnecessary also....)
default_a = 2
default_k = 6        # Parameter, NOTE: I Might take these out
default_birds = 5    # Number of birds per trial (assumes symetric trials)
default_turns = 100  # Number of turns per trial
default_trials = 10  # Number of trials per simulation

#global parameters: 
global RES_M, RES_F, A, K, BIRDS, TURNS, TRIALS
RES_M = default_res_m
RES_F = default_res_f
A = defaul_a
K = default_k
BIRDS = default_birds
TURNS = default_turns
TRIALS = default_trials

#Class of male cowbirds: 
#Includes: Resources, investment matrix, reward matrix, functions to alter investment
#Also includes choice function determining investment
class Male_bird(object):
    def __init__(self, num, resouces = 1, strategy = 1):
        self.num = num
        self.resources = resources
        self.strategy = strategy
##      Seed self investment, and normalize for resources
        self.investment = np.random.rand(BIRDS)
        self.investment = self.invesetment * resources / sum(self.investment)
        self.reward = np.zeros(BIRDS)
#   Functions to adjust and get info. 
    def change_investment(female,amount):
        self.investment[female] = self.investment[female] + amount  
    def change_reward(female,amount):
        self.reward[female] = self.reward[female] + amount
### NOTE: This is where males apply strategy
    def respond(history):
       investment = self.investment
       reward = self.reward
       new_investment = strategies.males(self.strategy,history)
       self.investment = new_investment 
       return self.investment 
    def total_reward():
        return sum(reward))
    def total_investment():
        return sum(investment))

#Class of female cowbirds: 
#Includes: resources, response matrix, reward matrix
#Also includes choice function determining response
class Female_bird(object):
    def __init__(self, num, resouces = 1, strategy = 1):
        self.num = num
        self.resources = resources
        self.strategy = strategy
##      Seed self  investment, NOTE that for females this is which males are investing in them, not vice versa
        self.investment = np.zeros(BIRDS)
#   Functions to adjust and get info. 
    def change_investment(male,amount):
        self.investment[male] = self.investment[male] + amount
    def change_response(male,amount):
        self.response[male] = self.response[male] + amount  
    def change_product(amount):
        self.product = self.product + amount
    def respond(history):
### NOTE: This is where strategy is applied
        investment = self.investment
        resources = self.resources
        new_response = strategies.females(self.strategy,history)
        self.response = new_response
        return self.response
# Get total production (which is a function of investment I assume) 
    def total_product(investment):
        return sum(self.product)
    def total_investment():
        return sum(investment))
#
# Function to plot the response curve of a male or female bird (which is contained in their class)
def plot_response(bird):
    strategy = bird.strategy
    function = strategies.function(strategy)
    plt.plot(function)
    plt.show()
#NOTE: This is probably wrong, go back and check it. 

#Array type object containing cowbirds, allowing cleverness: 
#Keep history of investment, reward for both males and females
class History(object):
    def __init__(self, turns, males, females):
## Initialize the matrix for the whole sim (save a bunch of time)
        self.invest_matrix = np.zeros([turns,males,females])
        self.reward_matrix = np.zeros([turns,males,females])
    def record(turn):
        self.invest_matrix[turn.n] = turn.invest
        self.reward_matrix[turn.n] = turn.reward

#Object containing turn
class Turn(object):
    def __init__(self, n, males, females, last_turn = None):
## Initialize based on last turn if desired, otherwise start blank
        if last_turn != None:
            self.invest = last_turn.invest
            self.reward = last_turn.reward
        else:
            self.invest = np.zeros([males,females])
            self.reward = np.zeros([males,females])
    def change_invest(male,female,amount):
        self.invest[male,female] += amount
    def change_reward(male,female,amount):
        self.reward[male,female] += amount
    def set_invest(male,female,amount):
        self.invest[male,female] = amount
    def set_reward(male,female,amount):
        self.reward[male,female] = amount

#Function determining reproductve output: 
#This is not technically important for the simulation, but it determines which strategy is best, which is important
def female_success(params):

    return success

def male_success(params):

    return success
#Function plotting history and outcome in interesting ways
def plot_history(history):

def run_simulation(trials = 10,turns = 100,n_males,n_females,strat_males = 1, strat_females = 1, res_males = 1, res_females = 1)
    history = History(turns,n_males,n_females)
#Menu allowing you to set the parameters (based on sys.argv)
def menu():

    return choice 
