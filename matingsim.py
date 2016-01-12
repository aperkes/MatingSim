#! /usr/bin/env python

""" Simulation to analyze mating dynamics of cowbirds (or more generally, monogamy in the absence of parental care) 
written by Ammon Perkes (perkes.ammon@gmail.com) at University of Pennsylvania
2016
""" 

import sys, os
import numpy as np
from  matplotlib as import pyplot as plt

#Default starting parameters: 
default_res_m = 1
default_res_f = 1
default_a = 1
default_k = 1

#global parameters: 
global RES_M, RES_F, A, K
RES_M = default_res_m
RES_F = default_res_f
A = defaul_a
K = default_k

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
    def invest():
       investment = self.investment
       reward = self.reward
       new_investment = 111  ## <- NOTE: fix this, duh. 
       self.investment = new_investment 
       return self.investment 
    def total_reward():
        return sum(reward))
    def total_investment():
        return sum(investment))
#Class of female cowbirds: 
#Includes: resources, response matrix, reward matrix
#Also includes choice function determining response
class female_bird(self):
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
    def respond():
### NOTE: This is where strategy is applied
        investment = self.investment
        resources = self.resources
        self.response = newresponse
        return self.response
# Get total production (which is a function of investment I assume) 
    def total_product(investment):
        return sum(self.product)
    def total_investment():
        return sum(investment))
#
# Function to plot the response curve of a male or female bird (which is contained in their class)
def plot_response(bird):

#Array type object containing cowbirds, allowing cleverness: 
#Keep history of investment, reward for both males and females
class history(self):

#Function determining reproductve output: 
def female_success(params):

    return success

#Function plotting history and outcome in interesting ways
def plot_history(history):


#Menu allowing you to set the parameters (based on sys.argv)
def menu():

    return choice 
