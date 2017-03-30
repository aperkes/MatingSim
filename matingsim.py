#! /usr/bin/env python

"""
Simulation to analyze mating dynamics of cowbirds (or more generally, monogamy in the absence of parental care) 
written by Ammon Perkes (perkes.ammon@gmail.com) at University of Pennsylvania
2016
""" 

import sys, os
import numpy as np
from  matplotlib import pyplot as plt
#External .py dependencies, contatining strategies and sims
import SimStrategies,SimStats

#Default starting parameters: 
default_strat_m = 0
default_strat_f = 1
default_res_m = 1    # Resource limitation for a male
default_res_f = 1    # Resource limitations for a female (NOTE: This might be unnecessary also....)
default_males = 5    # Number of birds per trial (assumes symetric trials)
default_females = 5    # Number of birds per trial (assumes symetric trials)
default_turns = 100  # Number of turns per trial
default_trials = 10  # Number of trials per simulation

#global parameters: 
global STRAT_M, STRAT_F, RES_M, RES_F, A, K, BIRDS, TURNS, TRIALS
N_MALES = default_males
N_FEMALES = default_females
RES_M = [default_res_m] * N_MALES
RES_F = [default_res_f] * N_FEMALES
TURNS = default_turns
TRIALS = default_trials
STRAT_M = [default_strat_m] * N_MALES
STRAT_F = [default_strat_f] * N_MALES

#Class of male cowbirds: 
#Includes: Resources, investment matrix, reward matrix, functions to alter investment
#Also includes choice function determining investment
class Male_bird(object):
    def __init__(self, num, strategy = 0, resources = 1): #Convention: Males have even strategies, females odd. 
        self.num = num
        self.strategy = strategy
        self.resources = resources
##      Seed self investment, and normalize for resources
#   Functions to adjust and get info. 
### NOTE: This is where males apply strategy, strategies are saved externally
    def respond(self,history):
       new_investment = SimStrategies.choose(self.strategy, self.resources, history, self.num)
       return new_investment 

#Class of female cowbirds: 
#Includes: resources, response matrix, reward matrix
#Also includes choice function determining response
class Female_bird(object):
    def __init__(self, num, strategy = 1, resources = 1):
        self.num = num
        self.strategy = strategy
        self.resources = resources
##      Seed self  investment, NOTE that for females this is which males are investing in them, not vice versa
#   Functions to adjust and get info. 
    def respond(self,history):
### NOTE: This is where strategy is applied
        resources = self.resources
        new_response = SimStrategies.choose(self.strategy,self.resources, history, self.num)
        return new_response
#
# Function to plot the response curve of a male or female bird (which is contained in their class)
def plot_response(bird):
    strategy = bird.strategy
    function = SimStrategies.function(strategy)
    plt.plot(function)
    plt.show()
#NOTE: This is probably wrong, go back and check it. 

#Array type object containing cowbirds, allowing cleverness: 
#Keep history of investment, reward for both males and females

#NOTE: I want to change this so that we have a full matrix of investment, where birds can invest in themselves, in same-sex birds, with negative investment, etc. This is a trivial change as it relates to history, but it will end up being quite a lot of changes down the line. 
class History(object):
    def __init__(self, turns, n_males, n_females):
## Initialize the matrix for the whole sim (save a bunch of time)
        self.invest_matrix = np.zeros([turns,n_males,n_females])
        self.reward_matrix = np.zeros([turns,n_males,n_females])
        self.turns = turns
        self.n_males = n_males
        self.n_females = n_females
        self.current_turn = 0
    def record(self,turn):
        self.current_turn = turn.n
        self.invest_matrix[turn.n] = turn.invest
        self.reward_matrix[turn.n] = turn.reward
    def initialize(self,initial_conditions = None): #set first investment conditions
        self.invest_matrix[0] = np.random.random((self.n_males,self.n_females))
## Initialize the matrix, then normalize either to 1 or to some matrix (i.e. male resources, or some skew)
## Normalizing is a little tricky due to the males first convention, as follows:
        if initial_conditions == None:
            self.invest_matrix[0] = self.invest_matrix[0] / self.invest_matrix[0].sum(1).reshape(self.n_males,1)
        else:
            self.invest_matrix[0] = self.invest_matrix[0] * initial_conditions / self.invest_matrix[0].sum(1).reshape(self.n_males,1)
    def advance(self):
        self.current_turn = self.current_turn + 1
             

#Object containing turn
class Turn(object):
    def __init__(self, n, n_males, n_females, last_turn = None):
## Initialize based on last turn if desired, otherwise start blank
        self.n = n
        if last_turn != None:
            self.invest = last_turn.invest
            self.reward = last_turn.reward
        else:
            self.invest = np.zeros([n_males,n_females])
            self.reward = np.zeros([n_males,n_females])
    def change_invest(self,male,female,amount):
        self.invest[male,female] += amount
    def change_reward(self,male,female,amount):
        self.reward[male,female] += amount
    def set_invest(self,male,female,amount):
        self.invest[male,female] = amount
    def set_reward(self,male,female,amount):
        self.reward[male,female] = amount


# Object containing all the birds. Cute, eh? 
class Aviary(object):
    def __init__(self, n_males = N_MALES, n_females = N_FEMALES, strat_males = STRAT_M, strat_females = STRAT_F, res_males = RES_M, res_females = RES_F):
# Initialize some parameters:
        self.n_males = n_males
        self.n_females = n_females
        self.strat_males = strat_males
        self.strat_females = strat_females
        self.res_males = res_males
        self.res_females = res_females
# Build the male and female lists in the aviary. 
        self.males = [Male_bird(num, strat_males[num], res_males[num]) for num in range(n_males)]
        self.females = [Female_bird(num, strat_females[num], res_females[num]) for num in range(n_females)]
    def respond(self,history):
# Initialize Turn
        turn = Turn(history.current_turn + 1,history.n_males,history.n_females)
# Initialize investment matrix
        invest = np.zeros([self.n_males,self.n_females])
# For each male, set new investment 
        for m in range(self.n_males):
            invest[m] = self.males[m].respond(history)            
# Save investment matrix to the turn
        turn.invest = invest
# As above, but for reward and females
        reward = np.zeros([self.n_males,self.n_females])
        for f in range(self.n_females):
            reward = self.females[f].respond(history)
        turn.reward = reward
        return turn

    def mrespond(self,history):
        invest = np.zeros([self.n_males,self.n_females])
        for m in range(self.n_males):
            invest[m] = self.males[m].respond(history)            
        return invest

    def frespond(self,history):
        reward = np.zeros([self.n_males,self.n_females])
        for f in range(self.n_females):
            reward = self.females[f].respond(history)
        return reward
#Will I need to give funcitons to change the birds, or will they change automatically? I feel like python is so pointer based that it will be hard to alter it...
# Actually, it's very easy, in fact, because it's a pointer, it alters the original instance, not merely the aviary object.
#Function determining reproductve output: 
#This is not technically important for the simulation, but it determines which strategy is best, which is important

def female_success(params):

    return success

def male_success(params):

    return success

def run_trial(turns = TURNS, n_males = N_MALES,n_females = N_FEMALES,strat_males = None, strat_females = None, res_males = None, res_females = None):
## Initialize full record...
## initialize history
    history = History(turns,n_males,n_females)
## give values to strats and res if none are given:
    if strat_males == None: 
        strat_males = [STRAT_M for n in range(n_males)]
    if strat_females == None:
        strat_females = [STRAT_F for n in range(n_females)]
    if res_males == None:
        res_males = [RES_M for n in range(n_males)]
    if res_females == None:
        res_females = [RES_F for n in range(n_females)]
# Build an aviary using the given parameters
    aviary = Aviary(n_males, n_females, strat_males, strat_females, res_males, res_females)
# Initialize initial investment (based on male resources, if specified)
    history.initialize()
# Get first female response:
    history.reward_matrix[0] = aviary.frespond(history)
    history.advance()
# For every turn, calculate response and record it in history.
    for t in range(1,turns-1):
        turn = aviary.respond(history)
        history.record(turn)
    return history
       
def run_simulation(trials = TRIALS, turns = TURNS, n_males = N_MALES, n_females = N_FEMALES, strat_males = None, strat_females = None, res_males = None, res_females = None):
    record = [0 for tr in range(trials)]
    for tr in range(trials):
        history = run_trial(turns, n_males, n_females, strat_males, strat_females, res_males, res_females)
        record[tr] = SimStats.get_stats(history) 
# For tidiness, stats is saved in a seperate file
    return record
    
#Function plotting history and outcome in interesting ways
def show_his_stats(history):
    stats = SimStats.get_stats(history)
    for stat in stats:
# for now, the stats object doesn't function, it's currently just a list of stats
        print stat
        #stat.print_stat()
        #stat.plot_stat()
        raw_input('press enter to continue...')

#Function for plotting a full simulation
def show_record_stats(record):
    r_stats = rec_stats(record)
    for stat in r_stats:
        stat.print_stat()
        stat.plot_stat()
        raw_input('press enter to continue...')

# Mini function to get birds
def get_birds():
    print "Default males: " + str(N_MALES)
    print "Default females " + str(N_FEMALES)
    choice = raw_input("Would you like to change these? [y/n]")
    if choice[0].lower() == 'y':
        try:
            n_males = raw_input('How many males? ')
            n_females = raw_input('How many females? ')
            n_males = int(n_males.strip())
            n_females = int(n_males.strip())
        except:
            print 'bad input, try again:'
            return get_birds()
        return(n_females,n_females)
    else:
        return (N_MALES,N_FEMALES)

## Mini function to get strategy. Strategy will always be an array of ints matching the number of birds.
#  If a single number is given, it fills the list (2 -> [2,2,2,2,2]
#  If it matches teh number of birds, it will just be one to one, otherwise, any leftovers will be filled by default.
def get_strategy(n_males = N_MALES, n_females = N_FEMALES):
    print "Default strategy: 1"
    print "Enter either a single strategy (2) or multiple strategies, seperated by commas (2,3,4,5)"
    print "If there are more birds than strategies, the rest will be filled by default."
    print "If there are more strategies than birds, the last will be cut off."
    print "...."
    print "N Males: " + str(n_males)
    print "What strategies would you like to use?"
# get input: 
    m_strat = raw_input()
# clean it up and put it in the right form.
    m_strat = m_strat.replace(' ','') 
    m_strat = m_strat.split(',')
    m_strat = map(int, m_strat)
# If a single number was given, it fills the list
    if len(m_strat) == 1:
        m_strat = m_strat * n_males
# If too few numbers were given, the rest are filled by default
    elif len(m_strat) < n_males:
        for i in range(len(m_strat),n_males):
            m_strat[i] = 1
# if the right number of numbers is given, well done
    elif len(m_strat) == n_males:
        pass
# IF too many numbers are given, the last ones are chopped off.
    elif len(m_strat) > n_males:
        del m_strat[n_males:]
    print "...."
    print "N Females: " + str(n_females)
    print "What strategies would you like to use?"
# get input: 
    f_strat = raw_input()
# clean it up and put it in the right form.
    f_strat = f_strat.replace(' ','') 
    f_strat = f_strat.split(',')
    f_strat = map(int, f_strat)
    if len(f_strat) == 1:
        f_strat = f_strat * n_females
    elif len(f_strat) < n_females:
        for i in range(len(f_strat),n_females):
            f_strat[i] = 1
    elif len(f_strat) == n_females:
        pass
    elif len(f_strat) > n_females:
        del f_strat[n_females:]
    return m_strat,f_strat

def get_resources(n_males = N_MALES, n_females = N_FEMALES):
    print "This will work just like strategy."
    print "Default males resources: " + str(RES_M)
    print 'What resources would you like the males to have'
    m_res = raw_input()
    m_res = m_res.replace(' ','')
    m_res = m_res.split(',')
    m_res = map(int, m_res)
    if len(m_res) == 1:
        m_res = m_res * n_males
    elif len(m_res) < n_males:
        for i in range(len(m_res), n_males):
            m_res[i] = 1
    elif len(m_res) == n_males:
        pass
    elif len(m_res) > n_males:
        del m_res[n_males:]
    print "...."
    print "N Females: " + str(n_females)
    print 'What resources would you like the females to have?'
# get input: 
    f_res = raw_input()
# clean it up and put it in the right form.
    f_res = f_res.replace(' ','') 
    f_res = f_res.split(',')
    f_res = map(int, f_res)
    if len(f_res) == 1:
        f_res = f_res * n_females
    elif len(f_res) < n_females:
        for i in range(len(f_res),n_females):
            f_res[i] = 1
    elif len(f_res) == n_females:
        pass
    elif len(f_res) > n_females:
        del f_res[n_females:]
    return m_res,f_res

# function to set up a run custon simulations
def build_simulation():
    print "This will help you build a simulation. Enter all values as integers (i.e., 11)"
    turns = raw_input("How many turns per trial would you like? ")
    turns = int(turns.strip())
    trials = raw_input("How many trials in the simulation would you like? ")
    trials = int(trials.strip())
    n_males,n_females = get_birds()
    stat_males,strat_females = get_strategy()
    res_males,res_females = get_res()
    run_simulation(trials, turns, n_males, n_females, strat_males, strat_females, res_males, res_females)
    
# List of Menu options
def print_menu():
    print "[0] - Run Default Simulation"    
    print "[1] - Run Default Single Trial"
    print "[2] - Run Custom Simulation (set parameters)"
    print "[9] - Quit"

#Menu allowing you to set the parameters (based on sys.argv)
def menu():
    while choice != 9:
        print_menu()
        choice = raw_input("Select option (enter a number)")
        if choice == 0:
            run_simulation()
        elif choice == 1:
            run_trial()
        elif choice == 2:
            build_simulation()
        elif choice == 9:
            print "How about a nice game of chess?"
    return choice
