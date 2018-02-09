#! /usr/bin/env python

"""
Simulation to analyze mating dynamics of cowbirds (or more generally, monogamy in the absence of parental care) 
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

#External .py dependencies, contatining strategies and sims
import SimStrategies,SimStats

#Default starting parameters: 
default_strat_m = 4
default_strat_f = 7
default_res_m = 1    # Resource limitation for a male
default_res_f = 1    # Resource limitations for a female (NOTE: This might be unnecessary also....)
default_males = 6    # Number of males per trial 
default_females = 6 # Number of females per trial 
default_q_males = 1.0
default_q_females = 1.0
default_turns = 6000  # Number of turns per trial
default_trials = 10  # Number of trials per simulation
default_alpha = 2.0
default_slippage = .25 # Amount of adjacency lost per turn
default_min_adjacency = .01

#global parameters: 
global STRAT_M, STRAT_F, RES_M, RES_F, A, K, BIRDS, TURNS, TRIALS, ALPHA, ALPHAS, SLIP, MIN_ADJ
N_MALES = default_males
N_FEMALES = default_females
RES_M = default_res_m
RES_F = default_res_f
RESS_M = [default_res_m] * N_MALES
RESS_F = [default_res_f] * N_FEMALES
Q_MALES = [default_q_males] * default_males
Q_FEMALES = [default_q_females] * default_females
TURNS = default_turns
TRIALS = default_trials
STRAT_M = default_strat_m
STRAT_F = default_strat_f
STRATS_M = [default_strat_m] * N_MALES
STRATS_F = [default_strat_f] * N_MALES
ALPHA = default_alpha
ALPHAS = [default_alpha] * N_FEMALES
SLIP = default_slippage
MIN_ADJ = default_min_adjacency

## Functions that define how matrices covert, these are two of the central mechanisms of the simulation
def adjacency_to_reward(history,turn):
    ## This provides males and females with a reward function on which to base decisions
    # Noteably, this is different from the original implementation, in which the reward *is* the female response
    # Because it needs to update immediately (and really everything should...) it needs the current turn also
    
    alpha = ALPHA # This should be a different parameter. Not everything is alpha
    omega = 0.5 ## This defines the maximum cost of crowding
    tau = 2.0  ## This defines how quickly crowding cost decreases
    kappa = 0.0 ## This defines base adjacency cost
    rho = 0.0 ## This defines risk from individual (also a cost)
    n_birds = history.n_males + history.n_females
    
    reward = np.zeros(np.shape(history.reward_matrix2[0]))

    #current_adjacency = history.adjacency_matrix[history.current_turn]   # This way it gets the most recent adjacency
    current_adjacency = turn.adjacency # This way it gets the most recent adjacency

    base_adjacency_cost = kappa
    individual_risk = rho
    
    # Presumably, costs/benefits should be different for males and females                  
    for i in range(n_birds):
        for j in range(n_birds):
            crowding_cost = omega / tau ** (n_birds - current_adjacency[:,j].sum())
            adjacency_cost = current_adjacency[i,j] * base_adjacency_cost
            reward[i,j] = ((current_adjacency[i,j] / current_adjacency[:,j].sum()) ** alpha * history.quality_vector[j] 
                               - crowding_cost - base_adjacency_cost - individual_risk)
        reward[i,i] = 0
    
    #If desired, set reward for same-sex pairs to 0
    #reward[0:n_males,0:n_males] = 0
    #reward[n_males:,n_males:] = 0
      
    return reward

# Translate investment into adjacency
def investment_to_adjacency(history):
    # Here, investment cuts a fraction of the distance, using new investment matrix
    
    iota = 1.0 ## This is a parameter that determines how investment scales
    phi = 0 ## This is a parameter that determines how investment in fleeing scales (get it? phleeing)
    shift_constant = .01 ## This scales down investment to make everything more gradual
    slip_constant = .001
    previous_adjacency = history.adjacency_matrix[history.current_turn - 1]
    
    ### The following line adds slippage, this is important to avoid crowding when no one is fleeing.
    previous_adjacency = previous_adjacency - previous_adjacency * slip_constant 
    
    previous_investment = history.invest_matrix2[history.current_turn - 1]

    shift_matrix = previous_investment ** iota
    shift_matrix[previous_investment < 0] = - (phi + (1 - phi) * previous_investment[previous_investment < 0] ** iota) 
    current_adjacency = previous_adjacency + (1 - previous_adjacency) * shift_matrix * shift_constant 
    return current_adjacency


#Class of male cowbirds: 
#Includes: Resources, investment matrix, reward matrix, functions to alter investment
#Also includes choice function determining investment
# Object containing all the birds. Cute, eh? 
class Aviary(object):
    def __init__(self, n_males = N_MALES, n_females = N_FEMALES, 
                 strats_males = STRATS_M, strats_females = STRATS_F, 
                 ress_males = RESS_M, ress_females = RESS_F, alphas = ALPHAS, q_males = Q_MALES, q_females = Q_FEMALES):
# Initialize some parameters:
        self.n_males = n_males
        self.n_females = n_females
        self.strats_males = strats_males
        self.strats_females = strats_females
        self.ress_males = ress_males
        self.ress_females = ress_females
        self.q_males = Q_MALES
        self.q_females = Q_FEMALES
# Build the male and female lists in the aviary. 
        self.males = [Male_bird(num, strats_males[num], ress_males[num], q_males[num]) for num in range(n_males)]
        self.females = [Female_bird(num, strats_females[num], ress_females[num], alphas[num], q_females[num]) for num in range(n_females)]

        ## This is a big important function. It is how the aviary of birds responds every turn
    def respond(self,history):
        # Initialize Turn
        turn = Turn(history.current_turn,history.n_males,history.n_females)
            
        # Save matrices to the turn
        turn.invest = self.mrespond(history)
        ## mrespond & frespond also 
        turn.reward = self.frespond(history)
        turn.adjacency = self.update_adjacency2(history, turn)
        turn.invest2 = self.update_invest2(history, turn)
        turn.reward2 = self.update_reward2(history, turn)
        return turn

    def mrespond(self,history):
        invest = np.zeros([self.n_males,self.n_females])
        for m in range(self.n_males):
            invest[m] = self.males[m].respond(history)[m]
        return invest

    def frespond(self,history):
        reward = np.zeros([self.n_males,self.n_females])
        for f in range(self.n_females):
            reward[:,f] = self.females[f].respond(history)[:,f]   
        return reward
    
    def update_adjacency(self,history, turn):
        ## Here, investment adds to adjacency, with a hard cap at 1, and optional slippage
        slippage = SLIP
        min_adjacency = MIN_ADJ
        previous_adjacency = history.adjacency_matrix[history.current_turn - 1]
        previous_investment = history.invest_matrix[history.current_turn - 1]
        investment_adj = get_adjacency(previous_investment)
        current_adjacency = previous_adjacency + investment_adj
        
        current_adjacency = current_adjacency - slippage
        #current_adjacency = current_adjacency * (investment_adj + 1) #update based on investment
        
        current_adjacency[current_adjacency <= min_adjacency] = min_adjacency #limit max distance
        current_adjacency[current_adjacency > 1.0] = 1.0
        return current_adjacency
    
    def update_adjacency1(self,history, turn):
        # Here investment cuts a fraction of the distance using old investment matrix
        # This means full inveestment will always get you to 1, but it also means you get diminishing returns
        alpha = ALPHA ## This is how investment scales up or down
        iota = 1.5 ## This is a parameter that determines how investment scales
        phi = 0 ## This is a parameter that determines how ivnestment in fleeing scales (get it? phleeing)
        previous_adjacency = history.adjacency_matrix[history.current_turn - 1]
        
        ### The following line adds slippage, this is important to avoid crowding when no one is fleeing.
        previous_adjacency = previous_adjacency - previous_adjacency * .01 
        previous_investment = history.invest_matrix[history.current_turn - 1]
        investment_adj = get_adjacency(previous_investment)
        shift_matrix = investment_adj ** alpha
        shift_matrix[investment_adj < 0] = - (phi + (1 - phi) * investment_adj[investment_adj < 0] ** iota) 
        current_adjacency = previous_adjacency + (1 - previous_adjacency) * shift_matrix
        return current_adjacency
        
    def update_adjacency2(self,history,turn):
        current_adjacency = investment_to_adjacency(history)
        return current_adjacency
    
    def update_invest2(self,history,turn):
        ## This provides both males and females an opportunity to return investment
        previous_invest = history.invest_matrix2[turn.n - 1]
        male_invest = np.zeros(np.shape(previous_invest))
        female_invest = np.zeros(np.shape(previous_invest))
        for m in range(history.n_males):
            male_invest[m] = self.males[m].respond2(history)[m]
        for f in range(history.n_females):
            female_invest[f + history.n_males] = self.females[f].respond2(history)[f + history.n_males]
        new_invest = male_invest + female_invest
        return new_invest
    
    def update_reward2(self,history,turn):
        new_reward2 = adjacency_to_reward(history,turn)
        return new_reward2
        
        
class Male_bird(object):
    def __init__(self, num, strategy = 0, resources = 1, quality = 1.0): #Convention: Males have even strategies, females odd. 
        self.num = num
        self.strategy = strategy
        self.resources = resources
        self.quality = quality
##      Seed self investment, and normalize for resources
#   Functions to adjust and get info. 
### NOTE: This is where males apply strategy, strategies are saved externally
    def respond(self,history):
        new_investment = SimStrategies.choose(self.strategy, self.resources, history, self.num)
        return new_investment 
    
    def respond2(self,history):
        new_investment = SimStrategies.choose('M1', self.resources, history, self.num)
        return new_investment

#Class of female cowbirds: 
#Includes: resources, response matrix, reward matrix
#Also includes choice function determining response
class Female_bird(object):
    def __init__(self, num, strategy = 1, resources = 1, alpha = ALPHA, quality = 1.0):
        self.num = num
        self.strategy = strategy
        self.resources = resources
        self.alpha = alpha
        self.quality = quality
##      Seed self  investment, NOTE that for females this is which males are investing in them, not vice versa
#   Functions to adjust and get info. 
    def respond(self,history):
### NOTE: This is where strategy is applied
        resources = self.resources
        new_response = SimStrategies.choose(self.strategy,self.resources, history, self.num, self.alpha)
        return new_response 
    
    def respond2(self,history):
        new_investment = SimStrategies.choose('F1', self.resources, history, self.num)
        return new_investment

#Array type object containing cowbirds, allowing cleverness: 
#Keep history of investment, reward for both males and females

#NOTE: I want to change this so that we have a full matrix of investment, where birds can invest in themselves, in same-sex birds, with negative investment, etc. This is a trivial change as it relates to history, but it will end up being quite a lot of changes down the line. 
class History(object):
    def __init__(self, n_turns = TURNS, n_males = N_MALES, n_females = N_FEMALES, q_males = Q_MALES, q_females = Q_FEMALES):
## Initialize the matrix for the whole sim (save a bunch of time)
        self.invest_matrix = np.zeros([n_turns,n_males,n_females])
        self.reward_matrix = np.zeros([n_turns,n_males,n_females])
        self.adjacency_matrix = np.zeros([n_turns,n_males + n_females, n_males + n_females])
        self.adjacency_matrix[0,:n_males,n_males:] = np.random.rand(n_males,n_females)
        self.invest_matrix2 = np.zeros(np.shape(self.adjacency_matrix))
        self.reward_matrix2 = np.zeros(np.shape(self.adjacency_matrix))       
        self.n_turns = n_turns
        self.n_males = n_males
        self.n_females = n_females
        self.current_turn = 0
        self.quality_males = q_males
        self.quality_females = q_females
        self.quality_vector = np.append(q_males,q_females)

    def record(self,turn): 
        #self.current_turn = turn.n
        self.invest_matrix[turn.n] = turn.invest
        self.reward_matrix[turn.n] = turn.reward
        self.adjacency_matrix[turn.n] = turn.adjacency
        self.invest_matrix2[turn.n] = turn.invest2
        self.reward_matrix2[turn.n] = turn.reward2
        self.advance()
        
    def initialize(self,initial_conditions = None): #set first investment conditions
        self.invest_matrix[0] = np.random.random((self.n_males,self.n_females))
        self.invest_matrix[0] = self.invest_matrix[0] / self.invest_matrix[0].sum(1) 

        self.reward_matrix[0] = np.random.random([self.n_males,self.n_females])
        self.reward_matrix[0] = self.reward_matrix[0] / self.reward_matrix[0].sum(0)
        
        self.invest_matrix2[0] = np.random.random([self.n_males + self.n_females, self.n_males + self.n_females]) * (1 - np.identity(self.n_males + self.n_females))
        self.invest_matrix2[0] = self.invest_matrix2[0] / self.invest_matrix2[0].sum(1)
        self.adjacency_matrix[0] = np.random.random(np.shape(self.invest_matrix2[0])) * (1 - np.identity(self.n_males + self.n_females))
        self.reward_matrix2[0] = self.update_reward_hist()
        
        if initial_conditions == None:
            self.invest_matrix[0] = self.invest_matrix[0] / self.invest_matrix[0].sum(1).reshape(self.n_males,1)
        else:
            self.invest_matrix[0] = self.invest_matrix[0] * initial_conditions / self.invest_matrix[0].sum(1).reshape(self.n_males,1)

        self.reward_matrix[-1] = self.reward_matrix[0] # This is for when the f_respond checks for previous
        self.invest_matrix[-1] = self.invest_matrix[0]
        self.invest_matrix2[-1] = self.invest_matrix2[0]
        self.reward_matrix2[-1] = self.reward_matrix2[0]
        self.adjacency_matrix[-1] = self.adjacency_matrix[0]
        
        
    def advance(self):
        self.current_turn = self.current_turn + 1
    
    def update_reward_hist(self):
        ## This provides males and females with a reward function on which to base decisions
        # Noteably, this is different from the original implementation, in which the reward *is* the female response
        # Because it needs to update immediately (and really everything should...) it needs the current turn also
        alpha = ALPHA # This should be a different parameter. Not everything is alpha
        
        new_reward2 = np.empty_like(self.reward_matrix2[0])
        #current_adjacency = history.adjacency_matrix[history.current_turn]
        current_adjacency = self.adjacency_matrix[self.current_turn]   # This way it gets the most recent adjacency
        n_birds = self.n_females + self.n_males
        omega = 0.5 ## This defines the maximum cost of crowding
        tau = 2.0  ## This defines how quickly crowding changes
        base_adjacency_cost = 0.0
        individual_risk = 0.0
        for i in range(n_birds):
            for j in range(n_birds):
                crowding_cost = omega / tau ** (1 - current_adjacency[:,j].sum())
                adjacency_cost = current_adjacency[i,j] * base_adjacency_cost
                new_reward2[i,j] = (current_adjacency[i,j] ** alpha * self.quality_vector[j] 
                                   - crowding_cost - base_adjacency_cost - individual_risk)
            new_reward2[i,i] = 0
        return new_reward2         

#Object containing turn, I don't actually use this at all
class Turn(object):
    def __init__(self, n, n_males, n_females, previous_turn = None):
## Initialize based on last turn if desired, otherwise start blank
        self.n = n
        if previous_turn != None:
            self.invest = previous_turn.invest
            self.reward = previous_turn.reward
            self.adjacency = previous_turn.adjacency
            self.invest2 = previous_turn.invest2
            self.reward2 = previous_turn.reward2
        else:
            self.invest = np.zeros([n_males,n_females])
            self.reward = np.zeros([n_males,n_females])
            self.adjacency = np.zeros([n_males+n_females,n_females+n_males])
            self.invest2 = np.zeros(np.shape(self.adjacency))
            self.reward2 = np.zeros(np.shape(self.adjacency))
    def change_invest(self,male,female,amount):
        self.invest[male,female] += amount
    def change_reward(self,male,female,amount):
        self.reward[male,female] += amount
    def set_invest(self,male,female,amount):
        self.invest[male,female] = amount
    def set_reward(self,male,female,amount):
        self.reward[male,female] = amount

#Will I need to give funcitons to change the birds, or will they change automatically? I feel like python is so pointer based that it will be hard to alter it...
# Actually, it's very easy, in fact, because it's a pointer, it alters the original instance, not merely the aviary object.
#Function determining reproductve output: 
#This is not technically important for the simulation, but it determines which strategy is best, which is important

def female_success(params):

    return success

def male_success(params):

    return success

def run_trial(n_turns = TURNS, n_males = N_MALES,n_females = N_FEMALES,
              strats_males = None, strats_females = None, 
              ress_males = None, ress_females = None, 
              alpha = None, alphas = ALPHAS, initial_conditions = None):

## Initialize full record...
## initialize history
    history = History(n_turns,n_males,n_females)
    history.initialize(initial_conditions)
## give values to strats and res if none are given:
    if strats_males == None: 
        strats_males = [STRAT_M for n in range(n_males)]
    if strats_females == None:
        strats_females = [STRAT_F for n in range(n_females)]
    if ress_males == None:
        ress_males = [RES_M for n in range(n_males)]
    if ress_females == None:
        ress_females = [RES_F for n in range(n_females)]
    if alpha == None:
        pass #alphas is already defined
    else:
        alphas = [alpha] * n_females
# Build an aviary using the given parameters
    aviary = Aviary(n_males, n_females, strats_males, strats_females, ress_males, ress_females, alphas)
# Initialize initial investment (based on male resources, if specified)
    history.initialize()
# Get first female response:
    history.reward_matrix[0] = aviary.frespond(history)
    history.advance()
# For every turn, calculate response and record it in history.
    for t in range(n_turns-1):
        turn = aviary.respond(history)
        history.record(turn)
    return history
       
def run_simulation(trials = TRIALS, n_turns = TURNS, 
                   n_males = N_MALES, n_females = N_FEMALES, 
                   strat_males = None, strat_females = None, 
                   ress_males = None, ress_females = None, alpha = None, alphas = ALPHAS):
    if alpha == None:
        pass
    else:
        alphas = [alpha] * n_females
    record = [0 for tr in range(trials)]
    for tr in range(trials):
        history = run_trial(n_turns, n_males, n_females, 
                            strat_males, strat_females, ress_males, ress_females, alpha, alphas)
        record[tr] = SimStats.get_stats(history) 
# For tidiness, stats is saved in a seperate file
    return record

# Function to plot the response curve of a male or female bird (which is contained in their class)
def plot_response(bird):
    strategy = bird.strategy
    function = SimStrategies.function(strategy)
    plt.plot(function)
    plt.show()
#NOTE: This is probably wrong, go back and check it.

#Function plotting history and outcome in interesting ways
def show_his_stats(history):
    stats = SimStats.get_stats(history)
    for stat in stats:
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
    n_turns = raw_input("How many turns per trial would you like? ")
    n_turns = int(turns.strip())
    trials = raw_input("How many trials in the simulation would you like? ")
    trials = int(trials.strip())
    n_males,n_females = get_birds()
    stat_males,strat_females = get_strategy()
    res_males,res_females = get_res()
    run_simulation(trials, n_turns, n_males, n_females, strat_males, strat_females, res_males, res_females)

### Plotting Function ###
# Plot history (with a fancy slider)
def plot_history(history):
    # generate a five layer data 
    data = history.invest_matrix
    # current layer index start with the first layer 
    idx = 0

    # figure axis setup 
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.15)

    # display initial image 
    im_h = ax.imshow(data[idx, :, :], vmin=0., vmax=1., cmap='hot', interpolation='nearest')

    # setup a slider axis and the Slider
    ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
    slider_depth = Slider(ax_depth, 'depth', 0, data.shape[0]-1, valinit=idx)

    # update the figure with a change on the slider 
    def update_depth(val):
        idx = int(round(slider_depth.val))
        im_h.set_data(data[idx, :, :])

    slider_depth.on_changed(update_depth)

    plt.show()  
    
def plot_history2(history_matrix):
    # generate a five layer data 
    data = history_matrix
    # current layer index start with the first layer 
    idx = 0

    # figure axis setup 
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.15)

    # display initial image 
    im_h = ax.imshow(data[idx, :, :], vmin=-1.0, vmax=1., cmap='bwr', interpolation='nearest')

    # setup a slider axis and the Slider
    ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
    slider_depth = Slider(ax_depth, 'depth', 0, data.shape[0]-1, valinit=idx)

    # update the figure with a change on the slider 
    def update_depth(val):
        idx = int(round(slider_depth.val))
        im_h.set_data(data[idx, :, :])

    slider_depth.on_changed(update_depth)

    plt.show()
    
### Several Functions for Network Plotting ###
def plot_network_progression_inv(history):
    turns = history.current_turn
    seed_turn = 0
    adj_rounded = get_adj_rounded(history.invest_matrix[seed_turn])

    G = nx.from_numpy_matrix(adj_rounded)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    pos = nx.spring_layout(G)
    #pos = nx.spectral_layout(G)
    #pos = nx.circular_layout(G)
    #pos = nx.shell_layout(G)
    #pos = nx.kamada_kawai_layout(G)
    
    step = 30
    wait_time = .4
    for t in range(0,turns,step):
        plt.cla()
        adjacency = get_adjacency(history.invest_matrix[t]) * 4
        adjacency.dtype = [('weight','float')]
        G = nx.from_numpy_matrix(adjacency,parallel_edges=False)
        pos = nx.spring_layout(G, pos=pos, fixed=None)
        color_map = []
        for node in G:
            if int(node) < history.n_males:
                color_map.append('blue')
            else: color_map.append('red')    
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]
        nx.draw(G, pos, node_color = color_map, with_labels = True, edges=edges, width=weights, node_size=450)
        plt.pause(wait_time)
        plt.draw()

def plot_network_progression(history):
    turns = history.current_turn
    seed_turn = 0
    adj_rounded = np.round(history.adjacency_matrix[seed_turn] + .4) * 4
    G = nx.from_numpy_matrix(adj_rounded)
    #G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    pos = nx.spring_layout(G)
    #pos = nx.spectral_layout(G)
    #pos = nx.circular_layout(G)
    #pos = nx.shell_layout(G)
    #pos = nx.kamada_kawai_layout(G)
    
    step = turns / 20
    wait_time = .5
    for t in range(0,turns,step):
        plt.cla()
        adj_rounded = history.adjacency_matrix[t]
        adj_rounded[adj_rounded < .1] = 0.0
        adj_rounded.dtype = [('weight','float')]
        G = nx.from_numpy_matrix(adj_rounded,parallel_edges=False)
        pos = nx.spring_layout(G, pos=pos, fixed=None)
        color_map = []
        for node in G:
            if int(node) < history.n_males:
                color_map.append('blue')
            else: color_map.append('red')    
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]
        nx.draw(G, pos, node_color = color_map, with_labels = True, edges=edges, width=weights, node_size=450)
        plt.pause(wait_time)
        plt.draw()
        
        
def get_adj_rounded(investment):
    n_males,n_females = np.shape(investment)
    size = n_males + n_females
    adjacency = np.zeros([size,size])
    adjacency[0:n_males,n_males:] = investment
    adjacency[n_males:,:n_females] = np.transpose(investment)
    adj_rounded = np.round(adjacency + .4)
    return adj_rounded

def get_adjacency(investment):
    n_males,n_females = np.shape(investment)
    size = n_males + n_females
    adjacency = np.zeros([size,size])
    adjacency[0:n_males,n_males:] = investment
    adjacency[n_males:,:n_females] = np.transpose(investment)
    return adjacency
    
def plot_network(investment):
    plt.cla()
    adjacency = get_adjacency(investment) * 4
    adjacency.dtype = [('weight','float')]
    G = nx.from_numpy_matrix(adjacency,parallel_edges=False)
    # Set layount: 
    pos = nx.spring_layout(G)
    color_map = []
    n_males, n_females = np.shape(investment)
    for node in G:
        if int(node) < n_males:
            color_map.append('blue')
        else: color_map.append('red')    

    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw(G, pos, node_color = color_map, with_labels = True, edges=edges, width=weights)

    plt.show()
        
# List of Menu options
def print_menu():
    print "[0] - Run Default Simulation"    
    print "[1] - Run Default Single Trial"
    print "[2] - Run Custom Simulation (set parameters)"
    print "[9] - Quit"

#Menu allowing you to set the parameters (based on sys.argv)
def menu():
    choice = 0
    while choice != 9:
        print_menu()
        choice = int(raw_input("Select option (enter a number):  "))
        if choice == 0:
            run_simulation()
        elif choice == 1:
            run_trial()
        elif choice == 2:
            build_simulation()
        elif choice == 9:
            print "How about a nice game of chess?"
    return choice

if __name__ == "__main__":
    #menu()    
    history = run_trial()
    #run_simulation(alphas = [1.5]*N_FEMALES)
    plot_history(history)
    #plot_network_progression(history)
    pass