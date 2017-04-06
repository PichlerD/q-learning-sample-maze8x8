# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 17:08:31 2017

@author: PichlerD
"""
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
import copy
import random
from random import randrange
plt.pause(0.001) #fixes a plt output issue

#Choose Policy functions
alpha = 0.9 #the learning rate
gamma = 0.5 #the discount factor
epsilon = 0.5 #the % of noise in the greedy epsilon policy
printlake = False #Can be True or False, Toggles Graphical Game Output 1st ep.
plotprint = True #Can be True or False, Final Graphical Plot Output
punishwall = False #If True punishes the AI for trying to run outside of maze

#Initialize and set number of steps
num_steps = 100000

#Initialize Variables and empty Q Matrix
x = 0 
y = 0
reward = 0
treward = 0
newstate = 0
hitwall = False
currentepisode = 0
currentstep = 0
stepstosolve = []
qsums = [0,0,0,0]
qsumsdifference = []
Q = np.zeros((64, 4))

#Initialize Maze structure
lake = [['S','F','F','F','F','F','F','H'],
        ['F','F','F','F','F','F','F','F'],
        ['F','F','F','F','F','H','F','F'],
        ['F','F','F','H','F','F','F','F'],
        ['F','F','F','H','F','F','F','F'],
        ['F','F','H','F','F','F','H','F'],
        ['F','F','F','F','F','F','F','F'],
        ['H','F','F','F','F','F','F','G']]

def printresults(lake,x,y): #prints an output of the maze to follow the bots behaviour
    temp = str(lake[x][y]) #creates an editable 1:1 copy so '.' can be added
    lake[x][y] = '.' # . represents the bot moving around
    for i in range(8):
        print(''.join(map(str, lake[i])))
    lake[x][y] = str(temp)


def action(command, x, y):#Definies actions the bot can take and can't take. 0=Right, 1= Down, 2= Left, 3 = Up
    hitwall = False
    if command == 0:
        x = x + 1
        if x > 7: #Keeping the Bot from going outside the maze by setting boundries for variables
            x = 7
            hitwall = True
    elif command == 1:
        y = y + 1
        if y > 7:
            y = 7
            hitwall = True
    elif command == 2:
        x = x-1
        if x < 0:
            x = 0
            hitwall = True
    elif command == 3:
        y = y -1
        if y < 0:
            y = 0
            hitwall = True
    else: print('Wrong Command') #for debugging
    return x,y,hitwall

def checkfield(x,y): #Checks each field for a reward and for reset commands
    if lake[x][y] =='H': #If Bot lands on a hole 'H' reward is deducted and he is reset to the start
        x = 0
        y = 0
        reward =-5
    elif lake[x][y] =='G': #If Bot reaches the Goal he will receive the reward and be reset
        reward = 10
        x = 0
        y = 0
        global currentepisode
        currentepisode = currentepisode + 1 #counter for episodes
        global stepstosolve
        stepstosolve.append(currentstep-sum(stepstosolve)) #calculates steps needed since last reward
        global printlake
        printlake = False
    else: reward = 0 #Any other fields 'F' give no reward
    return reward,x,y

def checkstate(x,y): #Checks the Q state number of the state the bot is currently standing on
    state = x+y*8 #adding coordinate values to calculate state
    return state

for i in range(num_steps):
    currentstep = currentstep+1 #counter for steps
    if printlake == True: #if true gives console output of maze
        printresults(lake,x,y)
    state = copy.copy(newstate)
    #epsilon greedy policy, 
    #a random chance epsilon decides between doing a random or known move 
    if epsilon < random.uniform(0, 1) and sum(Q[state,:]) != 0: 
        command = np.argmax(Q[state,:])		
    else:
        command = randrange(0, 4)
    x,y,hitwall = action(command,x,y)
    reward,x,y = checkfield(x,y) #Calls function to calculate if coordinates offers reward
    newstate = checkstate(x,y) #Calls function to check state number of new coordinates
    if punishwall == True:
        if hitwall == True:
            reward = -1
    if printlake == True:
        time.sleep(1) 
    treward += reward #calculates the total reward
    if currentepisode == 50:
        treward50 = copy.copy(treward) #stores the total reward at episode 50
    Q[state,command] = Q[state,command] + alpha * (reward+ gamma*np.max(Q[newstate,:])-Q[state,command])
    if printlake == True:
        print('Step Reward: ' + str(reward))
    
    if sum(stepstosolve[-20:-1]) != 0: #Break Calculation Steps neeeded for solving
        checkbreak = sum(stepstosolve[-20:-11])/sum(stepstosolve[-10:-1])
        if checkbreak < 1.005 and checkbreak > 0.995:
            print('break stepstosolve')
            break
        
    if reward == 10: #Break Calculation sums of q values and differences between them.
        qsums = np.vstack([qsums,sum(Q)])
        if len(qsums)>2:
            qsumsdifference.append(sum(abs(qsums[-1]-qsums[-2])))
            if len(qsums)>10 and sum(qsumsdifference[-10:-1]) < 0.01:
                print('break qsumdifference')
                break
    
if plotprint == True:
    plt.title('Icelake 8x8')
    plt.plot(stepstosolve, 'g-')
    plt.text(len(stepstosolve)/3*2-1, max(stepstosolve)/4*3*1.2, 'Alpha: '+ str(alpha)+ '\nGamma: ' +str(gamma) + '\nEpsilon: ' + str(epsilon) + '\nPunish Wall: ' + str(punishwall))
    plt.ylabel('Steps necessary until goal')
    plt.xlabel('Number of episodes until convergence')
    plt.axis([0,len(stepstosolve)-1,0,max(stepstosolve)*1.2])
    plt.show()