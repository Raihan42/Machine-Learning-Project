# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:39:42 2018

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 00:27:47 2018

@author: Nabila PC
"""

#The optimal values of m and b can be actually calculated with way less effort than doing a linear regression. 
#this is just to demonstrate gradient descent

from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def linear_reg(t0,t1,x):
    return t0+(t1*x)
def cost(t0, t1, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (t1 * x + t0)) ** 2
    return totalError / float(len(points))

def step_gradient(t0_current, t1_current, points, learningRate):
    t0_gradient = 0
    t1_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        t0_gradient += -(2/N) * (y - ((t1_current * x) + t0_current))
        t1_gradient += -(2/N) * x * (y - ((t1_current * x) + t0_current))
    new_t0 = t0_current - (learningRate * t0_gradient)
    new_t1 = t1_current - (learningRate * t1_gradient)
    return [new_t0, new_t1]

def gradient_descent_runner(points, starting_t0, starting_t1, learning_rate, num_iterations):
    t0 = starting_t0
    t1 = starting_t1
    for i in range(num_iterations):
        t0, t1 = step_gradient(t0, t1, array(points), learning_rate)
    return [t0, t1]

def run():
    points = array([[48.0, 62.5, 72.0, 49.8, 50.8, 48.6, 56.6, 72.2, 64.8, 64.8, 81.6, 55.5, 48.7, 54.7, 65.5, 60.2, 60.4, 60.6, 60.8, 61.0, 61.2, 61.4, 61.6, 61.8, 62.0],
                    [30.0, 40.0, 50.8, 35.0, 36.9, 35.8, 40.5, 60.8, 48.3, 50.7, 89.2, 32.6, 30.0, 33.8, 50.2, 50.0, 51.0, 53.0, 55.0, 57.0, 59.0, 61.0, 63.0, 65.0, 67]]) 
    learning_rate = 0.0001
    initial_t0 = 0
    initial_t1 = 0
    num_iterations = 100000
    print ("Starting gradient descent at t0 = {0}, t1 = {1}, cost = {2}".format(initial_t0, initial_t1, cost(initial_t0, initial_t1, points)))
    
    [t0, t1] = gradient_descent_runner(points, initial_t0, initial_t1, learning_rate, num_iterations)
    print ("After {0} iterations t0 = {1}, t1 = {2}, cost = {3}".format(num_iterations, t0, t1, cost(t0, t1, points)))
    reg=linear_reg(t0,t1,40.0)
    print("Pridected weight: ")
    print(reg)

run()