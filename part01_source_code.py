# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 21:01:54 2023

@author: saxen
"""

# Importing all libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from queue import PriorityQueue
import time
import math

# Creating class that creates objects with following attributes
class createNode:
    def __init__(self, pos, parent, c2c, tc, wL, wR, path):
        self.pos = pos
        self.parent = parent
        self.c2c = c2c
        self.tc = tc
        self.wL = wL
        self.wR = wR
        self.path = path


def Round_05(x):
    return int(round(x, 0))

# Functions to move in the allowed direction

def cost(node, goal_pos, wL, wR):
    child_path = []
    t = 0
    r = 3.3
    L = 16
    dt = 0.1
    xi, yi, thi = node.pos
    Thetan = np.deg2rad(thi)
    actionCost = 0
    Delta_Xn, Delta_Yn = (xi, yi)
    
    while t<1:
        t = t + dt
        Xs,Ys = Delta_Xn, Delta_Yn
        Delta_Xn += 0.5*r * (wL + wR) * math.cos(Thetan) * dt
        Delta_Yn += 0.5*r * (wL + wR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (wR - wL) * dt
        actionCost = actionCost + np.linalg.norm(np.asarray((Xs,Ys)) - np.asarray((Delta_Xn, Delta_Yn)))
        child_path.append([[Xs, Delta_Xn], [Ys, Delta_Yn]])
    Thetan = np.rad2deg(Thetan)
    xNew, yNew = (xi + Delta_Xn, yi + Delta_Yn)
    cgoal = np.linalg.norm(np.asarray((xNew, yNew)) - np.asarray((goal_pos[0], goal_pos[1])))
    child_created = createNode((Round_05(xNew), Round_05(yNew), Thetan), node, node.c2c + actionCost,
                               node.c2c + actionCost + cgoal , wL, wR, child_path)

    return actionCost, cgoal, child_created

# # Function to create obstacles in the visualization space
# def Obstacle_space(dims):
#     w, h = dims
#     angle = np.deg2rad(30)
#     grid = np.zeros((h + 1, w + 1, 3), dtype=np.uint8)
#     grid.fill(255)
    
#     # left rectangle
#     rect1 = np.array([[150, 0], [165, 0], [165, 125], [150, 125]])
    
#     # right rectangle
#     rect2 = np.array([[250, 200], [265, 200], [250, 75], [265, 75]])
    
#     # circle
#     cv2.circle(grid, (400,90), 50, (0,0,0), -1)
#     grid = cv2.fillPoly(grid, pts=[rect1, rect2, triangle, hexagon], color=(0, 0, 0))
#     grid = cv2.flip(grid, 0)
#     return grid


# Function to check if the robot is in the obstacle space (including the clearance margin)
def invade_obstacle(loc, gap):
    xMax, yMax = [600 + 1, 200 + 1]
    xMin, yMin = [0, 0]
    x, y, th = loc
    
    # walls
    if x <= gap or x >= xMax-gap or y <= gap or y >= yMax- gap:
#         print("hit wall")
        return False
    
    # left rectangle
    elif x>= 150-gap and x<= 165+gap and y >=75-gap and y <= 200-gap:
#         print("hit r1")
        return False
    
    # right rectangle
    elif x >= 250-gap and x<= 265+gap and y >=0+gap  and y <= 125+gap:
#         print("hit r2")
        return False
    
    # circle
    elif (x-400)**2 + (y-110)**2 <= (50+gap)**2:
#         print("hit circle")
        return False
    
    else:
#         print("no hit")
        return True


# Function to create possible children of a parent node within given constraints
def get_child(node, goal_pos, gap, w1, w2):
    w1 = w1 * (math.pi/30)
    w2 = w2 * (math.pi/30)
    xMax, yMax = [600 + 1, 250 + 1]
    xMin, yMin = [0, 0]
    children = []
    actionSet = [[0,w1], [w1,0], [w1,w1], [0,w2],
                 [w2,0], [w2,w2], [w1,w2], [w2,w1]]
    
    
    for action in actionSet:
        wL,wR = action 
        # check for each action set
        
        (actionCost, cgoal, child) = cost(node, goal_pos, wL, wR)
        if invade_obstacle(child.pos, gap):
            # if node is not generated, append in child list
            children.append((actionCost, cgoal, child))
        else:
            print("deleted:", child.pos)
            del child

    return children


# Function to create a backtrack path connecting all nodes resulting in shortest path by algorithm
def backtrack(current):
    path_bt = []
    parent = current
    while parent != None:
        path_bt.append(parent.pos)
        parent = parent.parent
    return path_bt


# A star path planning algorithm
def Astar_algo(start_pos, goal_pos, gap , w1, w2):
    if not invade_obstacle(start_pos, gap):
        print("start_pos position is in Obstacle grid")
        return False
    if not invade_obstacle(goal_pos, gap):
        print("goal_pos position is in Obstacle grid")
        return False
    tc_g = np.linalg.norm(np.asarray((start_pos[0], start_pos[1])) - np.asarray((goal_pos[0], goal_pos[1])))
    openList = []
    open_dict = {}
    closedList = []
    closed_dict = {}
    viz = []
    initial_node = createNode(start_pos, None, 0, tc_g, 0,0,[])
    openList.append((initial_node.tc, initial_node))
    open_dict[initial_node.pos] = initial_node

    start = time.time()
    while len(openList) > 0:
        openList.sort(key=lambda x: x[0])
        currentCost, current = openList.pop(0)
        open_dict.pop(current.pos)
        closedList.append(current)
        closed_dict[current.pos] = current
        if np.linalg.norm(np.asarray(current.pos[:2]) - np.asarray(goal_pos[:2])) <= 10.5:
            print("goal found")
            pathTaken = backtrack(current)
            end = time.time()
            print('Time taken to execute algorithm in sec: ',(end - start))
            print(pathTaken)
            print(viz)
            return pathTaken, viz
        else:
            childList = get_child(current, goal_pos, gap, w1, w2)
            for actionCost, actionGoal, child_created in childList:
#                 print(child_created.pos)
                if child_created.pos in closed_dict:
                    print("in CL: ",child_created.pos)
                    del child_created
                    continue
                if child_created.pos in open_dict:
                    print("in OL: ",child_created.pos)
                    if open_dict[child_created.pos].c2c > current.c2c + actionCost:
                        open_dict[child_created.pos].parent = current
                        open_dict[child_created.pos].c2c = current.c2c + actionCost
                        open_dict[child_created.pos].tc = open_dict[child_created.pos].c2c + actionGoal
                else:
                    print("new: ",child_created.pos)
                    child_created.parent = current
                    child_created.c2c = current.c2c + actionCost
                    child_created.tc = child_created.c2c + actionGoal
                    openList.append((child_created.tc, child_created))
                    open_dict[child_created.pos] = child_created
                    x, y, th = child_created.pos
                    viz.append(child_created)


# Function to visualize the visited nodes and the shortest path using openCV
def visualization(viz, pathTaken, start_pos, goal_pos):
    #save = cv2.VideoWriter('video_Astar_turtlebot.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (600 + 1, 250 + 1))
    plt.figure(figsize=(10,15), dpi=80)
    rectangle1 = plt.Rectangle((150, 75), 15, 125, fc='black')
    plt.gca().add_patch(rectangle1)

    rectangle2 = plt.Rectangle((250, 0), 15, 125, fc='black')
    plt.gca().add_patch(rectangle2)

    circle = plt.Circle((400,110),50, fc='black')
    plt.gca().add_patch(circle)

    plt.axis('scaled')
    plt.xlim((0,600))
    plt.ylim((0,200))
    plt.plot((start_pos[0],start_pos[1]),color='red')
    plt.plot((goal_pos[0],goal_pos[1]),color='green')
    start_time = time.time()
    for i in viz:
        for xv,yv in i.path:
            plt.plot(xv,yv,color="blue")

#     for i in range(len(pathTaken[::-1]) - 1):
#         x1, y1 = pathTaken[::-1][i][0], pathTaken[::-1][i][1]
#         x2, y2 = pathTaken[::-1][i + 1][0], pathTaken[::-1][i + 1][1]
#         cv2.arrowedLine(grid, (x1, grid.shape[0] - y1 - 1), (x2, grid.shape[0] - y2 - 1), [0, 0, 0], 1, tipLength=0.4)
#         save.write(grid)
#     save.write(grid)
#     save.write(grid)
#     save.write(grid)
#     save.write(grid)
#     save.release()
    end_time = time.time()
    print('Time taken to visualize in sec: ',(end_time - start_time))
    plt.show()


# User input variables
# xs= int(input('Enter start x-coordinate: '))
# ys= int(input('Enter start y-coordinate: '))
# ths= int(input('Enter start orientation in multiple of 30 deg: '))
start_pos= (50,50,30)

# xg= int(input('Enter goal x-coordinate: '))
# yg= int(input('Enter goal y-coordinate: '))
# thg= int(input('Enter goal orientation in multiple of 30 deg: '))
goal_pos= (100,150,30)

# gap = int(input('Enter clearance (robot radius + bloat): '))
# w1= int(input('Enter w1: '))
# w2= int(input('Enter w2: '))

pathTaken,viz = Astar_algo(start_pos,goal_pos,20,5,10)
visualization(viz,pathTaken,start_pos,goal_pos)