# Importing all libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import math
import csv
import os

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

# Function to check if the robot is in the obstacle space (including the clearance margin)
def invade_obstacle(loc, gap):
    xMax, yMax = [600 + 1, 200 + 1]
    xMin, yMin = [0, 0]
    x, y, th = loc
    
    # walls
    if x <= gap or x >= xMax-gap or y <= gap or y >= yMax- gap:
        return False
    
    # left rectangle
    elif x>= 150-gap and x<= 165+gap and y >=75-gap and y <= yMax-gap:
        return False
    
    # right rectangle
    elif x >= 250-gap and x<= 265+gap and y >=yMin+gap  and y <= 125+gap:
        return False
    
    # circle
    elif (x-400)**2 + (y-110)**2 <= (50+gap)**2:
        return False
    
    else:
        return True
    

def get_child(node, goal_pos, gap, w1, w2):
    children = []
    actionSet = [[0,w1], [w1,0], [w1,w1], [0,w2],
                 [w2,0], [w2,w2], [w1,w2], [w2,w1]]
    
    
    for action in actionSet:
        wL,wR = action 
        wL= wL*(math.pi/30)
        wR= wR*(math.pi/30)
        child_path = []
        # check for each action set
        t = 0
        r = 3.3
        L = 16
        dt = 0.1
        validFlag = True
        xi, yi, thi = node.pos
        Thetan = np.deg2rad(thi)
        actionCost = 0
        Xn=xi
        Yn=yi

        while t<1:
            t = t + dt
            Xs,Ys = Xn,Yn
            Thetan += (r / L) * (wR - wL) * dt
            Xn,Yn = (Xs + 0.5*r * (wL + wR) * math.cos(Thetan) * dt, Ys + 0.5*r * (wL + wR) * math.sin(Thetan) * dt)
            child_path.append([[Xs, Xn], [Ys, Yn]])
            if not invade_obstacle((Xn, Yn, Thetan), gap):
                validFlag = False
                break
        if validFlag:
            Thetan = np.rad2deg(Thetan)
            actionCost = np.linalg.norm(np.asarray((Xs,Ys)) - np.asarray((Xn,Yn)))
            cgoal = np.linalg.norm(np.asarray((Xn, Yn)) - np.asarray((goal_pos[0], goal_pos[1])))
            child_created = createNode((Round_05(Xn), Round_05(Yn), Thetan), node, node.c2c + actionCost,
                                       node.c2c + actionCost + cgoal , wL, wR, child_path)
            children.append((actionCost, cgoal, child_created))
    return children


# Function to create a backtrack path connecting all nodes resulting in shortest path by algorithm
def backtrack(current):
    path = []
    speeds= []
    trajectory = []
    parent = current
    while parent != None:
        path.append(parent.pos)
        speeds.append((parent.wL,parent.wR))
        trajectory.append(parent)
        parent = parent.parent
    return path, speeds, trajectory


# A star path planning algorithm
def Astar_algo(start_pos, goal_pos, gap , w1, w2, thresh):
    tc_g = np.linalg.norm(np.asarray((start_pos[0], start_pos[1])) - np.asarray((goal_pos[0], goal_pos[1])))
    openList = []
    open_dict = {}
    closedList = []
    closed_dict = {}
    viz = []
    initial_node = createNode(start_pos, None, 0, tc_g, 0,0, [])
    openList.append((initial_node.tc, initial_node))
    open_dict[initial_node.pos] = initial_node

    start = time.time()
    while len(openList) > 0:
        openList.sort(key=lambda x: x[0])
        currentCost, current = openList.pop(0)
        open_dict.pop(current.pos)
        closedList.append(current)
        closed_dict[current.pos] = current
        if np.linalg.norm(np.asarray(current.pos[:2]) - np.asarray(goal_pos[:2])) <= thresh:
            pathTaken, rpms, trajectory = backtrack(current)
            end = time.time()
            print("Path Found")
            print('Time taken to execute algorithm: ',(end - start)," sec")
            with open('velocity.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for w1,w2 in rpms[::-1]:    
                    writer.writerow([round(w1,3),round(w2,3)])
            return (pathTaken,rpms,trajectory, viz)
        else:
            childList = get_child(current, goal_pos, gap, w1, w2)
            for actionCost, actionGoal, child_created in childList:
                if child_created.pos in closed_dict:
                    del child_created
                    continue
                if child_created.pos in open_dict:
                    if open_dict[child_created.pos].c2c > current.c2c + actionCost:
                        open_dict[child_created.pos].parent = current
                        open_dict[child_created.pos].c2c = current.c2c + actionCost
                        open_dict[child_created.pos].tc = open_dict[child_created.pos].c2c + actionGoal
                else:
                    child_created.parent = current
                    child_created.c2c = current.c2c + actionCost
                    child_created.tc = child_created.c2c + actionGoal
                    openList.append((child_created.tc, child_created))
                    open_dict[child_created.pos] = child_created
                    x, y, th = child_created.pos
                    viz.append(child_created)


# Function to visualize the visited nodes and the shortest path using openCV
def visualization(viz, pathTaken,rpms,trajectory, start_pos, goal_pos, animate=False):
    imCtr = 0
    plt.rcParams["figure.figsize"] = [30,20]
#     os.rmdir(dir_path)
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.margins(0)
    plt.xlim(0,600)
    plt.ylim(0,200)
    if animate:
        if not os.path.exists('animate'):
            os.makedirs('animate')
        save = cv2.VideoWriter('turtlebot.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(2160,1440))
    # set goal and start
    ax.scatter(start_pos[0],start_pos[1],color = "red")
    ax.scatter(goal_pos[0],goal_pos[1],color = "green")
    
    # draw obstacle space
    xObs, yObs = np.meshgrid(np.arange(0, 600), np.arange(0, 200))
    rectangle1 = plt.Rectangle((150, 75), 15, 125, fc='black')
    ax.add_artist(rectangle1)
    
    rectangle2 = plt.Rectangle((250, 0), 15, 125, fc='black')
    ax.add_artist(rectangle2)
    
    cc = plt.Circle(( 400 , 110 ), 50, color = "black") 
    ax.add_artist( cc )

    boundary1 = (xObs<=5) 
    ax.fill(xObs[boundary1], yObs[boundary1], color='black')
    boundary2 = (xObs>=595) 
    ax.fill(xObs[boundary2], yObs[boundary2], color='black')
    boundary3 = (yObs<=5) 
    ax.fill(xObs[boundary3], yObs[boundary3], color='black')
    boundary4 = (yObs>=195) 
    ax.fill(xObs[boundary4], yObs[boundary4], color='black')
    ax.set_aspect(1)
    if animate:
        plt.savefig("animate/animateImg"+str(imCtr)+".png")
        imCtr += 1
    start_time = time.time()
    
    # to visualise child exploration
    for ch in viz:
        for xv,yv in ch.path:
            ax.plot(xv,yv, color="cyan")
        if animate:
            plt.savefig("animate/animateImg"+str(imCtr)+".png")
            imCtr += 1
            
    
#     for x,y,th in pathTaken[::-1]:
#         ax.scatter(x,y, color="black")
#         if animate:
#             plt.savefig("animate/animateImg"+str(imCtr)+".png")
#             imCtr += 1
            
    # to visualize backtrack path
    for pt,bt_path in zip(trajectory[::-1],pathTaken[::-1]):
        ax.scatter(bt_path[0],bt_path[1], color="black")
        for xt,yt in pt.path:
            ax.plot(xt,yt, color="red")
        if animate:
            plt.savefig("animate/animateImg"+str(imCtr)+".png")
            imCtr += 1
    if animate:
        for filename in os.listdir("animate"):
            img = cv2.imread(os.path.join("animate",filename))
            save.write(img)
        save.release()
    end_time = time.time()
    print('Time taken to visualize: ',(end_time - start_time)," sec")


# User input variables
def checkStartGoal(start_pos, goal_pos, gap):
    if not invade_obstacle(start_pos, gap):
        print("Start position is in Obstacle space")
        return False
    elif not invade_obstacle(goal_pos, gap):
        print("Goal position is in Obstacle space")
        return False
    else:
        return True

def getInputs():
    xs= int(input('Enter start x-coordinate: '))
    ys= int(input('Enter start y-coordinate: '))
    ths= int(input('Enter start orientation in deg: '))
    start_pos= (xs,ys,ths)

    xg= int(input('Enter goal x-coordinate: '))
    yg= int(input('Enter goal y-coordinate: '))
    goal_pos= (xg,yg,0)

    gap = int(input('Enter clearance (robot radius + bloat): '))
    w1= int(input('Enter w1: '))
    w2= int(input('Enter w2: '))
    thresh = int(input('Enter goal threshold: '))
    return start_pos, goal_pos, gap, w1, w2, thresh
start_pos, goal_pos, gap, w1, w2, thresh = getInputs()
returnVal = checkStartGoal(start_pos, goal_pos, gap)
while True:
    if returnVal:
        print("Finding Path....")
        pathTaken,rpms,trajectory,viz = Astar_algo(start_pos,goal_pos,gap,w1,w2, thresh)
        animate = False
        print("Building Visualisation: Animate = ", animate)
        visualization(viz,pathTaken,rpms,trajectory,start_pos,goal_pos,animate = animate)
        break
    else:
        start_pos, goal_pos, gap, w1, w2, thresh = getInputs()
        returnVal = checkStartGoal(start_pos, goal_pos, gap)