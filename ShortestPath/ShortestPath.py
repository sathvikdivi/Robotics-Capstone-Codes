#!/usr/bin/python

import numpy as np
import yaml
import math

def dijkstras(occupancy_map,x_spacing,y_spacing,start,goal):
    """
    Implements Dijkstra's shortest path algorithm
    Input:
    occupancy_map - an N by M numpy array of boolean values (represented
        as integers 0 and 1) that represents the locations of the obstacles
        in the world
    x_spacing - parameter representing spacing between adjacent columns
    y_spacing - parameter representing spacing between adjacent rows
    start - a 3 by 1 numpy array of (x,y,theta) for the starting position 
    goal - a 3 by 1 numpy array of (x,y,theta) for the finishing position 
    Output: 
    path: list of the indices of the nodes on the shortest path found
        starting with "start" and ending with "end" (each node is in
        metric coordinates)
    """
    
    
    # finding the size of the map
    mapsize = occupancy_map.shape
    rows = mapsize[0]  # no. of rows
    cols = mapsize[1]  # no. of cols
    
    # directions
    delta = [[-1,0], # go up
             [0,-1], # go left
             [1,0],  # go down
             [0,1]]  # go right
    
    # indices of start position
    iStart = int(np.rint(start[1]/y_spacing - 0.5))
    jStart = int(np.rint(start[0]/x_spacing - 0.5))
    
    # indices of goal position
    iGoal = int(np.rint(goal[1]/y_spacing - 0.5))
    jGoal = int(np.rint(goal[0]/x_spacing - 0.5))
    
    # direction handles
    delta_name = ['^','<','V','>']
    
    # matrix that checks if a node has been visited before or not
    closed = np.full((rows,cols),0)
    closed[iStart,jStart] = 1 # closing the starting node
    
    # expand matrix
    expand = np.full((rows,cols),-1)
    
    # action matrix
    action = np.full((rows,cols),-1)       
    
    # starting states
    xStart = start[0]
    yStart = start[1]
    thetaStart = start[2]
    g = 0 # cost based on distance
    
    open = [[g,iStart,jStart]]
    
    found = False # flag that is set when search is complete
    resign = False # flag that is set if we can't find expand
    count = 0
    
    while ((not found) and (not resign)):
        if len(open) == 0:
            resign = True
            return "Fail"
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            iTemp = next[1]
            jTemp = next[2]
            g = next[0]
            expand[iTemp][jTemp] = count
            count += 1
            xTemp = (jTemp + 0.5)*x_spacing
            yTemp = (iTemp + 0.5)*y_spacing
            
            if (iTemp == iGoal and jTemp == jGoal):
                found = True
            else:
                for dvar in range(len(delta)):
                    iNew = iTemp + delta[dvar][0]
                    jNew = jTemp + delta[dvar][1]
                    xNew = (jNew + 0.5)*x_spacing
                    yNew = (iNew + 0.5)*y_spacing
                    if (iNew >= 0 and iNew < rows) and (jNew >= 0 and jNew < cols):
                        if (closed[iNew][jNew] == 0) and (occupancy_map[iNew][jNew] == 0):
                            gNew = g + math.sqrt((xNew - xTemp)**2 + (yNew - yTemp)**2)
                            open.append([gNew,iNew,jNew])
                            closed[iNew][jNew] = 1
                            action[iNew][jNew] = dvar
    
    policy =  np.full((rows,cols),' ')
    policy[iGoal][jGoal] = '*'
    
    iTemp2 = iGoal
    jTemp2 = jGoal
    
    pathIndex = [[iTemp2,jTemp2]]
    
    while (iTemp2 != iStart) or (jTemp2 != jStart):
        iNew2 = iTemp2 - delta[action[iTemp2][jTemp2]][0]
        jNew2 = jTemp2 - delta[action[iTemp2][jTemp2]][1]
        policy[iNew2][jNew2] = delta_name[action[iTemp2][jTemp2]]
        pathIndex.append([iNew2,jNew2])
        iTemp2 = iNew2
        jTemp2 = jNew2
        
    
    for var in policy:
        print(var)
    
    pathIndex.reverse()
    
    path = []
    
    for ivar in range(len(pathIndex)):
        iTemp3 = pathIndex[ivar][0]
        jTemp3 = pathIndex[ivar][1]
        if iTemp3 == iStart and jTemp3 == jStart:
            xTemp3 = xStart
            yTemp3 = yStart
        elif iTemp3 == iGoal and jTemp3 == jGoal:
            xTemp3 = goal[0]
            yTemp3 = goal[1]
        else:
            xTemp3 = (jTemp3 + 0.5)*x_spacing
            yTemp3 = (iTemp3 + 0.5)*y_spacing
        
        path.append([xTemp3,yTemp3])
    
    print(path)
    a = 1
    return path

def test():
    """
    Function that provides a few examples of maps and their solution paths
    """
    test_map1 = np.array([
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]])
    x_spacing1 = 0.13
    y_spacing1 = 0.2
    start1 = np.array([[0.3], [0.3], [0]])
    goal1 = np.array([[0.6], [1], [0]])
    path1 = dijkstras(test_map1,x_spacing1,y_spacing1,start1,goal1)
    true_path1 = np.array([
        [ 0.3  ,  0.3  ],
        [ 0.325,  0.3  ],
        [ 0.325,  0.5  ],
        [ 0.325,  0.7  ],
        [ 0.455,  0.7  ],
        [ 0.455,  0.9  ],
        [ 0.585,  0.9  ],
        [ 0.600,  1.0  ]
        ])
    if np.array_equal(path1,true_path1):
      print("Path 1 passes")

    test_map2 = np.array([
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1],
             [1, 0, 0, 1, 1, 0, 0, 1],
             [1, 0, 0, 1, 1, 0, 0, 1],
             [1, 0, 0, 1, 1, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1],
             [1, 0, 0, 0, 0, 0, 0, 1],
             [1, 1, 1, 1, 1, 1, 1, 1]])
    start2 = np.array([[0.5], [1.0], [1.5707963267948966]])
    goal2 = np.array([[1.1], [0.9], [-1.5707963267948966]])
    x_spacing2 = 0.2
    y_spacing2 = 0.2
    path2 = dijkstras(test_map2,x_spacing2,y_spacing2,start2,goal2)
    true_path2 = np.array([[ 0.5,  1.0],
                           [ 0.5,  1.1],
                           [ 0.5,  1.3],
                           [ 0.5,  1.5],
                           [ 0.7,  1.5],
                           [ 0.9,  1.5],
                           [ 1.1,  1.5],
                           [ 1.1,  1.3],
                           [ 1.1,  1.1],
                           [ 1.1,  0.9]])
    if np.array_equal(path2,true_path2):
      print("Path 2 passes")

def test_for_grader():
    """
    Function that provides the test paths for submission
    """
    test_map1 = np.array([
              [1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 1, 0, 0, 0, 1, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 1, 0, 1, 0, 1, 0, 1],
              [1, 0, 0, 0, 1, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1]])
    x_spacing1 = 1
    y_spacing1 = 1
    start1 = np.array([[1.5], [1.5], [0]])
    goal1 = np.array([[7.5], [1], [0]])
    path1 = dijkstras(test_map1,x_spacing1,y_spacing1,start1,goal1)
    s = 0
    for i in range(len(path1)-1):
      s += np.sqrt((path1[i][0]-path1[i+1][0])**2 + (path1[i][1]-path1[i+1][1])**2)
    print("Path 1 length:")
    print(s)


    test_map2 = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]])
    start2 = np.array([[0.4], [0.4], [1.5707963267948966]])
    goal2 = np.array([[0.4], [1.8], [-1.5707963267948966]])
    x_spacing2 = 0.2
    y_spacing2 = 0.2
    path2 = dijkstras(test_map2,x_spacing2,y_spacing2,start2,goal2)
    s = 0
    for i in range(len(path2)-1):
      s += np.sqrt((path2[i][0]-path2[i+1][0])**2 + (path2[i][1]-path2[i+1][1])**2)
    print("Path 2 length:")
    print(s)



def main():
    
    #test_for_grader()
    test()
    # Load parameters from yaml
    param_path = 'params.yaml' # rospy.get_param("~param_path")
    f = open(param_path,'r')
    params_raw = f.read()
    f.close()
    params = yaml.load(params_raw)
    # Get params we need
    occupancy_map = np.array(params['occupancy_map'])
    pos_init = np.array(params['pos_init'])
    pos_goal = np.array(params['pos_goal'])
    x_spacing = params['x_spacing']
    y_spacing = params['y_spacing']
    path = dijkstras(occupancy_map,x_spacing,y_spacing,pos_init,pos_goal)
    print(path)

    
if __name__ == '__main__':
    main()

