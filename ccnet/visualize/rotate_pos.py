import numpy as np

def rotate_pos(pos, angle=None, root_ind=None):
    """
    rotate the postion so that the root cell is at the top
    pos:
        n*2 array
    angle:
        the angle 
    root_ind: int
        root index
        
    """
    
    angle = None
    if angle is not None:
        theta = angle * np.pi / 180
    elif root_ind is not None:
        rx, ry = pos[root_ind, ]
        if rx>0:
            angle = np.arccos(ry/np.sqrt(rx*rx + ry*ry))
        elif rx<0:
            angle = -np.arccos(ry/np.sqrt(rx*rx + ry*ry))
        elif rx==0 and ry<0:
            angle = np.pi
        else:
            angle = 0
        theta = angle 
    else:
        return pos

    b = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    return np.dot(pos, b)