import cv2 as cv
import numpy as np
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def get_mapa():
    """
    Get the image of the map"""
    return cv.imread(os.path.join(FILE_PATH, 'mapa.png'))

def get_biggest_line(img):
    """
    Get the biggest line in the image using HoughLinesP"""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 200)
    if lines is not None:
        return lines[0]
    
def get_relation_px_to_km(verbose=False):
    """
    Get the relation between pixels and kilometers in the image
    bearing in mind that the text in the image is '2 km' long"""

    relation_img = cv.imread(os.path.join(FILE_PATH, 'measure.png'))
    black_and_white = cv.cvtColor(relation_img, cv.COLOR_BGR2GRAY)
    line = get_biggest_line(relation_img)
    x1, y1, x2, y2 = line[0]
    cv.line(relation_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if verbose:
        cv.imshow('Relation px to km', relation_img)
        cv.waitKey(0)
    
    return (x2-x1)/2

def third_point_of_triangle(p1,p2,c, d, ang):
    """

    Args:
    p1: point 1
    p2: point 2
    c: center between p1 and p2
    d: distance between p1 and p2
    ang: angle between p1 and p2 from p3

    """
    distance_to_p3 = d/np.tan(ang)
    # print(f"Distance to p3: {distance_to_p3}")
    # vector perpendicular to the line p1-p2
    # print(f"p2-p1: {np.array(p2)-np.array(p1)}")
    v = np.array([p2[1]-p1[1], p1[0]-p2[0]])
    # normalize the vector
    v = v/np.linalg.norm(v)
    # print(f"v: {v}")
    # get the third point
    p3 = c + distance_to_p3*v
    # print(f"p1: {p1}")
    # print(f"p2: {p2}")
    # print(f"p3: {p3}")

    return p3
    

def circle_from_points(p1, p2, p3):
    """
    Get the circle from three points"""
    import sympy as sp
    from sympy.solvers import solve
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # (x1-x)^2 + (y1-y)^2 = r^2
    # (x2-x)^2 + (y2-y)^2 = r^2
    # (x3-x)^2 + (y3-y)^2 = r^2
    # solve the system of equations
    x_sp, y_sp, r_sp = sp.symbols('x y r')
    eq1 = sp.Eq((x1-x_sp)**2 + (y1-y_sp)**2, r_sp**2)
    eq2 = sp.Eq((x2-x_sp)**2 + (y2-y_sp)**2, r_sp**2)
    eq3 = sp.Eq((x3-x_sp)**2 + (y3-y_sp)**2, r_sp**2)
    sol = solve([eq1, eq2, eq3], dict=True)
    
    #print(f"Solution: {sol}")

    correct = None
    for s in sol:
        if s[r_sp] > 0:
            # if the radius is positive
            if correct is not None and s[r_sp] < correct[r_sp]:
                # if there is a solution with a smaller radius
                correct = s
            elif correct is None:
                correct = s
    #print("Correct: ", correct)
    
    x = int(correct[x_sp])
    y = int(correct[y_sp])
    r = int(correct[r_sp])
    return (x, y), r


def intersection(circle1, circle2):
    """
    Get the intersection of two circles"""
    import sympy as sp
    from sympy.solvers import solve
    x1, y1 = circle1 [0]
    r1 = circle1[1]
    x2, y2 = circle2[0]
    r2 = circle2[1]
    
    x, y = sp.symbols('x y')
    eq1 = sp.Eq((x-x1)**2 + (y-y1)**2, r1**2)
    eq2 = sp.Eq((x-x2)**2 + (y-y2)**2, r2**2)
    sol = solve([eq1, eq2], dict=True)

    to_return = [(int(s[x]), int(s[y])) for s in sol]
    return to_return
        

if __name__ == "__main__":
    print(f"{get_relation_px_to_km(verbose=True)} px/km")