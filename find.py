#!/usr/bin/env python

import cv2 as cv
from collections import deque
import numpy as np
from umucv.util import putText
from utils import get_mapa, get_relation_px_to_km, third_point_of_triangle, circle_from_points, intersection

# Points in the image of reference
points_img = []
# Distance in the image
d_img = 0

# Points in the map of reference
points = []

# Calibration matrix, used to get the angle in the image
K = np.array([[971, 0, 636], [0, 970, 365], [0, 0, 1]])

# Angle in the image
ang = 0
ang2 = 0
# Distance in the map
d = 0
d_2 = 0

# Circle in the map
circle = None
circle2 = None

# Intersections of the circles
intersections = None


def getXYZ(p):
    """
    Get the XYZ coordinates of a point in the image"""
    return np.linalg.inv(K).dot([p[0],p[1],1])

def angle(p1,p2):
    vP1 = getXYZ(p1)
    vP2 = getXYZ(p2)
    return np.arccos(
        np.dot(vP1,vP2) 
        / 
        (np.linalg.norm(vP1) * np.linalg.norm(vP2))
    )
def rad2gra(rad):
    """
    Convert radians to degrees"""
    return rad * 180 / np.pi

def pixel_distance(p1, p2):
    """
    Get the distance between two points in pixels"""
    return np.linalg.norm(np.array(p2)-p1)

def fun(event, x, y, flags, param):
    """
    Callback function for the mouse event in the map image"""
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))
    if len(points) == 2:
        global d
        
        d = pixel_distance(points[0], points[1])

        if len(points_img) >= 2:
            circleF()
    
    if len(points) == 4:
        global d_2
        d_2 = pixel_distance(points[2], points[3])
        if len(points_img) >= 2:
            circleF()

def funImg(event, x, y, flags, param):
    """
    Callback function for the mouse event in the reference image"""
    if event == cv.EVENT_LBUTTONDOWN:
        points_img.append((x,y))
    if len(points_img) == 2:
        global ang
        global d_img
        

        ang = angle(points_img[0],points_img[1])
        d_img = pixel_distance(points_img[0], points_img[1])
        
        if len(points) >= 2:
            circleF()
    
    if len(points_img) == 4:
        global ang2
        ang2 = angle(points_img[2],points_img[3])
        if len(points) >= 2:
            circleF()

    
        
def circleF():
    """
    Get the circle from the points in the map and the angle obtained
    from the image of reference
    
    Args:
    circle_n: number of the circle in the map"""

    global circle, circle2, intersections

    c = np.mean(points[0:2], axis=0).astype(int)
    p3 = third_point_of_triangle(points[0], points[1], c, d, ang)
    circle = circle_from_points(points[0], points[1], p3)

    if len(points) == 4 and len(points_img) == 4:
        c = np.mean(points[2:], axis=0).astype(int)
        p3 = third_point_of_triangle(points[2], points[3], c, d_2, ang2)
        circle2 = circle_from_points(points[2], points[3], p3)

        if circle is not None and circle2 is not None:
            intersections = intersection(circle, circle2)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Locate you given a image and 4 points of reference in the map and 4 points of reference in the image')
    parser.add_argument('-i', '--image', type=str, help='Path to the image', required=True)
    parser.add_argument('-c', '--calibration-file', type=str, help='Calibration file', required=True)
    args = vars(parser.parse_args())

    image = cv.imread(args['image'])
    calibdata = np.loadtxt(args['calibration_file'])

    K = calibdata[:9].reshape(3,3)

    cv.namedWindow("mapa")
    cv.setMouseCallback("mapa", fun)

    cv.namedWindow("image")
    cv.setMouseCallback("image", funImg)


    mapa = get_mapa()
    relation = get_relation_px_to_km()
    putText(mapa, f'{relation:.1f} px/km', (10,10))


    while True:
        toshowMapa = mapa.copy()
        toshowImg = image.copy()
        for p in points:
            cv.circle(toshowMapa, p,3,(0,0,255),-1)
        if len(points) >= 2:
            cv.line(toshowMapa, points[0],points[1],(0,0,255))
            c = np.mean(points, axis=0).astype(int)
            putText(toshowMapa,f'{d/relation:.1f} km',c+(0,40))

            if circle is not None:
                #print(f"Circle: {circle}")
                cv.circle(toshowMapa, circle[0],3,(0,255,0),-1)
                cv.circle(toshowMapa, circle[0], circle[1],(0,255,0),2)
            
            if len(points) == 4:
                cv.line(toshowMapa, points[2],points[3],(0,0,255))
                c = np.mean(points[2:], axis=0).astype(int)
                putText(toshowMapa,f'{d_2/relation:.1f} km',c+(0,40))

                if circle2 is not None:
                    #print(f"Circle: {circle}")
                    cv.circle(toshowMapa, circle2[0],3,(0,255,0),-1)
                    cv.circle(toshowMapa, circle2[0], circle2[1],(0,255,0),2)

        
        for p in points_img:
            cv.circle(toshowImg, p,3,(0,0,255),-1)
        if len(points_img) >= 2:
            cv.line(toshowImg, points_img[0],points_img[1],(0,0,255))
            c = np.mean(points_img[0:2], axis=0).astype(int)
            putText(toshowImg,f'{rad2gra(ang):.1f}deg',c+(0,20))
            putText(toshowImg,f'{d_img:.1f} pix',c)

            if len(points_img) == 4:
                cv.line(toshowImg, points_img[2],points_img[3],(0,0,255))
                c = np.mean(points_img[2:], axis=0).astype(int)
                putText(toshowImg,f'{rad2gra(ang2):.1f}deg',c+(0,20))
                putText(toshowImg,f'{d_img:.1f} pix',c)

        if intersections is not None:
            for i in intersections:
                cv.circle(toshowMapa, (int(i[0]), int(i[1])), 3, (0,0,0), -1)
                putText(toshowMapa, f'{i[0]:.1f}, {i[1]:.1f}', (int(i[0]), int(i[1])+20))


        cv.imshow("image", toshowImg)
        cv.imshow("mapa", toshowMapa)
        if cv.waitKey(1) == ord('q'):
            break
    
cv.destroyAllWindows()

