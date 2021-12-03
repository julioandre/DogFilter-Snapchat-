"""Test for face detection"""

import cv2
import numpy as np

from step_4_dog_mask_simple import apply_mask

apply_mask

def main():
    #Initializing video feed
    cap = cv2.VideoCapture(0)
    mask = cv2.imread('assets/dog.png')
    #initialize front face classifier
    cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
    #Load image children.png
    while True:
        #frame=cv2.imread('assets/children.png')
        
        # Converting image to black and white
        ret, frame = cap.read()
        frame_h,frame_w,_=frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blackwhite=cv2.equalizeHist(gray)
        rects= cascade.detectMultiScale(blackwhite, scaleFactor=1.3, minNeighbors=4, minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE)
        #scaleFactor specifies how much the image is reduced along each dimension.
        #minNeighbors denotes how many neighboring rectangles a candidate rectangle needs to be retained.
        #minSize is the minimum allowable detected object size. Objects smaller than this are discarded.
        # returns list of tuples, each tuple has minimum X,minimum y, width and height of the rectangle


        #The second and third arguments are opposing corners of the rectangle.
        #The fourth argument is the color to use. (0, 255, 0) corresponds to green for our RGB color space.
        #The last argument denotes the width of our line
        for x,y,w,h in rects:
            #cv2.rectangle(frame,(x,y),(x+y,w+h),(0,255,0),2)
            #crops frame slighltly larger than face
            y0,y1=int(y-0.25*h),int(y+0.75*h)
            x0,x1=x,x+w
            # check in case face is too close to edge
            if x0<0 or y0<0 or x1>frame_w or y1>frame_h:
                continue
            #apply mask
            frame[y0:y1,x0:x1]=apply_mask(frame[y0:y1,x0:x1],mask)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()