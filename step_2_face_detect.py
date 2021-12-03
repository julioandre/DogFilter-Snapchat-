"""Test for face detection"""

import cv2


def main():
    #initialize front face classifier
    cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
    #Load image children.png
    frame=cv2.imread('assets/children.png')
    # Converting image to black and white
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
        cv2.rectangle(frame,(x,y),(x+y,w+h),(0,255,0),2)
    cv2.imwrite('outputs/children_detected.png', frame)
if __name__ == '__main__':
    main()