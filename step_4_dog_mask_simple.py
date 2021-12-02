"Adding Dog Mask"

import cv2
import numpy as np

def apply_mask(face:np.array,mask:np.array)->np.array:
    "Add mask to provided face"
    """
    Maintaining aspect ratio for dog mask to face so we get dimensions
    """
    #extract height and width of both images
    mask_h,mask_w,_=mask.shape
    face_h,face_w,_=face.shape
    #Resize the mask to fit the face
    factor = min(face_h/mask_h, face_w/mask_w)
    #computing new shape
    new_mask_w=int(factor*mask_w)
    new_mask_h=int(factor*mask_h)
    new_mask_shape =(new_mask_w,new_mask_h)
    #resize dog mask to shape
    resized_mask=cv2.resize(mask,new_mask_shape)
    face_with_mask=face.copy()
    #Image is represented in 3 dimensions colour is the third argument
    #So axis = 2 means to search all the values on the color axis listindex=2
    non_white_pixels = (resized_mask<250).all(axis=2)
    #computing offset to center image
    off_h=int((face_h-new_mask_h)/2)
    off_w = int((face_w-new_mask_w)/2)
    #copying all pixels from dog image to child image
    face_with_mask[off_h:off_h+new_mask_h,off_w: off_w+new_mask_w][non_white_pixels]=\
        resized_mask[non_white_pixels]
    return face_with_mask
    
    #cv2.imwrite('outputs/resized_dog.png', resized_mask)


def main():
    face=cv2.imread('assets/child.png')
    mask = cv2.imread('assets/dog.png')
    face_with_mask=apply_mask(face,mask)
    cv2.imwrite('outputs/child_with_dog_mask.png', face_with_mask)

if __name__=='__main__':
    main()