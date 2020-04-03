import cv2
import time
import matplotlib.pyplot as plt
import numpy as np


def get_mask(image, kernel = (9,9)):
    opening = cv2.morphologyEx(image,cv2.MORPH_OPEN, kernel, iterations = 2)
    return cv2.dilate(opening,kernel,iterations=3)

def get_segmentation(image, scale=.25, kernel=np.ones((3,3), np.uint8)):
    h,w,d = image.shape
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    shifted = cv2.pyrMeanShiftFiltering(small, 31, 31)
    board_color = np.ones_like(shifted)*np.mean(shifted[0], axis=0)*.9
    thresh = np.logical_and.reduce(shifted<board_color, axis=2)*1.0
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,np.ones((3,3), np.uint8), iterations = 2)
    mask = cv2.dilate(opening,np.ones((5,5), np.uint8),iterations=3)
    full_mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_CUBIC)
    # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg,sure_fg)
    # ret, markers = cv2.connectedComponents(sure_fg)
    # markers = markers+1
    # markers[unknown==255] = 0
    # markers = cv2.watershed(image,markers)
    # image[markers == -1] = [255,0,0]
    return full_mask

cap = cv2.VideoCapture("test.mov")
cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
i = 30
cap.set(cv2.CAP_PROP_POS_FRAMES, i)
ret, prev_frame = cap.read()
gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
motion_field = np.zeros_like(gray)
board = prev_frame
h,w,d = prev_frame.shape
out = cv2.VideoWriter('mean_shift.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w,h))

while ret:
    t1 =time.time()
    ret, frame = cap.read(); i+=1
    if not ret:
        plt.show()
        break
    mask = get_segmentation(frame)
    board[mask<.5] = frame[mask<.5]
    out.write(board)
    # plt.imshow(board[..., ::-1])
    # plt.title(i)
    # plt.pause(0.0005)
    print(i, 1/(time.time()-t1))
out.release()
plt.show()
