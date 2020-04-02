import cv2
import time
import matplotlib.pyplot as plt
import numpy as np



def delta_pix(frame, prev_frame, mask, alpha=.2, scale=.3):
    h,w,d = frame.shape
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    previous = cv2.resize(prev_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    current = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    difference = cv2.absdiff(previous, current)
    _, difference = cv2.threshold(difference, 3, 255, cv2.THRESH_BINARY)
    difference = cv2.resize(difference, (w,h), interpolation=cv2.INTER_CUBIC)
    mask = (1-alpha)*mask + alpha*difference
    return mask
def get_mask(image, kernel = (9,9)):
    opening = cv2.morphologyEx(image,cv2.MORPH_OPEN, kernel, iterations = 2)
    return cv2.dilate(opening,kernel,iterations=3)

cap = cv2.VideoCapture("test.mov")
cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
i = 30
cap.set(cv2.CAP_PROP_POS_FRAMES, i)
ret, prev_frame = cap.read()
gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
motion_field = np.zeros_like(gray)
board = prev_frame
h,w,d = prev_frame.shape
out = cv2.VideoWriter('diff_blend.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w,h))

while ret:
    t1 =time.time()
    ret, frame = cap.read(); i+=1
    if not ret:
        plt.show()
        break
    motion_field = delta_pix(frame, prev_frame, motion_field)
    mask = get_mask(motion_field)
    prev_frame = frame
    board[mask<1] = frame[mask<1]
    out.write(board)

#     plt.imshow(board[..., ::-1])
#     plt.title(i)
#     plt.pause(0.0005)
# plt.show()
    print(1/(time.time()-t1))
out.release()
