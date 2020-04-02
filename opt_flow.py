import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

def get_flow(frame, prev_frame, scale=.15):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_img = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    prev_img = cv2.resize(prev_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(prev_img, img, None, 0.5, 3, 5, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # max_flow = np.product(flow, axis=2)
    # print(f"product{np.max(max_flow)} flow{np.max(flow)}")
    magnitude = cv2.resize(magnitude, None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_AREA)
    return magnitude

# def update_mask(flow, mask, alpha=.2):
#
#     mask = (1-alpha)*mask + alpha*flow


def segment(image, kernel = (9,9)):
    plt.imshow(image); plt.show()
    print(type(image[0,0]))
    image = cv2.pyrMeanShiftFiltering(image, 21, 51)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # elif type(image[0,0]) != 'numpy.uint8':
        # image = cv2.normalize(image,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
        # image = np.array(image, dtype=np.uint8)
        # thresh = cv2.threshold(image.astype(np.uint8), 0, 255, np.median(image) )[1]

    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers[unknown==255] = 0
    # markers = cv2.watershed(image,markers)
    # image[markers == -1] = [255,0,0]
    return markers
cap = cv2.VideoCapture("test.mov")
cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
min_magnitude = .000001
kernel = (9,9)
print(cnt)
i = 200
cap.set(cv2.CAP_PROP_POS_FRAMES, i)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_frame = cv2.GaussianBlur(gray, (5, 5), 0)
mask = np.zeros_like(prev_frame[...,0])
alpha = .2
board = prev_frame
h,w,d = prev_frame.shape
out = cv2.VideoWriter('opt_flow.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (w,h))

def dumb_diff(frame, prev_frame, mask, alpha=.7):
    difference = cv2.absdiff(prev_frame, frame)
    _, difference = cv2.threshold(difference, 5, 255, cv2.THRESH_BINARY)
    mask = (1-alpha)*mask + alpha*difference
    return mask

# subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=5)
difference = np.zeros_like(prev_frame)
while ret:
    ret, frame = cap.read(); i+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(gray, (5, 5), 0)
    difference = delta_pix(frame, prev_frame, difference)
    prev_frame = frame
    # difference = subtractor.apply(frame)

    if not ret:
        plt.show()
        break
    # flow = get_flow(frame, prev_frame)
    # flow_mask = update_mask(flow, flow_mask)
    # frame_mask = background(frame, flow_mask)
    # image_mask = motion_mask*alpha+(1-alpha)*image_mask
    # board[frame_mask] = frame[frame_mask]
    out.write(difference)
    plt.imshow(difference)
    plt.title(i)
    plt.pause(0.0005)
    prev_frame = frame
    # print(1/(time.time()-t1))
out.release()
