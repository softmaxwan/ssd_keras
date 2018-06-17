import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from tqdm import tqdm_notebook as tqdm
import time

# Parameters
input_video_path = '/Users/boma/traffic_project_trainning/morning/AB17-0830H.avi'
output_video_path = '/Users/boma/traffic_project_trainning/output/AB17-0830H_output.mp4'

print("Video Preprossing start!")
video = cv.VideoCapture(input_video_path)
ret, test_frame = video.read()

fps = video.get(cv.CAP_PROP_FPS)
img_height = test_frame.shape[0]
img_width = test_frame.shape[1]

train_its = 500
start_frame = 0 + train_its
total_frames = 350

count = 0
while (total_frames == -1):
    ret, frame = video.read()
    if ret:
        count += 1
    else:
        total_frames = count
        break

video.release()

dilation=None
def refine_fgmask(fg_mask):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))

    # Fill any small holes
    closing = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel)

    # Remove noise
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)

    # Dilate to merge adjacent blobs
    dilation = cv.dilate(opening, kernel, iterations=2)

    # threshold
    th = dilation[dilation < 240] = 0

    fg_mask = dilation

    return fg_mask

def get_centroid(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return (cx, cy)

def contour_to_pts(contour):
    (x, y, w, h), _ = contour
    pt1 = [x, y]
    pt2 = [x + w, y]
    pt3 = [x + w, y + h]
    pt4 = [x, y + h]
    return np.array([pt1, pt2, pt3, pt4])

def countour_filter(contours):
    matches = []
    # filtering by with, height
    for (i, contour) in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(contour)
        contour_valid = (w >= min_contour_width) and (
            h >= min_contour_height)
        if not contour_valid:
            continue
        # getting center of the bounding box
        centroid = get_centroid(x, y, w, h)
        matches.append(((x, y, w, h), centroid))
    return matches


min_contour_width = 30
min_contour_height = 30

font = cv.FONT_HERSHEY_SIMPLEX
video = cv.VideoCapture(input_video_path)
output_frames = []
bg_subtractor = cv.createBackgroundSubtractorMOG2(history=train_its, detectShadows=True)
print("Video Preprossing end!")
# pbar = tqdm(total_frames)
print("Video Detectiong start!")
prev = None
count = 0
while (total_frames == -1 or len(output_frames) < total_frames):
    ret, frame = video.read()
    if ret:
        fg_mask = bg_subtractor.apply(frame, None, 0.001)
        if count >= start_frame:
            fg_mask = refine_fgmask(fg_mask)

            # finding external contours
            im, contours, hierarchy = cv.findContours(
                fg_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)

            matches = countour_filter(contours)

            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            for match in matches:
                pts = contour_to_pts(match)
                cv.polylines(frame, [pts], True, (0, 255, 0), thickness=3)

            print("Vehicles detected in current frame is {}".format(len(matches)))

            output_img = cv.putText(frame,
                                    "COUNT: {}".format(len(matches)),
                                    (10, 50),
                                    font, 2,
                                    (100, 100, 200), 3,
                                    cv.LINE_AA)
            output_frames.append(output_img)
            # pbar.update(1)
    else:
        break
    count += 1

# pbar.close()

video.release()

print("Video Detectiong end and Start to print result!")
# ### Outputs: matches or pts
#
# fourcc = cv.VideoWriter_fourcc(*'MP4V')
# out = cv.VideoWriter()
#
# opened = out.open(output_video_path, fourcc, fps, (img_width, img_height))
#
# for img in output_frames:
#     img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
#     out.write(img)
#
# out.release()
#
# fig = plt.figure(figsize=(20,12))
# plt.subplot(221),plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB)),plt.title('Original')
# # plt.subplot(222),plt.imshow(dilation),plt.title('Foreground Mask')
# plt.subplot(223),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)),plt.title('Contours')
# plt.show()
# # second phase
#
# plt.imshow(frame)
# plt.show()
#
# # Third phase
# plt.imshow(fg_mask)
# plt.show()