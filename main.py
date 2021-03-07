import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from scipy.spatial import distance
import tkinter as tk
from tkinter import filedialog, messagebox


#print(cv2.__version__)

# This code is to hide the main tkinter window
root = tk.Tk()
root.withdraw()

# https://stackoverflow.com/questions/177287/alert-boxes-in-python
# Message Box
messagebox.showwarning("Diffraction Pattern", "Please select an image!")

#https://stackoverflow.com/questions/9319317/quick-and-easy-file-dialog-in-python
file_path = filedialog.askopenfilename()

print(file_path)

# Import the image
img = cv2.imread(file_path)

#https://stackoverflow.com/questions/59758904/check-if-image-is-all-white-pixels-with-opencv
H, W = img.shape[:2]

#https://towardsdatascience.com/computer-vision-for-beginners-part-1-7cca775f58ef
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#if (cv2.countNonZero(img_gray) >= ((H * W))/2):
if np.mean(img_gray) >= 200:
    print('Negative')
else:
    print('Positive')
    img = cv2.bitwise_not(img)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
img_denoised = cv2.fastNlMeansDenoising(img_gray, None, 5, 7, 21)

_, thresh = cv2.threshold(img_denoised, 127, 255, cv2.THRESH_BINARY)

#https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
_, thresh_O = cv2.threshold(img_denoised, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#https://towardsdatascience.com/computer-vision-for-beginners-part-2-29b3f9151874
adap_gaussian_8 = cv2.adaptiveThreshold(img_denoised, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 95, 3)

#https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
cnts = cv2.findContours(thresh, cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

#https://stackoverflow.com/questions/58959488/cv2-drawcontours-isnt-displaying-correct-color
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
#https://docs.opencv.org/master/d4/d73/tutorial_py_contours_begin.html
#cv2.drawContours(thresh, cnts, -1, (255,0,0), 2)

#https://stackoverflow.com/questions/61541559/finding-the-contour-closest-to-image-center-in-opencv2
#find center of image and draw it (blue circle)
#cv2.circle(thresh, (int(W/2), int(H/2)), 3, (0, 0, 255), 2)

spots = []

#https://stackoverflow.com/questions/60637120/detect-circles-in-opencv
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    area = cv2.contourArea(c)
    if len(approx) > 3 and area < H*W/4:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        #cv2.circle(thresh, (int(x), int(y)), int(r), (36, 255, 12), 1)
        #https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/

        # calculate distance to image_center
        distances_to_center = (distance.euclidean((W/2, H/2), (x, y)))

        # save to a list of dictionaries
        spots.append({'contour': c, 'center': (int(x), int(y)), 'radius': int(r), 'distance_to_center': distances_to_center})

# sort the spots
sorted_spots = sorted(spots, key=lambda i: i['distance_to_center'])

#cv2.circle(thresh, sorted_spots[0]['center'], sorted_spots[0]['radius'], (127, 127, 127), 1)

#sort again by size
sorted_size_spots = sorted(sorted_spots[0:9], key=lambda i: i['radius'], reverse=True)

# for i in sorted_size_spots:
#     cv2.circle(thresh, i['center'], i['radius'], (255, 0, 0), 1)

cv2.circle(thresh, sorted_size_spots[0]['center'], sorted_size_spots[0]['radius'], (36, 255, 12), 1)

spots = []

for s in sorted_spots:
    # calculate distance to central spot
    distances_to_center = (distance.euclidean(sorted_size_spots[0]['center'], s['center']))

    #https://stackoverflow.com/questions/17418108/elegant-way-to-perform-tuple-arithmetic
    centre = tuple(np.subtract(s['center'], sorted_size_spots[0]['center']))

    # save to a list of dictionaries
    spots.append({'contour': s['contour'], 'center_center': centre, 'center': s['center'], 'radius': s['radius'], 'distance_to_center': distances_to_center})

# sort the spots
sorted_spots = sorted(spots, key=lambda i: i['distance_to_center'])
sorted_spots = sorted_spots[1:11]

print (sorted_spots)

for s in sorted_spots:
    cv2.circle(thresh, s['center'], s['radius'], (255, 0, 0), 1)


angles = []



#https://stackoverflow.com/questions/24886625/pycharm-does-not-show-plot
images = [img, img_gray, img_denoised, thresh, thresh_O, adap_gaussian_8]

fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 15))
for ind, p in enumerate(images):
    ax = axs[ind%2, ind//2]
    ax.imshow(p, cmap = 'gray')
    ax.axis('off')
plt.show()