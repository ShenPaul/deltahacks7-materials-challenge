import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


print(cv2.__version__)

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

_, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

_, thresh_O = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#https://towardsdatascience.com/computer-vision-for-beginners-part-2-29b3f9151874
adap_gaussian_8 = cv2.adaptiveThreshold(img_gray, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 95, 3)

#https://stackoverflow.com/questions/24886625/pycharm-does-not-show-plot
images = [img, img_gray, thresh, thresh_O, adap_gaussian_8]

fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15, 15))
for ind, p in enumerate(images):
    ax = axs[ind%2, ind//2]
    ax.imshow(p, cmap = 'gray')
    ax.axis('off')
plt.show()