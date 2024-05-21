# Image Processing with OpenCV

Image processing is a crucial aspect of computer vision applications, and OpenCV (Open Source Computer Vision Library) stands out as one of the most powerful and versatile tools for such tasks. In this section, we'll explore various techniques and functionalities provided by OpenCV for image processing.

By the end of this section, you'll have a solid understanding of basic image processing techniques using OpenCV, laying the foundation for more advanced applications in computer vision and machine learning. Let's embark on this journey to uncover the power of OpenCV in image processing.



## Importing pakages

```copy
import pandas as pd
import numpy as np

from glob import glob

import cv2
import matplotlib.pylab as plt

plt.style.use('ggplot')
```

## Reading in Images

```copy
cat_files = glob('data/training_set/cats/*.jpg')
dog_files = glob('data/training_set/dogs/*.jpg')

img_mpl = plt.imread(cat_files[20])
img_cv2 = cv2.imread(cat_files[20])
img_mpl.shape, img_cv2.shape
```

## Image Array

```copy
pd.Series(img_mpl.flatten()).plot(kind='hist',
                                  bins=50,
                                  title='Distribution of Pixel Values')
plt.show()
```

## Display Images

```copy
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_mpl)
ax.axis('off')
plt.show()
```

## Display RGB Channels

```copy
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img_mpl[:,:,0], cmap='Reds')
axs[1].imshow(img_mpl[:,:,1], cmap='Greens')
axs[2].imshow(img_mpl[:,:,2], cmap='Blues')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[0].set_title('Red channel')
axs[1].set_title('Green channel')
axs[2].set_title('Blue channel')
plt.show()
```

## Matplotlib vs cv2 Numpy Arrays

```copy
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_cv2)
axs[1].imshow(img_mpl)
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('CV2 Image')
axs[1].set_title('Matplotlib Image')
plt.show()
```

## Converting from BGR to RGB

```copy
img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
fig, ax = plt.subplots()
ax.imshow(img_cv2_rgb)
ax.axis('off')
plt.show()
```

## Image Manipulation

```copy
img = plt.imread(dog_files[4])
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.axis('off')
plt.show()
```
## Gray Image

```copy
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_gray, cmap='Greys')
ax.axis('off')
ax.set_title('Grey Image')
plt.show()
```
## Resizing and Scaling

```copy
img_resized = cv2.resize(img, None, fx=0.25, fy=0.25)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_resized)
ax.axis('off')
plt.show()
```
## Different Size

```copy
img_resize = cv2.resize(img, (100, 200))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_resize)
ax.axis('off')
plt.show()
```
## Image resize

```copy
img_resize = cv2.resize(img, (5000, 5000), interpolation = cv2.INTER_CUBIC)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_resize)
ax.axis('off')
plt.show()
```
## Sharpen Image

```copy
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])

sharpened = cv2.filter2D(img, -1, kernel_sharpening)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(sharpened)
ax.axis('off')
ax.set_title('Sharpened Image')
plt.show()
```
## Blurring the image

```copy
kernel_3x3 = np.ones((3, 3), np.float32) / 9
blurred = cv2.filter2D(img, -1, kernel_3x3)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(blurred)
ax.axis('off')
ax.set_title('Blurred Image')
plt.show()
```
## Save Image

```copy
plt.imsave('mpl_dog.png', blurred)
cv2.imwrite('cv2_dog.png', blurred)
```

## Conclusion
We began by importing necessary packages, then delved into reading and displaying images, understanding image arrays, and manipulating images using both Matplotlib and OpenCV libraries.
This tutorial serves as a foundational guide for building more complex digit recognition models. By experimenting with different architectures, hyperparameters, and preprocessing techniques, you can further refine the model's accuracy and robustness.

Remember, the key to successful deep learning projects lies in experimentation and iteration. Continuously evaluate and refine your model to achieve the best possible results.s.
