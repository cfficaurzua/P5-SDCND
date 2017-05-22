# Vehicle Detection Project

--

# Introduction

This project aims to detect vehicles from each frame of a video and keep a track of each one, to achieve this goal a svm classifier approach is selected, using diferent features of color and gradient of a a region of pixels, the region of pixels are obtained using a sliding windows search that execute the algorithm within the current window.

## Goals

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

I downloaded the data from vehicles and non-vehicles provided by udacity, and stored them in separate folders.
I built a function *data_look()* that takes the list of paths of both cars and non cars and return a summary of the data, including a example picture of each set,  the n° of pictures for both set, and finally the shape and data type of the images.

![alt text][image1]

I tried different color spaces, both the L channel from *Luv* and all the channels from *YCrCB* had good results, 

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

https://www.youtube.com/watch?v=44fYlcBOYA0

## Discussion

The algorithm, overall, performs well however the quantity, position and size of every windows needs to be wisely set up, because the time it takes to analyse each window is considerably high, at first I chose many windows and it took over 10 seconds for each frame, then I significantly tighten the search region so the time diminished to more or less 3 seconds without compromise the accuracy. 
I think that the algorithm is very slow, because is nested into two for loops, in order to increase the performance, the algorithm can be implemented with parallel computing using the gpu instead of the cpu. This will have an enourmous effect. I search other approaches within the state of the art and found the YOLO (You Only Look Once) system, that was incredibly faster, other classmates chose to work with CNN classifiers instead of SVM, and had much better results, and found easily and smoothly the vehicles, I did not tried to use deep learning, because I wanted to dive into svm classifiers, at the end, probably a   
