# Vehicle Detection Project

[//]: # (Image References)
[image1]: ./output_images/video_example.gif
[image2]: ./output_images/data_look_up.PNG
[image3]: ./output_images/scaled_features.PNG


![alt text][image1]

---

## Introduction

This project aims to detect vehicles from each frame of a video and keep a track of each one, to achieve this goal a a computer vision technique, the svm classifier approach is selected, using diferent features of color and gradient of a a region of pixels, the region of pixels are obtained using a sliding windows search that execute the algorithm within the current window.

## Goals

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Data Look Up

I downloaded the data from vehicles and non-vehicles provided by udacity, and stored them in separate folders.
I built a function *data_look()* that takes the list of paths of both cars and non cars and return a summary of the data, including a example picture of each set,  the n° of pictures for both set, and finally the shape and data type of the images.

![alt text][image2]

# Feature extraction

I tried different using only HOG, but color information improved the acurracy so then I decided to use all the techniques used in the udacity course, HOG, bin spatial and color historam. I tried using different color spaces, The L channel from *Luv*, the Saturation channel, and all the channels from *YCrCb* had good results, at the end, I kept the YcrCb as my classmates encourage to use and I was pretty satisfied with the results.

I first tried using all the images but my computer couldn't cope with the memory needed to handle all the pictures so I cutted the set to use only 100 images, with the only purpose of trying different parameters. I tried changing the pixels, per cell and orients but as I increase the numbers I noticed that the time spended also increase, and with the default values (orient = 9, pixel_per_cell=8, cell_per_block = 2, spatial_size = (32,32) and hist_bins = 32) I got good results, over 86% accuracy, then I thought that with more pictures the classifier would be more robust, and If the algorithm encounters false positives I could treat them with a later postprocessing using the heatmap approach.

# Feature Scaling

The features were then scaled with the standardScaler provided in the skimage.preprocessing library, as shown below the scaled Features are more consistent.

![alt text][image3]

# SVM Training

Then I extracted the features and train the linear SVM using the following parameters

|Parameter  |value |
|-----------|------|
|C|1.0|
|class_weight|None|
|dual|True|
|fit_intercept|True|
|intercept_scaling|1|
|loss|'squared_hinge'|
|max_iter|1000|
|multi_class|'ovr'|
|penalty|'l2'|
|random_state|None|
|tol|0.0001|

Using a AWS machine I was able to use every image handeling efficiently the amount of memory without collapsing.
The results for the test set was

True Negatives = 1804
False Positives = 11
False Negatives = 17
True Positives = 1720

The final accuracy of the classifier was 99.21%, more than enough. the remainder false postives and negatives can be filtered later.

Here is an example of 100 random pictures, as you can see, all of them were clasified correctly

![alt text][image3]
     
## Sliding Window Search

As recommended in the course, I extracted all the features first for the region of interest and then isolated all the features within a window in order to obtain a better performace.
the process was iterated with different scales, this emulates different windows sizes. 
I first tried using three scales 1.25, 1.5 and 2 but some detections were redundant and only slow down the pipeline, so I used two scales 1.25 and 1.5 and with y_limits of (400,500) and (380,600) respectively, to ensure to find out vehicles that are close to the car as well to those that are far away and appear smaller in size.

Here it can be seen the picture sliced and scaled in the Ycrcb color space as well as detections found.

![alt text][image3]

## Heatmap

In order to remove the false positives and bypass the missing positives detections a heatmap class was defined, the heatmap itself consisted on an array of the same size of the image but with only two dimensions width and height. the heatmap was initialized with zeros. Then in each frame it is updated. 
First when a vehicle is detected, a value of one is added within the region of each bounding box. this will create a heatmap, where the pixels that are contained in more than one box have higher values than the ones that are only contained inside one bounding box, and the rest of the pixels, where no vehicles, were found remained as zero.

Of course, as the video proceed, the values only increased, so my first approach to decrease the values of the pixels where no vehicles were found, was to sustract a value of 1 from the previous heatmap in each pixel where no vehicles were found.

but as the cars move along and disappear, the heatmap failed to reset to zero values, my second approach was to divide by two each pixel of the previous heatmap were no vehicles was found, this came up to be a better solution, similar to the lifetime of compounds, the value of the pixels is decreased in half at each frame, if nothing is found there, so the heatmap quickly resets itself. even if the car has been in a long time at the same area. for example if the car has been in the same area for 1024 consecutive frames (~ a minute and a half @ 10 fps) , it would take only 10 frames (1 second) to reset.

to remove any false positive from the heatmap a threshold of 3 is applied, then the image is labeled, and the final bounding boxes extracted.

---
## Results
The video of the processed project video can be seen in [this link](https://www.youtube.com/watch?v=44fYlcBOYA0)
As you can see, I blend the heatmap over the original video, and then add the bounding boxes on top of the image to enhance the vehicles detected.

## Discussion

The algorithm, overall, performs well however the quantity, position and size of every windows needs to be wisely set up, because the time it takes to analyse each window is considerably high, at first I chose many windows and it took over 10 seconds for each frame, then I significantly tighten the search region so the time diminished to more or less 3 seconds without compromise the accuracy. 
I think that the algorithm is very slow, because is nested into two for loops, in order to increase the performance, the algorithm can be implemented with parallel computing using the gpu instead of the cpu. This will have an enourmous effect. I search other approaches within the state of the art and found the YOLO (You Only Look Once) system, that was incredibly faster, other classmates chose to work with CNN classifiers instead of SVM, and had much better results, and found easily and smoothly the vehicles, I did not tried to use deep learning, because I wanted to really dive into svm classifiers, and time was scarce: at the end, probably all classifiers will be used at some point in autonomous vehicles, so is important to have experience with each one, and do not favored one over the others.
The windows regions to search are fixes, that means, that there are the same for every frame, a major improvement can be made to boost speed if the regions were dynamically changed, first search only on the areas were a car can appear (on the right, left, and far in the horizon) and once a car is found keep track of the window and search there with a error margin.

The algorithm could potentially fail with certain slopes and closed curves, because it assumes that the cars far away in the horizon are seen smaller than the ones close to the car, but with high slopes, the relative size is different as well as with close turns, therefore, in order to improve the algorithm, this has to be taken into account.
The svm is not trained to dinstinguish between different kinds of vehicles, like trucks, bikes, SUVs, Etc. 

As a next step, it would be nice to add the previous works that i've done in this course, lane finding, camera calibration, traffic signs detections onto a single pipeline.
