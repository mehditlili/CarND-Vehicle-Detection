##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_features_0.png
[image2]: ./output_images/example_features_1.png
[image3]: ./output_images/example_features_2.png
[image4]: ./output_images/test_result_0.png
[image5]: ./output_images/test_result_1.png
[image6]: ./output_images/test_result_2.png
[image7]: ./output_images/small_search_windows.png
[image8]: ./output_images/big_search_windows.png
[image9]: ./output_images/test_result_label_5.png
[image10]: ./output_images/test_result_5.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function called "train_model".  

I started by reading in all the `vehicle` and `non-vehicle` images. 
Then I extracted image features for each car and not-car images.


I then explored different color spaces and different `skimage.hog()` 
parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  
I grabbed random images from each of the two classes and displayed them to get a 
feel for what the `skimage.hog()` output looks like.
Here are three examples of car/notcar data samples using the `YCrCb` color space and HOG parameters 
of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.
The spacial binning was 32 pixels in both directions and 32 histogram bins were used.


![alt text][image1]



![alt text][image2]



![alt text][image3]



####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and following the original HOG paper, used no more than 
9 orientations.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I decided to use all features presented in the lesson: HOG, Spacial binning and Color histogram.
After testing all combinations, I came up with the result that LinearSVC reaches validation accuracy
above 99% using all features and all YCrCb color space channel.
###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I watched the videos provided in this project as well as other videos and noticed that it is safe to assume that
cars will appear in some predefined region of the image. In this case I chose this region to be:

rows: half of the image until the bottom - 30 rows (where the car itself appears)

columns: starting from column 200 from the left, everything before that is not part of the highway itself.

In this region of interest, I used two scales for sliding windows: 120 and 90.
The next figures show the regions of interests scanned by the sliding windows (Green for 90 pix windows and blue
for 120 pix windows):


![alt text][image7]
![alt text][image8]


I chose those values after trial and error. They provide a good tradeoff between computation speed and area
coverage such that both near and far cars can be detected reliably.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on two scales of sliding windows using YCrCb 3-channel HOG features plus spatially binned color and histograms 
of color in the feature vector, which provided a nice result.  
Here are some example images:

![alt text][image4]
![alt text][image5]
![alt text][image6]


I had to use all features to achieve a validation score above 99% which facilitates the task of filtering out the 
remaining few false positives (around 1 false positive per 100 samples, in my case I have 200 sliding windows in total
so I could expect false positives from time to time)
### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video
Here's a [link to my video result](https://youtu.be/IBFjwE8tFmw)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I started by creating a kind of FIFO queue to record positive detections for each frame.
I've chosen for this queue a size of 15. This queue provides the possibility of stacking up heatmaps of each frame
on top of each other continuously, and keeping at any given time the last 15 heatmaps.

My vehicle detection is then done by summing up all those heatmaps and filtering with a threshold.
I found 3 to be a robust threshhold. So the combination (15, 3) means that if in the last 15 frames, 
at position(x, y) in the image a car was detected at least 3 times, then this position is a valid candidate.
As cars do not disappear as ghosts do, having eventually 12 frames (half a second) where a car might be still
displayed as detected whereas there is no car is not a big problem (That would be a corner case where the car is 
slowly vanishing from the field of view of the camera, so having a buffer of 12 detection would even be better
because if gives the car the time to disappear totally from the image before the detection square also disappears)

(15, 3) delivered a stable detection with very few false positives (that could be even further filtered because laying in an image
area that could be excluded from my current ROI).

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()`:

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from 15 frames:

where dark blue equals zero. Other colors are positive integers (stacked up heat)
![alt text][image9]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image10]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Due to time limitation and huge amount of delay that I had due to personal reasons, I couldn't further improve
my solution to this task. So I will submit it as "good enough" hopefully to succeed.

There is some points that I would further integrate in this project if I had more time:

### Investigate more features:
I worked before a lot with SIFT, SURF, BRISK and other corner based features.
I am pretty confident that using those features would achieve a much faster and robust detection.

### Investigate more classifiers:
In my case I only tested LinearSVC and also SVC. The advantage of the standard SVS solution is that it also 
can predict with probability so that I can decrease the number of false positives by accepting a vehicle prediction
only if it has a certitude above 90% or so. using LinearSVC, a detection with 51% would count as valid.
But SVC is much slower than its linear counterpart so one has to be careful.

### Vehicle tracking:
One of the problems you might notice in my solution is that vehicle detection tends to fuse two vehicles as one
when those two vehicles get too close to each other. (see my video, around sec. 30). I am not sure how bad this 
would be for a real life vehicle detection system but using a vehicle tracking system would help remove this problem.
Having a tracker that tracks each detected vehicle would keep track of each vehicle by providing more logic that can
be used to separate the vehicles when they are fused.

### Make it real time!:
My solution ran at approximately 1Hz on my laptop so it was not a lot of fun letting it run on the 50 sec video.



 

