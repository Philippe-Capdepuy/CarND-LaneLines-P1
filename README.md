# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve.jpg "Grayscale"
[image2]: ./test_images_output/solidWhiteRight.jpg "Grayscale"
[image3]: ./test_images_output/solidYellowCurve.jpg "Grayscale"
[image4]: ./test_images_output/solidYellowCurve2.jpg "Grayscale"
[image5]: ./test_images_output/solidYellowCurveLeft.jpg "Grayscale"
[image6]: ./test_images_output/whiteCarLaneSwitch.jpg "Lanes"

---

### Reflection

### 1. Pipeline Description

The pipeline works as follows:
	
* The LaneFinder class is instantiated

* On the first frame of the video, the pipeline is initialized with the frame size and the region of interest (polygon) is computed as a ratio of the image sizes, to accomodate for the resolution difference between basic videos and challenge video.

* For each frame we first identify the raw lines using the same principle and mostly same parameters as the exercices:

	* convert to grayscale

	* perform some Gaussian smoothing

	* extract edges with Canny filter

	* apply 4-sided polygon mask to keep only road parts

	* extract raw lines with Hough transform

* The extracted raw lines are then processed to identify the lanes:

	* Convert each line to a richer data structure Line

	* Filter out lines that have irrelevant angles (too horizontal)

	* Merge lines together based on angular distance and horizontal shift (by creating groups of similar lines), use a score mainly based on line length for computing a merging weight. Extend merged line to the full visible range of both original lines combined (not necessarily top to bottom of region of interest, but it generally converges to it on videos).

	* Merge in the same process with previously detected lanes with artificially high score to obtain a value similar to a moving average of the selected lines.

	- Keep only the two best scoring lanes. Store them for next iteration

- Draw the found lanes on the image. There is no significant change to the line drawing function as the 'extrapolation' (more an interpolation in our case) is done during the merging process.


### Result images

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]


### 2. Shortcomings


No color filtering is used, it could potentially be useful especially on the challenge video to extract the yellow lines on dark background and obtain easier edges in the zone where they disappear, but it's generally not very reliable.

It is assumed that the first frame will provide relevant lanes that will then be partially propagated over time.

Because of the moving average, fast changes in lane positions would be accounted for after some delay (rather short though).

Temporary loss of detection is not handled, instead the moving average will continue to propagate the last detection, and it will update when detection recovers (if not too far from last one).



### 3. Possible improvements

A possible improvement would be to keep track of other hypotheses that were not selected as a lane but that may become relevant if first detection was incorrect, or if a lane is not detected anymore. Those hypotheses would be reinforced by perceived raw lines, and attenuated when no raw lines matches them.

Locally adaptive edge detection could be useful for handling the loss of edge detecton in the challenge video. Some tuning of the raw line filter could probably help too.
