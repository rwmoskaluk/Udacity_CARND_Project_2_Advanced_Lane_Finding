The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `output_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

The Writeup
---

#### Camera Calibration
The code for this step is contained in the function called calibration_step().  This function only takes one parameter for visualizing output when debugging.  Otherwise it will return the following: ret, mtx, dist from calibrating the camera based on the provided calibration images.  I started with looking through the calibration images to determine the number of corners in the x and y direction, this turned out to be 9 by 6 respectively.  I then initialized object points to create an x and y matrix holding z constant at 0 as the assumption for these calibration images.  The directory, camera_cal/, was then read for all of the camera images using the glob library.  These images were then stored and processed one by one via the following steps: converting to grayscale from RGB, calling opencv's findChessboard() function, checking that the ret returned value was true for locating a chessboard of the described size, then appending these points and corners to a running list for all images, with all of the points and corners found for the given images the calibrateCamera() from opencv was then called to process and determine the calibration matrix.
![Alt text](output_images/calibration_results.png?raw=true "Calibration Results")


#### Applied Camera Calibration
The drawn chessboard images that passed successfully for being 9x6 are stored in the folder output_images/ as calibration_results.png.  Calibration1, Calibration4, and Calibration5 failed as they were 8x6.

With the reprojection error, camera matrix, and distortion coefficients from calibration step the function calibration_step_applied() handles applying these correction factors to each image.  This is done by passing the function an image, camera matrix, and camera distortion coefficients.  Using opencv's undistort function a calibrated image is generated.  This is then returned for further utilization in the main pipeline.  Corrected calibration images can be found in output_images/applied_undistort.png.  This shows from all the test images what the corrected image should look like given the camera parameters from the first step.
![Alt text](output_images/applied_undistort.png?raw=true "Calibration Applied to Images")


#### Thresholding
The next step in the lane finding pipeline is thresholding step.  This can be found under the function threshold_step().  The idea of this function is to use a combination of the following methods: HLS  color thresholding utilizing H, RGB thresholding utilizing R, Sobel X and Y, Sobel magnitude, and Sobel directional thresholding.  The color thresholding is done in the function hls_threshold().  In this function the image is broken down into HLS and RGB component matrices.  The H and R are then used to create a binary thresholded image where they equal 1 for being within the specified threshold range.  The threshold range was found via testing a few different test images.  For the Sobel gradient method in the X and Y direction the function abs_sobel_threshold() can be passed either 'x' or 'y' and return the binary threshold image in that direction.  A range of 20 to 70 was found to produce the best results from the test images.  The magnitude of the sobel is computed for images via the mag_threshold() function.  This uses a a threshold of 120 to 255 for computing the Sobel X and Y and taking the magnitude of both values.  Lastly, the directional threshold is taken using the Sobel X and Y results and computing the arctangent of the results.  The threshold range that was set for this function is 0.7 to 1.1.  The final step is combining the different methods.  The methods that are equal to eachother and then returned are where the Sobel X and Y are equal, magnitude and directional are equal, or the H and R color thresholding is equal.  The result is a binary image which can be seen in the output_images/threshold.png image.  This shows all of the different methods and the final result compared to the original image.
![Alt text](output_images/threshold.png?raw=true "Threshold Results")

#### Perspective Transform
For step 4) of the pipeline an image perspective transform was taken.  This was done to create a bird's eye view of the lane lines for further processing in the pipeline. The function perspective_transform_step() was created to handle this by passing the image file in and having it return a warped image.  The source points were determined from test images as just outside of the lane lines.  The destination points for the image transform were the corners of the image space to create the "bird's eye view".  The opencv function getPrespectiveTransform() was used here with the source and destination points.  This returned the transform matrix, M.  The inv transform was computed as well too for further use when the reverse was needed in a later step.  The image was then warped using opencv's warpPerspective() and then returned for further use in the pipeline.  The result of this process can be seen in output_images/warped_image.png
![Alt text](output_images/warped_image.png?raw=true "Warped Results")


#### Lane Detection
With the warped image from the previous step the lane lines can now be detected and drawn.  This is done in the lane_detection_step() function.  The first step to this function is fit polynomials to both the right and left lane lines in the images.  This is accomplished with the function fit_polynomial().  The fit_polynomial() function takes in the binary image and then uses a sliding window to determine where the lane lines and then fits a polynomial around that.  The left and right lane windows and centers are found in the sliding_window() function.  The histogram max value peaks are taken for the binary image to find where the lanes lines might reside.  From there a window around this region is searched and returned for the bottom half of the image.  The polyfit function is then used to fit a second order polynomial to the discovered center points.  The output image and resulting fitted image are returned from this function. 
![Alt text](output_images/lane_detected.png?raw=true "Lane Detected Results")

The final function in the processing pipeline is visualize_step().  This step takes the warped image and lane points that were found in the previous step and draws them onto the original image.  The left and right lanes are drawn and the image is dewarped to be properly overlayed onto the untouched original image that started in the beginning of the processing pipeline.
![Alt text](output_images/lane_detected_final.png?raw=true "Final Results")

#### Example output
![Alt text] (https://media.giphy.com/media/8AdjEaMFDBJ05UfGi1/giphy.gif)


## Issues/Improvements

#### Improvements
The major improvements that can be made to this pipeline are optimizations.  This pipeline takes a long time to compute the values for each frame and this can sped up a few different ways.  The first would be to reduce the search region to the last know point as the next image is processed.  If the lane was not able to be found within a certain tolerance then skip to a larger search region.  Improving the filtering technique for using successful averaged values for detected lane lines would drastically help with the first challenge video and running into different shaded pavements.  Further exploring reducing the detection window and staggering the search regions could help with a curved result for the harder challenge video.
#### Issues
Some issues I ran into were originally trying to get the perspective transform source points to come out straight when running the transform .  I believe I corrected this but I feel there could be some more improvement to it.  Another issue that arose was trying to set different thresholds for the thresholding step.  Each method is different and required some trial and error which I feel there would be a better way to handle this automatically in the future to reduce human bias.  When porting this over to run on my local machine there were initial complications with the required libraries for making the videos and plotting images.
