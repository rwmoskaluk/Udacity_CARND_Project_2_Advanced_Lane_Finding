#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
# from IPython.display import HTML
from IPython import get_ipython

ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

'''
TODO Task List:
1) Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2) Apply a distortion correction to raw images.
3) Use color transforms, gradients, etc., to create a thresholded binary image.
4) Apply a perspective transform to rectify binary image ("birds-eye view").
5) Detect lane pixels and fit to find the lane boundary.
6) Determine the curvature of the lane and vehicle position with respect to center.
7) Warp the detected lane boundaries back onto the original image.
8) Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''


def calibration_step(visualize=False):
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    nx = 9  # 9 corners inside the checkerboard in the x direction
    ny = 6  # 6 corners inside the checkerboard in the y direction
    objp = np.zeros((nx * ny, 3), np.float32)  # create object points based on checkerboard size initialized to 0
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x, y coordinates generated based on checkerboard size
    ax = []  # axis object array for visualization
    rows = 5
    cols = 4
    fig = []
    gray = []
    if visualize is True:
        fig = plt.figure(figsize=(40, 40))  # set the figure size to be 40 by 40 for viewing

    # read in the filenames for all of the calibration images using glob
    # images = glob.glob("/home/workspace/CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg")
    images = glob.glob("camera_cal/calibration*.jpg")

    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert image to grayscale for processing

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)  # find the chess board corners

        if ret is True:
            imgpoints.append(corners)
            objpoints.append(objp)

        if visualize is True:
            # plot all calibration images on subplot to visualize
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)  # draw chessboard onto images
            ax.append(fig.add_subplot(rows, cols, i + 1))  # append subplot to ax array for manipulation later
            base = os.path.basename(fname)  # strip filename from filepath
            ax[-1].set_title(base)  # set title
            plt.imshow(img)

    if visualize is True:
        # display plot with all images on it that are calibrated
        plt.savefig('output_images/calibration_results.png',
                    bbox_inches='tight')
        plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist]


def calibration_applied_step(img, mtx, dist, visualize=False):
    # step for applying calibration parameters for the camera to a raw image    
    calibrated_img = cv2.undistort(img, mtx, dist, None, mtx)

    if visualize is True:
        # images = glob.glob('test_images/*.jpg')
        ax = []  # axis object array for visualization
        rows = 1
        cols = 2
        fig = plt.figure(figsize=(40, 40))  # set the figure size to be 40 by 40 for viewing
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)
        fname = 'camera_cal/calibration2.jpg'
        img = cv2.imread(fname)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        # plot original image next to undistorted image
        # plot original image
        ax.append(fig.add_subplot(rows, cols, 1))  # append subplot to ax array for manipulation later
        base = os.path.basename(fname)  # strip filename from filepath
        ax[-1].set_title(base)  # set title
        plt.imshow(img)

        # plot undistorted image
        ax.append(fig.add_subplot(rows, cols, 2))  # append subplot to ax array for manipulation later
        ax[-1].set_title('undistorted ' + base)  # set title
        plt.imshow(dst)

        # display plot with all images on it to show comparisons
        plt.savefig('output_images/applied_undistort.png',
                    bbox_inches='tight')
        plt.show()

    return calibrated_img


def threshold_step(img, s_thresh=(170, 255), r_thresh=(200, 255), x_thresh=(20, 70), y_thresh=(20, 70),
                   mag_thresh=(120, 255), dir_thresh=(0.7, 1.1), visualize=False):
    # create roi of image for reduced search area
    # img = roi(img)
    # color transform
    binary_s = hls_threshold(img, s_thresh=s_thresh, r_thresh=r_thresh)

    # gradient transform
    sobel_kernel = 3
    gradx = abs_sobel_threshold(img, orient='x', sobel_kernel=sobel_kernel, thresh=x_thresh)
    grady = abs_sobel_threshold(img, orient='y', sobel_kernel=sobel_kernel, thresh=y_thresh)
    binary_mag = mag_threshold(img, sobel_kernel=sobel_kernel, thresh=mag_thresh)
    binary_dir = dir_threshold(img, sobel_kernel=15, thresh=dir_thresh)

    binary_combined = np.zeros_like(binary_dir)
    binary_combined[((gradx == 1) & (grady == 1)) | ((binary_mag == 1) & (binary_dir == 1)) | (binary_s == 1)] = 1

    if visualize is True:
        ax = []
        rows = 3
        cols = 4
        fig = plt.figure(figsize=(40, 40))  # set the figure size to be 70 by 70 for viewing
        fig.tight_layout()

        ax.append(fig.add_subplot(rows, cols, 1))
        ax[-1].set_title("S and R threshold")
        plt.imshow(binary_s)

        ax.append(fig.add_subplot(rows, cols, 2))
        ax[-1].set_title("Gradient x")
        plt.imshow(gradx, cmap='gray')

        ax.append(fig.add_subplot(rows, cols, 3))
        ax[-1].set_title("Gradient y")
        plt.imshow(grady, cmap='gray')

        ax.append(fig.add_subplot(rows, cols, 4))
        ax[-1].set_title("Magnitude Gradient")
        plt.imshow(binary_mag, cmap='gray')

        ax.append(fig.add_subplot(rows, cols, 5))
        ax[-1].set_title("Directional Gradient")
        plt.imshow(binary_dir, cmap='gray')

        ax.append(fig.add_subplot(rows, cols, 6))
        ax[-1].set_title("Combined (S, R, Sobelx, Sobely, MagSobel, Directional) Threshold")
        plt.imshow(binary_combined, cmap='gray')

        ax.append(fig.add_subplot(rows, cols, 7))
        ax[-1].set_title("Original Image")
        b, g, r = cv2.split(img)  # get b,g,r
        rgb_img = cv2.merge([r, g, b])  # switch it to rgb
        plt.imshow(rgb_img)

        plt.savefig('output_images/threshold.png', bbox_inches='tight')
        plt.show()

    return binary_combined


def hls_threshold(img, s_thresh=(0, 255), r_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)  # convert image into hls channels
    # h = hls[:, :, 0]
    # l = hls[:, :, 1]
    s = hls[:, :, 2]
    r = img[:, :, 0]
    # g = img[:, :, 1]
    # b = img[:, :, 2]

    binary_s = np.zeros_like(s)
    binary_s[((s > s_thresh[0]) & (s <= s_thresh[1])) | ((r > r_thresh[0]) & (r <= r_thresh[1]))] = 1

    return binary_s


def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = 0
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.absolute(sobel)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Threshold x gradient
    binary_sobel = np.zeros_like(scaled_sobel)
    binary_sobel[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_sobel


def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # scale_factor = np.max(gradmag)/255

    binary_mag = np.zeros_like(gradmag)
    binary_mag[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    return binary_mag


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale for processing
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_dir = np.zeros_like(absgraddir)
    binary_dir[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_dir


def perspective_transform_step(img, visualize=False):
    img_size = (img.shape[1], img.shape[0])
    # src = np.float32([[515, 480], [785, 480], [1250, img.shape[0]], [50, img.shape[0]]])
    src = np.float32([[475, 500], [800, 500], [1230, img.shape[0]], [50, img.shape[0]]])
    dst = np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]])

    m = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)
    warped_img = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)

    if visualize is True:
        ax = []
        rows = 1
        cols = 2
        fig = plt.figure(figsize=(40, 40))  # set the figure size to be 40 by 40 for viewing
        fig.tight_layout()

        ax.append(fig.add_subplot(rows, cols, 1))
        ax[-1].set_title("original image")
        plt.imshow(img)

        ax.append(fig.add_subplot(rows, cols, 2))
        ax[-1].set_title("perspective transform image")
        plt.imshow(warped_img)

        plt.savefig('output_images/warped_image.png', bbox_inches='tight')
        plt.show()

    return warped_img, m, minv


def lane_detection_step(img, visualize=False):
    # detects lane boundaries and fits lines to them
    output_img, ploty_px, leftx_fit_px, rightx_fit_px, leftx_fit_cr, rightx_fit_cr, center = fit_polynomial(img)
    left_curve_radius, right_curve_radius = measure_curvature(ploty_px, leftx_fit_cr, rightx_fit_cr)
    # fig = []
    if visualize is True:
        print(left_curve_radius, 'm', right_curve_radius, 'm')
        print(center, 'm')
        # fig = plt.figure(figsize=(40, 40))  # set the figure size to be 40 by 40 for viewing
        # Plots the left and right polynomials on the lane lines
        plt.plot(leftx_fit_px, ploty_px, color='yellow')
        plt.plot(rightx_fit_px, ploty_px, color='yellow')
        plt.imshow(output_img)
        plt.savefig('output_images/lane_detected.png', bbox_inches='tight')
        plt.show()

    return leftx_fit_px, rightx_fit_px, ploty_px, center, left_curve_radius, right_curve_radius


def sliding_window(img, nwindows=9, margin=100, minpix=50):
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    output_img = np.dstack((img, img, img))
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(img.shape[0] // nwindows)
    nonzero = img.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = output_img.shape[0] - (window + 1) * window_height
        win_y_high = output_img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(output_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 120, 0), 2)
        cv2.rectangle(output_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 120, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, output_img


def fit_polynomial(img, ym_per_pix=15 / 720, xm_per_pix=3.7 / 950):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = sliding_window(img)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # corrected for real world units in meters
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # calculate center
    img_center_x = img.shape[1] / 2
    center = measure_lane_center(left_fit, right_fit, img_center_x)

    # Generate x and y values for plotting
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Visualization
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [120, 0, 0]
    out_img[righty, rightx] = [0, 0, 120]

    return out_img, ploty, left_fitx, right_fitx, left_fit_cr, right_fit_cr, center


def measure_curvature(ploty, left_fit, right_fit, ym_per_pix=15 / 720, xm_per_pix=3.7 / 950):
    # calculate the curvature of the fitted polynomials to the lane lines

    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curve_radius = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curve_radius = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix +
                                right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curve_radius, right_curve_radius


def measure_lane_center(left_fit, right_fit, img_center_x, xm_per_pix=3.7 / 950):
    # calculates the lane center based on the detected lane edges
    # take fit values and measure distance between both sides
    # compute the center of that distance
    # compare to center of image
    # convert to meters from pixel space and this is distance offset
    y_eval = 720
    a = left_fit[0] * (y_eval ** 2) + left_fit[1] * y_eval + left_fit[2]
    b = right_fit[0] * (y_eval ** 2) + right_fit[1] * y_eval + right_fit[2]
    calculated_center = (a + b)/2

    center = (img_center_x - calculated_center) * xm_per_pix

    return center


def visualize_step(img, warped, minv, left_fitx, right_fitx, ploty, visualize=False):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    if visualize is True:
        plt.imshow(result)
        plt.savefig('output_images/lane_detected_final.png', bbox_inches='tight')
        plt.show()

    return result


def detection_checker():
    # checks to see if detection was good
    # 1) Parallel check between lane lines
    #   1a) If failed use previous frame information
    # 2) Store last N detections
    #   2a) Average filter these for smoothing
    # 3) Narrow search space of next frame immediately based on previous frame success
    #   3a) If in between window not inline then utilize previous and future window areas to interpolate
    pass


def part_1_testing():
    # runs the testing for the pipeline
    [ret, mtx, dist] = calibration_step(visualize=False)
    print("ret (reprojection error): {}".format(ret))
    print("mtx (camera matrix): {}".format(mtx))
    print("dist (distortion coefficients): {}".format(dist))
    print("Finished Calibration")
    # img = cv2.imread('/home/workspace/CarND-Advanced-Lane-Lines/test_images/straight_lines2.jpg')
    img = cv2.imread('test_images/test1.jpg')
    calibrated_img = calibration_applied_step(img, mtx, dist, visualize=False)
    thresh_img = threshold_step(calibrated_img, visualize=False)
    warped_img, m, minv = perspective_transform_step(thresh_img, visualize=True)
    leftx_fit_px, rightx_fit_px, ploty_px, center, left_curve, right_curve = lane_detection_step(warped_img,
                                                                                                 visualize=True)
    result = visualize_step(calibrated_img, warped_img, minv, leftx_fit_px, rightx_fit_px, ploty_px, visualize=True)
    pipeline(img)
    print('Worked!')


def part_2_production():
    # runs the videos in production utilizing the pipeline
    video_output = 'output_videos/project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4").subclip(0, 4)
    detected_clip = clip1.fl_image(pipeline)  # NOTE: this function expects color images!!
    # get_ipython().run_line_magic('time', 'detected_clip.write_videofile(video_output, audio=False)')
    detected_clip.write_videofile(video_output)


def part_optional_production():
    # runs the videos in production utilizing the pipeline
    video_output = '/home/workspace/CarND-Advanced-Lane-Lines/output_videos/challenge_video.mp4'
    clip1 = VideoFileClip("/home/workspace/CarND-Advanced-Lane-Lines/challenge_video.mp4")
    detected_clip = clip1.fl_image(pipeline)  # NOTE: this function expects color images!!
    # get_ipython().run_line_magic('time', 'detected_clip.write_videofile(video_output, audio=False)')
    detected_clip.write_videofile(video_output)


def pipeline(img, first_pass=True):
    # the pipeline for image processing and lane detection
    if first_pass:
        [ret, mtx, dist] = calibration_step(visualize=False)
        first_pass = False

    calibrated_img = calibration_applied_step(img, mtx, dist, visualize=False)
    thresh_img = threshold_step(calibrated_img, visualize=False)
    warped_img, m, minv = perspective_transform_step(thresh_img, visualize=False)
    leftx_fit_px, rightx_fit_px, ploty_px, center, left_curve, right_curve = lane_detection_step(warped_img,
                                                                                                 visualize=False)
    result = visualize_step(calibrated_img, warped_img, minv, leftx_fit_px, rightx_fit_px, ploty_px, visualize=False)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2

    center_txt = "center = {0:.2f} m".format(center)
    left_curve_txt = "left curvature = {0:.2f} m".format(left_curve)
    right_curve_txt = "right curvature = {0:.2f} m".format(right_curve)
    txt = center_txt + " " + left_curve_txt + " " + right_curve_txt
    cv2.putText(result, txt, (30, 40), font, font_scale, font_color, line_type)

    return result


if __name__ == "__main__":
    part_1_testing()
    # part_2_production()
    # part_optional_production()
