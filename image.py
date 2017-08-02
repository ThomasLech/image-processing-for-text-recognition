from skimage import img_as_ubyte
from skimage.io import imread
from skimage.filters import gaussian, threshold_minimum
from skimage.morphology import square, erosion, thin

import numpy as np
import cv2

# Process requested image
def binarize(image_abs_path):

	# Convert color image (3-channel deep) into grayscale (1-channel deep)
	# We reduce image dimensionality in order to remove unrelevant features like color.
	grayscale_img = imread(image_abs_path, as_grey=True)

	# Apply Gaussian Blur effect - this removes image noise
	gaussian_blur = gaussian(grayscale_img, sigma=1)

	# Apply minimum threshold
	thresh_sauvola = threshold_minimum(gaussian_blur)

	# Convert thresh_sauvola array values to either 1 or 0 (white or black)
	binary_img = gaussian_blur > thresh_sauvola

	return binary_img

def shift(contour):

	# Get minimal X and Y coordinates
	x_min, y_min = contour.min(axis=0)[0]

	# Subtract (x_min, y_min) from every contour point
	return np.subtract(contour, [x_min, y_min])

def get_scale(cont_width, cont_height, box_size):

	ratio = cont_width / cont_height

	if ratio < 1.0:
		return box_size / cont_height
	else:
		return box_size / cont_width

def extract_patterns(image_abs_path):

	max_intensity = 1
	# Here we define the size of the square box that will contain a single pattern
	box_size = 32

	binary_img = binarize(image_abs_path)

	# Apply erosion step - make patterns thicker
	eroded_img = erosion(binary_img, selem=square(3))

	# Inverse colors: black --> white | white --> black
	# It needs to be inverted before we apply thin method
	binary_inv_img = max_intensity - eroded_img

	# Apply thinning algorithm
	thinned_img = thin(binary_inv_img)

	# Before we apply opencv method,
	# we need to convert thinned_img from scikit-image format to opencv format
	thinned_img_cv = img_as_ubyte(thinned_img)

	# Find contours
	_, contours, _ = cv2.findContours(thinned_img_cv, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

	# Sort contours from left to right (sort by bounding rectangle's X coordinate)
	contours = sorted(contours, key=lambda cont: cv2.boundingRect(cont)[0])

	# Initialize patterns array
	patterns = []

	for contour in contours:

		# Initialize blank white box that will contain a single pattern
		pattern = np.ones(shape=(box_size, box_size), dtype=np.uint8) * 255

		# Shift contour coordinates so that they are now relative to its square image
		shifted_cont = shift(contour)

		# Get size of the contour
		cont_width, cont_height = cv2.boundingRect(contour)[2:]
		# boundingRect method returns width and height values that are too big by 1 pixel
		cont_width -= 1
		cont_height -= 1

		# Get scale - we will use this scale to interpolate contour so that it fits into
		# box_size X box_size square box.
		scale = get_scale(cont_width, cont_height, box_size)

		# Interpolate contour and round coordinate values to int type
		rescaled_cont = np.floor(shifted_cont * scale).astype(dtype=np.int32)

		# Get size of the rescaled contour
		rescaled_cont_width, rescaled_cont_height = cont_width * scale, cont_height * scale

		# Get margin
		margin_x = int((box_size - rescaled_cont_width) / 2)
		margin_y = int((box_size - rescaled_cont_height) / 2)

		# Center pattern wihin a square box - we move pattern right by a proper margin
		centered_cont = np.add(rescaled_cont, [margin_x, margin_y])

		# Draw centered contour on a blank square box
		cv2.drawContours(pattern, [centered_cont], contourIdx=0, color=(0))

		patterns.append(pattern)

	return patterns
