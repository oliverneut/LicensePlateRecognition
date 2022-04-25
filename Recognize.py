import cv2
import numpy as np
import os

"""
In this file, you will define your own segment_and_recognize function.
To do:
    1. Segment the plates character by character
    2. Compute the distances between character images and reference character images(in the folder of 'SameSizeLetters' and 'SameSizeNumbers')
    3. Recognize the character by comparing the distances
Inputs:(One)
    1. plate_imgs: cropped plate images by Localization.plate_detection function
    type: list, each element in 'plate_imgs' is the cropped image(Numpy array)
Outputs:(One)
    1. recognized_plates: recognized plate characters
    type: list, each element in recognized_plates is a list of string(Hints: the element may be None type)
Hints:
    You may need to define other functions.
"""


def contour(frame, y, x, height, width):
	contour = []

	queue = []
	queue.append((y, x))

	while (len(queue) > 0):
		(i, j) = queue.pop()
		frame[i, j] = 0
		contour.append((i, j))

		# check if left pixel is 1 and add to queue if true
		if j > 0:
			if frame[i, j - 1] == 255:
				queue.append((i, j - 1))

		# check if right pixel is 1 and add to queue if true
		if j < width - 1:
			if frame[i, j + 1] == 255:
				queue.append((i, j + 1))

		# check if down pixel is 1 and add to queue if true
		if i < height - 1:
			if frame[i + 1, j] == 255:
				queue.append((i + 1, j))

		# check if up pixel is 1 and add to queue if true
		if i > 0:
			if frame[i - 1, j] == 255:
				queue.append((i - 1, j))

	return (frame, contour)


def find_contours(frame):
	duplicate = np.copy(frame)
	contours = []
	for y in range(duplicate.shape[0]):
		for x in range(duplicate.shape[1]):
			if duplicate[y, x] == 255:
				duplicate, cnt = contour(duplicate, y, x, duplicate.shape[0], duplicate.shape[1])
				contours.append(cnt)

	return contours


def preprocess(plate):
	gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

	# Otsu's thresholding after Gaussian filtering
	blur = cv2.GaussianBlur(gray, (3, 3), 0)

	ret3, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	height, width = plate.shape[0:2]
	mask = np.zeros((height, width), np.uint8)
	border_width = round(0.1 * height)
	mask[border_width:height - border_width, border_width:width - border_width] = 255


	masked = cv2.bitwise_and(thresh, mask, mask=None)

	# how_image(masked)
	return masked


def draw_contour(contour, plane):
	count = 0
	for point in contour:
		plane[point[0], point[1]] = (255, 255, 255)

	return plane


def find_borders(character):
	y_values = [c[0] for c in character]
	x_values = [c[1] for c in character]

	l_min = np.min(x_values)
	r_max = np.max(x_values)

	t_min = np.min(y_values)
	b_max = np.max(y_values)

	return l_min, r_max, t_min, b_max


def filter_contours(plate, contours):
	# minimum relative area of character contour is 0.04585152838427948 based on this a threshold was made
	area_list = []
	height, width = plate.shape[0:2]

	character_list = []

	for cnt in contours:
		area = len(cnt)
		relative_area = area / (height * width)
		l, r, t, b = find_borders(cnt)
		relative_width = (r - l) / width
		if relative_width < 0.20:
			if relative_area > 0.018:
				character_list.append(cnt)

	return character_list


def bounding_box(plate, characters):
	bounding_boxes = []
	height, width = plate.shape[0:2]
	blank_lp = np.zeros((height, width, 3), np.uint8)

	constant = 1

	widths = 0
	for c in characters:
		blank_lp = draw_contour(c, blank_lp)
		l, r, t, b = find_borders(c)

		l = l - constant
		r = r + constant
		t = t - constant
		b = b + constant

		widths += r - l

		box = blank_lp[t:b + 1, l:r + 1]
		bounding_boxes.append([l, box])

	avg_width = widths/len(characters)

	boxes = sorted(bounding_boxes, key=lambda x: x[0])
	idxs = []
	for i in range(1, len(boxes)):
		if boxes[i][0] - boxes[i-1][0] > avg_width * 1.6:
			idxs.append(i)
	for idx in idxs:
		boxes[idx][0] = -1

	return boxes


def resize_char(c, c_box):
	height, width = c.shape[0:2]
	scale = c_box.shape[0] / height
	resized = cv2.resize(c, None, fx=scale, fy=scale)
	return resized


def compare_chars(char, c_box):
	ref_char = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
	total = ref_char.shape[0] * ref_char.shape[1]
	both = cv2.bitwise_not(cv2.bitwise_xor(cv2.cvtColor(c_box, cv2.COLOR_BGR2GRAY), ref_char))
	# show_image(both)
	return np.sum(both) / total


def read_boxes(character_boxes):
	cwd = os.getcwd()
	directory = os.listdir(cwd + "/imageprocessingcourse/SameSizeLetters")
	#directory = os.listdir("SameSizeLetters")
	lp_text = ""

	# iterate over all detected characters in license plate
	for c_box in character_boxes:
		score_card = []

		# iterate over all LETTERS in test character set
		for filename in directory:
			if filename != ".DS_Store":
				char = cv2.imread(cwd + "/imageprocessingcourse/SameSizeLetters/" + filename)
				#char = cv2.imread("SameSizeLetters/" + filename)

				r_char = resize_char(char, c_box[1])

				# slide test character over character from license plate, and compute highest score
				intermediate_score = []
				for start in range(3):
					if start + c_box[1].shape[1] <= r_char.shape[1]:
						window = r_char[:, start:start + c_box[1].shape[1]]
						if (window.shape[1] == 0) | (window.shape[0] == 0):
							continue
						intermediate_score.append((compare_chars(window, c_box[1])))
					else:
						intermediate_score.append(0)
				score_card.append((filename[0], max(intermediate_score)))
		scores = [s[1] for s in score_card]
		if c_box[0] == -1:
			lp_text += "-"
		lp_text += score_card[np.argmax(scores)][0]

	return lp_text

def read_plate(license_plate):
	# show_image(license_plate)
	# preprocess the image and output a binary image of license plate
	binary_plate = preprocess(license_plate)

	# find contours in binary image
	contours = find_contours(binary_plate)

	# filter out the characters from the contours
	characters = filter_contours(license_plate, contours)

	if len(characters) < 4:
		return False, ""

	# create bounding boxes around contours and output to a list
	# (which is sorted from left-most character to right-most)
	character_boxes = bounding_box(license_plate, characters)

	# read the characters
	LP_text = read_boxes(character_boxes)

	return True, LP_text


def segment_and_recognize(plate_imgs):
	print("Recognition :")

	result = []
	for plate in plate_imgs:
		if plate[2]:
			LP = read_plate(plate[1])
			if LP[0]:
				result.append((plate[0], LP[1], True, plate[3]))
			else:
				result.append((plate[0], "", False, plate[3]))

	return result
