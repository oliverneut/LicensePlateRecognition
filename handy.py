import cv2
import pickle
import numpy as np
import math

# transform image
def affine(frame, width, height, tl, tr, bl):
    TL = [0, 0]
    TR = [0, width]
    BL = [height, 0]

    A = np.array([
        [TL[1], TL[0], 1, 0, 0, 0],
        [0, 0, 0, TL[1], TL[0], 1],
        [TR[1], TR[0], 1, 0, 0, 0],
        [0, 0, 0, TR[1], TR[0], 1],
        [BL[1], BL[0], 1, 0, 0, 0],
        [0, 0, 0, BL[1], BL[0], 1]
    ])

    b = np.array([tl[1], tl[0], tr[1], tr[0], bl[1], bl[0]])

    H = np.linalg.lstsq(A, b, rcond=None)[0]
    TM = np.array([[H[0], H[1], H[2]], [H[3], H[4], H[5]],[0, 0, 1]])

    frame_copy = frame.copy()
    license_plate = np.zeros((height, width, 3), np.uint8)

    for i in range(height):
        for j in range(width):
            xy1 = [j, i, 1]
            result = np.matmul(TM, xy1)
            PX = int(round(result[0]))
            PY = int(round(result[1]))

            if (PY < 480) & (PX < 640) & (PX > 1) & (PY > 1):
                license_plate[i, j] = frame[PY, PX]

    return license_plate



def homography(frame, width, height, tl, tr, br, bl):
    TL = [0, 0]
    TR = [0, width - 1]
    BL = [height - 1, 0]
    BR = [height - 1, width - 1]

    A = np.array([
        [TL[1], TL[0], 1, 0, 0, 0, -tl[1] * TL[1], -tl[1] * TL[0]],
        [0, 0, 0, TL[1], TL[0], 1, -tl[0] * TL[1], -tl[0] * TL[0]],
        [TR[1], TR[0], 1, 0, 0, 0, -tr[1] * TR[1], -tr[1] * TR[0]],
        [0, 0, 0, TR[1], TR[0], 1, -tr[0] * TR[1], -tr[0] * TR[0]],
        [BL[1], BL[0], 1, 0, 0, 0, -bl[1] * BL[1], -bl[1] * BL[0]],
        [0, 0, 0, BL[1], BL[0], 1, -bl[0] * BL[1], -bl[0] * BL[0]],
        [BR[1], BR[0], 1, 0, 0, 0, -br[1] * BR[1], -br[1] * BR[0]],
        [0, 0, 0, BR[1], BR[0], 1, -br[0] * BR[1], -br[0] * BR[0]]
    ])

    b = np.array([tl[1], tl[0], tr[1], tr[0], bl[1], bl[0], br[1], br[0]])

    H = np.linalg.lstsq(A, b, rcond=None)[0]

    TM = np.array([
        [H[0], H[1], H[2]],
        [H[3], H[4], H[5]],
        [H[6], H[7], 1]
    ])

    frame_copy = frame.copy()
    license_plate = np.zeros((height, width, 3), np.uint8)

    for i in range(height):
        for j in range(width):
            xy1 = [j, i, 1]
            result = np.matmul(TM, xy1)
            P = [int(round(result[0] / result[2])), int(round(result[1] / result[2]))]
            if (P[1] < 480) & (P[0] < 640):
                license_plate[i, j] = frame[P[1], P[0]]

    return license_plate

def shear(angle, y, x):
    tangent = math.tan(angle / 2)
    new_x = round(x - y * tangent)
    new_y = y

    # shear 2
    new_y = round(new_x * math.sin(angle) + new_y)  # since there is no change in new_x according to the shear matrix

    # shear 3
    new_x = round(new_x - new_y * tangent)  # since there is no change in new_y according to the shear matrix

    return new_y, new_x


def rotate_matrix(frame, angle):
    height, width = frame.shape[0:2]

    new_height = round(abs(height * math.cos(angle)) + abs(width * math.sin(angle))) + 1
    new_width = round(abs(width * math.cos(angle)) + abs(height * math.sin(angle))) + 1

    result = np.zeros((new_height, new_width, 3), np.uint8)

    original_centre_height = round(((height + 1) / 2) - 1)
    original_centre_width = round(((width + 1) / 2) - 1)

    new_centre_height = round(((new_height + 1) / 2) - 1)
    new_centre_width = round(((new_width + 1) / 2) - 1)

    for i in range(height):
        for j in range(width):
            y = height - 1 - i - original_centre_height
            x = width - 1 - j - original_centre_width

            new_y, new_x = shear(angle, y, x)

            new_y = new_centre_height - new_y
            new_x = new_centre_width - new_x

            if(new_y < new_height) & (new_x< new_width):
                result[new_y, new_x, :] = frame[i, j, :]

    show_image(result, "Result")





def find_corners2(contour):
    hull = cv2.convexHull(contour)
    rect = np.zeros((4, 2))
    pts = []
    for pt in hull:
        pts.append(pt[0])

    s = np.sum(pts, axis=1)
    # Top-left
    rect[0] = pts[np.argmin(s)]
    # Bottom-right
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect





def verify_plate(box):
    """
    Verifies that a plate correspond to basic standards and has appropriate properties
    :param box: coordinates of the four vertices of the bounding rectangle
    :return: boolean value: True if plate is acceptable, False otherwise
    """
    rect = order_points(box)
    (tl, tr, br, bl) = rect

    # Computes the width of the plate
    lower_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    upper_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    Width = max(int(lower_width), int(upper_width))

    # Computes the height of the plate
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    Height = max(int(right_height), int(left_height))

    # Calculate aspect_ratio of the plate
    if Width and Height:
        aspect_ratio = Height / Width
    else:
        aspect_ratio = 1

    # Calculate Area of the plate
    area = cv2.contourArea(box)

    # Set conditions for an acceptable plate
    return (Width > 100) and (aspect_ratio < 0.3) and (area > 2600)

def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def plate_transform(image, box):
    """
    Transforms a inclined plate into a straight plate
    :param image: plate image
    :param box: list of the four vertices' coordinates of the plate's bounding rectangle
    :return: straightened image
    """

    # obtain the bounding rectangle's vertices and order them
    rect = order_points(box)
    (tl, tr, br, bl) = rect

    # Computes the width of the plate
    lower_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    upper_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(lower_width), int(upper_width))

    # Computes the height of the plate
    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(right_height), int(left_height))

    # Construct the set of destination points to obtain a "birds eye view" of the plate
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (width, height))

    # return the warped image
    return warped

def plate_detection(image, contours):
    """
    Detects the plate on the frame
    :param image: frame to be analyzed
    :param contours: contours retrieved of the pre-processed frame
    :return: list containing images of all plates detected
    """
    final_contours = []
    i = 0
    corner_table = np.zeros((4, 2))
    # hull_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    for cnt in contours:  # Loops and verify all contours for acceptable plates
        rect = cv2.minAreaRect(cnt) # allowed - returns the top left of the rectangle and the width adn height and angle
        box = cv2.boxPoints(rect) #
        box = np.int0(box)
        if verify_plate(box):
            corners = find_corners2(cnt)

            """corners = find_corners(cnt)
            for i in range(4):
                corner_table[i, 0] = int(corners[i, 0, 0])
                corner_table[i, 1] = int(corners[i, 0, 1])
            print(corner_table)
            for pnt in cross_points:
                cv2.circle(houghed, pnt, 5, (0,0,255), -1)
            cv2.imshow("hough",houghed)
            cv2.waitKey(0)"""

            final_contours.append(box)
        i += 1

    if not final_contours:  # Returns None if no acceptable plate found
        return None

    # localized = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(localized, final_contours, 0, (0, 255, 0), 3)

    # Transforms and straighten each acceptable contours
    plate_img = []
    for cnt in final_contours:
        plate_img.append(plate_transform(image, cnt))

        # Show each localized plates
        # show_image(plate_img[len(plate_img)-1])
    return plate_img



"""    MY CODE BELOW THIS POINT        """
""""""""""" --------------------------- "" --------------------------- "" --------------------------- "" --------------------------- "" --------------------------- """""""""""

def hough_transform(frame):
    img_shape = frame.shape
    x_max = img_shape[1]
    y_max = img_shape[0]

    # Initialize theta bounds
    theta_min = -1.0 * np.pi / 4.0
    theta_max = 3.0 * np.pi / 4.0

    # Initialize rho bounds
    rho_min = 0.0
    rho_max = math.hypot(x_max, y_max)

    # Segment theta and r into dim "bins"
    rho_dim = 180
    theta_dim = 180
    # Initialize accumulator hough_space
    hough_space = np.zeros((rho_dim, theta_dim), np.uint8)

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            point = frame[i, j]
            if (point != 0):
                for theta_i in range(theta_dim):
                    theta = theta_min + 1.0 * theta_i * (theta_max - theta_min) / theta_dim
                    rho = i * math.cos(theta) + j * math.sin(theta)
                    rho_i = int(rho_dim * rho / rho_max)
                    hough_space[rho_i, theta_i] += 1
    return hough_space

def extract_lines(hough_space, image, height, width):
    theta_min = -1.0 * np.pi / 4.0
    theta_max = 3.0 * np.pi / 4.0
    theta_dim = 180
    rho_dim = 180

    rho_min = 0.0
    rho_max = math.hypot(width, height)

    max_value = np.amax(hough_space)
    max_index = np.where(hough_space == max_value)
    print(max_value, max_index)

    rho_i = max_index[0][0]
    theta_i = max_index[1][0]

    theta = theta_min + 1.0 * theta_i * (theta_max - theta_min) / theta_dim
    print('Theta', theta)
    rho = rho_i * rho_max / rho_dim

    m = np.cos(theta) / np.sin(theta)
    b = rho / np.sin(theta)

    print(m, b)

    x0 = width / 2
    y0 = m * x0 + b

    x1 = width / 4
    y1 = m * x1 + b

    print(x0, y0, x1, y1)
    """
    for i in range(len(max_list)):
        rho = max_idx[i][0]
        theta = max_idx[i][1]
        m = np.cos(np.deg2rad(theta))/np.sin(np.deg2rad(theta))
        b = rho / np.sin(np.deg2rad(theta))
        x0 = width/2
        y0 = m*x0 + b
        x1 = width/4
        y1 = m*x1 + b
        print(y1)
        lines[i] = [x0, y0, x1, y1]
        lines = lines.astype(int)
        print(lines[i])

    for points in lines:
        cv2.line(image, (points[0], points[1]), (points[2], points[3]),(255), 2)
    cv2.imshow("Lines", image)
    cv2.waitKey(3000)

    for point in max_idx:
        rho = point[0]
        theta = point[1]"""


def yellow_mode(frame):

    # Blur the image to uniformize color of the plate
    blur = cv2.GaussianBlur(frame, (9, 9), 0)

    # Keep record of gray_scales frame for window detection

    # Convert to HSV color model
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Yellow parts extraction
    light_orange = (15, 60, 70)
    dark_orange = (37, 255, 220)
    mask = cv2.inRange(hsv_img, light_orange, dark_orange)
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    # initialize kernel for morphological operations
    kernel = np.ones((5,5),np.uint8)

    # perform opening on the mask
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    masked_opening = cv2.bitwise_and(frame, frame, mask=opening)

    # perform dilation on the mask
    dilate = cv2.dilate(opening, kernel, iterations = 1)
    LP_I = cv2.bitwise_and(frame, frame, mask=dilate)

    show_image(LP_I, "LP_I")

    # BGR to gray scale conversion
    gray = cv2.cvtColor(LP_I, cv2.COLOR_BGR2GRAY)
    show_image(gray, "gray")

    # Binarize frame with very low threshold to ease edge detection
    (thresh, binary) = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    show_image(binary, "thresh")

    # Perform canny edge detection
    edged = cv2.Canny(binary, 50, 100)

    # retrieve contours of the plate
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Change original image to gray scale
    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Localize all plates in the frame and return them in a single list
    plates =  plate_detection(gray_original, contours)

    return plates

def plate_detection(img):
    # preprocess the image
    #gray = preprocess(img)
    #show_image(gray)

    print(np.average(img[:, :, 0]))
    print(np.average(img[:, :, 1]))
    print(np.average(img[:, :, 2]))

    lower = np.array([8, 73, 85])
    upper = np.array([95, 200, 190])

    lower = np.array([8, 73, 95])
    upper = np.array([86, 166, 199])

    mask = cv2.inRange(img, lower, upper)

    show_image(mask)

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    dilation = cv2.dilate(opening, kernel, iterations=1)
    show_image(dilation)

    cnts, tree = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0,255,0), 3)
    show_image(img)

    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    cv2.drawContours(img, screenCnt, -1, (0,255,0), 3)
    show_image(img)

    plate_imgs = img
    return plate_imgs


