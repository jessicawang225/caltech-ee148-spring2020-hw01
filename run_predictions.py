import os
import numpy as np
import json
from PIL import Image


def compute_bounding_box(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order).
    '''

    height, width = np.shape(I)
    I = np.array(I)
    redCoords = list(zip(*np.where(I == 2)))
    blackCoords = list(zip(*np.where(I == 1)))

    redCoords = sorted(redCoords, key=lambda element: (redCoords[0], redCoords[1]))
    blackCoords = sorted(blackCoords, key=lambda element: (blackCoords[0], blackCoords[1]))

    coords = []
    if (len(redCoords) == 0):
        return []
    left, top = redCoords[0]
    minLeft, minTop, maxRight, maxBottom = left, top, left, top
    i = 1
    while (i < len(redCoords) - 1):
        right, bottom = redCoords[i]
        if ((right > left + 3) and (bottom > top + 3)):
            maxRight = right
            maxBottom = bottom
            if (maxRight < minLeft):
                temp = minLeft
                minLeft = maxRight
                maxRight = temp
            if (maxBottom < minTop):
                temp = minTop
                minTop = maxBottom
                maxBottom = temp
            coords.append([int(minLeft), int(minTop), int(maxRight), int(maxBottom)])
            left, top = redCoords[i + 1]
            minLeft, minTop, maxRight, maxBottom = left, top, left, top
        else:
            left = right
            top = bottom
        i += 1

    for box in coords:
        left, top, right, bottom = box
        for i in range(-2, 2):
            for j in range(-2, 2):
                if (
                        0 <= left + i < width and 0 <= right + i < width and 0 <= top + j < height and 0 <= bottom + j < height):
                    if (I[left + i][top + j] == 0 or I[right + i][top + j] == 0 or I[left + i][bottom + j] == 0 or
                            I[right + i][bottom + j] == 0):
                        if box in coords:
                            coords.remove(box)

    return coords


def detect_black_and_red(I):
    '''
    This function takes a numpy array <I> and returns a 2D array.
    This 2D array has the same dimensions as the image of interest.
    A '1' is used to indicate the color of that pixel as black or
    approximately black and a '2' is used to indicate the color of that
    pixel as red or approximately red. All other pixels are indicated
    with a '0'.
    '''

    height, width, channels = np.shape(I)
    newI = [[0 for x in range(width)] for y in range(height)]

    for y in range(height):
        for x in range(width):
            r = I[y, x, 0]
            g = I[y, x, 1]
            b = I[y, x, 2]
            # Check if the pixel color is black or approximately black
            if (r < 60 and g < 60 and b < 60):
                newI[y][x] = 1
            # Check if the pixel color is red or approximately red
            if (r >= 2.5 * g and r >= 2.5 * b):
                newI[y][x] = 2

    return newI


def detect_red_light(I):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the
    image. Each element of <bounding_boxes> should itself be a list, containing
    four integers that specify a bounding box: the row and column index of the
    top left corner and the row and column index of the bottom right corner (in
    that order).

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    bounding_boxes = []  # This should be a list of lists, each of length 4. See format example below.

    newI = detect_black_and_red(I)
    bounding_boxes = compute_bounding_box(newI)

    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4

    return bounding_boxes


# set the path to the downloaded data:
data_path = './data'

# set a path for saving predictions:
preds_path = './predictions'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

preds = {}
for i in range(len(file_names)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds.json'), 'w') as f:
    json.dump(preds, f)
