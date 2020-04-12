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

    stack = []

    for y in range(height):
        for x in range(width):
            value = I[y][x]
            if (value == 2):
                stack.append((y, x))
                left = x
                right = x
                top = y
                bottom = y
                while stack:
                    new = stack.pop()
                    newY, newX = new[0], new[1]
                    I[newY][newX] = 0
                    for i in range(-1, 1):
                        for j in range(-1, 1):
                            if (newY + i >= 0 and newY + i < height and newX + j >= 0 and newX + j < width and
                                    I[newY + i][newX + j] == 2):
                                stack.append((newY + i, newX + j))
                    if (newX + 1 > right):
                        right = newX + 1
                    if (newY + 1 > bottom):
                        bottom = newY + 1
                    if (newX - 1 < left):
                        left = newX - 1
                    if (newY - 1 < top):
                        top = newY - 1
                print(left, right, top, bottom)


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
    compute_bounding_box(newI)

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
# for i in range(len(file_names)):
for i in range(1):
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds.json'), 'w') as f:
    json.dump(preds, f)
