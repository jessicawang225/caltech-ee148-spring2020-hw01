# import os
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

    if len(redCoords) == 0 or len(redCoords) == 1:
        return []

    redCoords = sorted(redCoords, key=lambda element: (redCoords[0], redCoords[1]))
    blackCoords = sorted(blackCoords, key=lambda element: (blackCoords[0], blackCoords[1]))

    # Determine the coordinates of the bounding box
    coords = []
    i = 1
    widths = set()
    heights = set()
    widths.add(redCoords[0][0])
    heights.add(redCoords[0][1])
    while i < len(redCoords):
        x, y = redCoords[i]
        if (x > max(widths) + 1 or x < min(widths) - 1) and (y > max(heights) + 1 or y < min(heights) - 1):
            left = min(widths)
            top = min(heights)
            right = max(widths)
            bottom = max(heights)
            if abs(left - right) > 3 or abs(top - bottom) > 3:
                coords.append([int(left), int(top), int(right), int(bottom)])
            widths = set()
            heights = set()
            widths.add(redCoords[i][0])
            heights.add(redCoords[i][1])
        else:
            widths.add(x)
            heights.add(y)
        i += 1

    coordsCopy = coords[:]
    for box in coordsCopy:
        left, top, right, bottom = box
        # The bounding box of a red light should be square-like
        if (right - left) >= 3 * (bottom - top) or (bottom - top) >= 3 * (right - left):
            if box in coords:
                coords.remove(box)
        # Traffic lights boxes are typically black so check that the outer edges of the bounding box are black
        for i in range(-1, 1):
            for j in range(-1, 1):
                if 0 <= left + i < width and 0 <= right + i < width and 0 <= top + j < height and 0 <= bottom + j < height:
                    if (I[left + i][top + j] == 0 and I[right + i][top + j] == 0 and I[left + i][bottom + j] == 0 and
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
    newI = [[0 for y in range(height)] for x in range(width)]

    for x in range(width):
        for y in range(height):
            r = int(I[y, x, 0])
            g = int(I[y, x, 1])
            b = int(I[y, x, 2])
            # Check if the pixel color is red or approximately red
            if r > 2.5 * g and r > 2.5 * b:
                newI[x][y] = 2
            # Check if the pixel color is black or approximately black
            elif r < 70 and g < 70 and b < 70:
                newI[x][y] = 1
            else:
                newI[x][y] = 0

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
    if i % 5 == 0:
        print('Red light image detection completed for : {}/{} images'.format(i, len(file_names)))
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds[file_names[i]] = detect_red_light(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds.json'), 'w') as f:
    json.dump(preds, f)
