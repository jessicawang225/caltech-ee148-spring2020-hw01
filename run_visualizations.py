import json
import numpy as np
from PIL import Image, ImageDraw
import os


def draw(I, boxes):
    for box in boxes:
        draw = ImageDraw.Draw(I)
        # Draw bounding box in neon yellow
        draw.rectangle(box, outline=(204, 255, 0))
        del draw
    return I


# set the path to the downloaded data:
data_path = './data'

# set a path for saving predictions:
preds_path = './predictions'

# set a path for saving visualizations:
vis_path = './visualizations'

os.makedirs(preds_path, exist_ok=True)  # create directory if needed
os.makedirs(vis_path, exist_ok=True)  # create directory if needed

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

# get bounding boxes
with open(os.path.join(preds_path, 'preds.json')) as f:
    bounding_boxes = json.load(f)

for i in range(len(file_names)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names[i]))

    I = draw(I, bounding_boxes[file_names[i]])

    I.save(os.path.join(vis_path, file_names[i]))