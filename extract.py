import os
import argparse
# Parse xml
import xml.etree.ElementTree as ET
import numpy as np
import cv2
# One-hot encoder/decoder
#import one_hot
# Load / dump data
import pickle

data_dir = '../CROHME_extractor-master/data/CROHME_full_v2/'
version_choices = ['2011', '2012', '2013']

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset_version', required=True, help="Specify the dataset version(s) to extract.", choices=version_choices, nargs='+')
ap.add_argument('-t', '--thickness', required=False, help="Specify the thickness of extracted patterns.", default=1, type=int)
args = vars(ap.parse_args())

# Extract inkML files
all_inkml_files = []
training_inkmls = []
testing_inkmls = []
for year in args.get('dataset_version'):
    # Chose directory containing data based on dataset version selected
    working_dir = os.path.join(data_dir, 'CROHME{}_data'.format(year))
    # List folders found within working_dir
    for root, folders, files in os.walk(working_dir):
        inkml_files = [os.path.join(root, inmkl_file) for inmkl_file in files if inmkl_file.endswith('.inkml')]
        all_inkml_files += inkml_files

        if len(inkml_files) != 0:
            print('Folder:', root)
            print('Numb. of inkml files:', len(inkml_files), '\n')

for file in all_inkml_files:
    if "train" in file.lower():
        training_inkmls.append(file)

    if "test" in file.lower() and "gt" in file.lower() and not 'Prime_in_row' in file:
        testing_inkmls.append(file)

print("Total inkML files: ", len(all_inkml_files))
print("Number of training inkML files: ", len(training_inkmls))
print("Number of testing inkML files:  ", len(testing_inkmls))

def extract_data(file):
    root = ET.parse(file).getroot()

    # Extract label
    label = None
    for elm in root.find("annotation"):
        if elm.attr["type"] == "truth":
            label = elm.text
            break
    else:
        raise Exception("No label!")

    # Extract traces
    traces = {}
    for trace in root.findall("trace"):
        coords = []
        for point in trace.text.replace('\n', '').split(","):
            # Split coordinates
            point = list(filter(None, coord.split(' ')))

            # Convert to integers & round coordinates
            point = np.array([round(10000 * float(x)) for x in point])
            coords.append(point)
        traces.append(coords)

    # Put everything together
    return {"label": label, "traces": traces}

def get_img_data(traces, img_size, img_padding):
    # Compute min x and min y
    min_x = 0, min_y = 0, max_x = 100000000, max_y = 1000000000
    for trace in traces:
        for point in trace:
            if (point[0] < min_x):
                min_x = point[0]

            if (point[1] < min_y):
                min_y = point[1]

            if (point[0] > max_x):
                max_x = point[0]

            if (point[1] < max_y):
                max_y = point[1]

    trace_size = np.array([max_x - min_x, max_y - min_y])

    # Compute scaling factor
    f = 10000000000;
    for i in [0, 1]:
        _f = (img_size[i] - 2*img_padding[i])/trace_size[i]
        if (_f < f):
            f = _f

    return np.array([min_x, min_y]), trace_size, f

def draw_image(traces, img_size, img_padding):
    # Get image data
    min_point, trace_size, f = get_img_data(traces, img_size, img_padding);

    # Transform image
    for trace in traces:
        for point in trace:
            point = f * (point - min_point) + ((img_size - f * trace_size) // 2)

    # Draw image
    img = 255 * np.ones(tuple(img_size), dtype=np.uint8)
    for trace in traces:
        for i in range(1, len(trace)):
            cv2.line(img, tuple(trace[i - 1]), tuple(trace[i]), color=(0), thickness=thickness)

    return img

IMG_SIZE = np.array([512, 512])
IMG_PADDING = np.array([10, 10])

def process_files(inkml_files):
    container = []
    for file in inkml_files:
        # Extract data and draw image
        data = extract_data(file)
        img = draw_image(data, IMG_SIZE, IMG_PADDING)

        container += {"label": data["label"], "image": img}
    return container

training_data = process_files(training_inkmls)
testing_data = process_files(testing_inkmls)
