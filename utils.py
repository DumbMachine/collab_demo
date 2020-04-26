import os
import cv2
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import ImageDraw, Image
from tqdm import tqdm

category_index = pickle.load(
    open("/home/dumbmachine/code/SVMWSN/.data/category_index.pkl", "rb"))


def load_model():
    """
    Loading a simple Tensorflow ObjectDetction Api model if the user doesn't supply a model
    """
    model_name = "ssd_mobilenet_v1_coco_2018_01_28"
    path = os.path.expanduser("~")
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = os.path.join(model_dir, "saved_model")

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


def create_screenshots_from_video(video_path, nos_frames=1, verbose=1, train=True):
    """Will create the dataset from the video frames present in the video clip

    Arguments:
        video_path {str} -- filepath of the video clip

    Keyword Arguments:
        nos_frames {int} -- The number of frames to be taken from each interval (default: {2})
        fps {int} -- requried to distributre the video frames properly and avoid redundant frames (default: {24})
    """
    # creating the variables to take care of the video
    video_path = './vids/GTA 5 MOVIE 60FPS All Cutscenes Grand Theft Auto V - Full Story-JF8t4ygOZwo-25-of-32.mp4'
    breakpoint_loop = 10
    start = 0
    frames = []
    model = None
    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    end = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = tqdm(total=int(end/fps))

    for _ in range(0, end, fps):
        progress.set_description(f"Batch number {int(_/fps)}")
        frames = [video.read()[1] for _ in range(fps)]
        idxs = [int(i) for i in np.random.uniform(0, fps, 2)]
        images = [frames[i] for i in idxs]
        model = make_prediction(images, model, train=train)

        progress.update(1)
        if breakpoint_loop <= 0:
            break
        else:
            breakpoint_loop -= 1


def make_prediction(images, model=None, train=True):
    """Generate the predictions

    Arguments:
        images {list} -- list of images

    Keyword Arguments:
        model {[type]} --  The model in serving mode (default: {None})
    """
    if model is None:
        model = load_model()

    output_dicts = []
    images = np.array(images)
    input_tensor = tf.convert_to_tensor(images)
    output_dict = model(input_tensor)
    detections = output_dict.pop("num_detections").numpy()

    for i in range(len(images)):
        temp_dict = {}
        for key in output_dict.keys():
            temp_dict[key] = output_dict[key][i]

        # getting rid of the trash predictions
        num_detections = detections[i]
        temp_dict = {key: value[:int(num_detections)].numpy()
                     for key, value in temp_dict.items()}
        temp_dict['num_detections'] = num_detections
        temp_dict['detection_classes'] = temp_dict['detection_classes'].astype(
            np.int64)

        # sending the predictions for saving
        drawmage = save_bboxes(
            image=images[i],
            output_dict=temp_dict,
            train=train
        )
    return model


def save_bboxes(image, output_dict, draw=False, train=True):
    """Will save the image and the bboxes

    Arguments:
        image {[type]} -- [description]
        output_dict {[type]} -- [description]

    Keyword Arguments:
        draw {bool} -- [description] (default: {False})
    """
    # the information to be saved here
    csv_save = dict.fromkeys(["filename", "width", "height",
                            "class", "xmin", "ymin", "xmax", "ymax"])
    if train:
        csv_path = "data/annotations/train_labels.csv"
    else:
        csv_path = "data/annotations/test_labels.csv"
    try:
        CSV = pd.read_csv(csv_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        CSV = pd.DataFrame([], columns=["filename", "width", "height",
                                        "class", "xmin", "ymin", "xmax", "ymax"])
    if train:
        csv_save['filename'] = f"data/images/train/{CSV.shape[0]}.png"
    else:
        csv_save['filename'] = f"data/images/test/{CSV.shape[0]}.png"

    ret = []
    image = cv2.resize(image, (800, 600))

    csv_save['width'] = 600
    csv_save['height'] = 800

    copy_image = image.copy()
    copy_image = Image.fromarray(copy_image)
    draw = ImageDraw.Draw(copy_image)
    im_width, im_height = copy_image.size

    for cat, bbox, score in zip(
            output_dict['detection_classes'],
            output_dict['detection_boxes'],
            output_dict['detection_scores']):
        if score > 0.45:
            details = csv_save
            details['class'] = category_index[cat]['name']

            ymin, xmin, ymax, xmax = bbox
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                          ymin * im_height, ymax * im_height)

            details['ymin'], details['xmin'], details['ymax'], details['xmax'] = (
                left, right, top, bottom)

            if draw:
                draw.line([(left, top), (left, bottom), (right, bottom),
                           (right, top), (left, top)], width=4, fill="red")

            print("details", details)
            CSV.loc[CSV.shape[0]] = details

    CSV.to_csv(csv_path, index_label=False, index=False)
    cv2.imwrite(csv_save['filename'], image)
