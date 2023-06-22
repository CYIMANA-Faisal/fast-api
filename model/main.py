import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import cv2 as cv
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import joblib

from utils.get_path_util import get_path


def show_mask(mask, ax, random_color=False):

    if random_color:

        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Extract features


def calculate_area(image):
    # Load the image

    # img = cv2.imread(file)

    img = image

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to segment the sheep
    ret, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Find contours

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the area of the largest contour
    sheep_area = 0
    largest_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > sheep_area:
            sheep_area = area
            largest_contour = cnt

    return sheep_area


def compute_texture(path):

    # Load the image

    # img = image
    img = path

    # Define the properties to compute
    properties = ['contrast', 'homogeneity', 'energy', 'correlation']

    # Define the GLCM distance and angle
    distance = 1
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    # Compute the GLCM for each property and angle
    glcm = graycomatrix(
        img, distances=[distance], angles=angles, symmetric=True, normed=True)
    # texture_features = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
    texture_features = np.hstack(
        [graycoprops(glcm, prop).ravel() for prop in properties])

    # Compute the mean of the texture features
    texture_mean = np.mean(texture_features)

    return texture_mean
    """
    YOLO

    In the task of sheep detection, Yolo V8 plays a crucial role by quickly identifying the location of the sheep in an image.
    By leveraging its efficient object detection capabilities, the model swiftly recognizes the sheep's location and proceeds
    to create a bounding box around it. This bounding box acts as a visual enclosure that encompasses the sheep,
    enabling further analysis and processing.

    To facilitate subsequent steps in the sheep detection process,
    the main center point within the bounding box is computed.
    This center point serves as a reference for various subsequent tasks,
    including the utilization of the SAM (Segment Anything Model).

    """


model_yolo = YOLO('yolov8n.pt')  # Load an official model


def get_center_position(image_path):

    image = cv.imread(image_path)
    objects = model_yolo(image, save=False, classes=[18])  # [1,2,3,4,5,6,7,8]

    for result in objects:

        boxes = result.boxes  # Boxes object for bbox outputs
        cls = boxes.cls
        class_names = 'sheep'
        output_index = cls
        class_name = 'sheep'

        if len(cls) > 0 and cls[0] == 18:

            # Get the coordinates of the bounding box
            x1, y1, x2, y2 = boxes.boxes[0][:4].tolist()
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            center_point = np.array([[center_x, center_y - 23]])

            return center_point
        """
    SAM

     Customization is implemented to the SAM (Segment Anything Model) in order to utilize the center point received from Yolo V8.
     This customization allows the SAM to use the center point as a reference for the precise input of the mask, enabling accurate segmentation of the sheep.
     Once the center point is received, the customized SAM model proceeds to generate the mask using the provided reference.

    """


def process_image(path, position_of_sheep):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = get_path("model", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sam = sam_model_registry[model_type](
        checkpoint=sam_checkpoint).to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(image)

    input_point = position_of_sheep

    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    Position = max(range(len(scores)), key=lambda d: scores[d])

    def save_segmented_image(image, mask_array):

        image = image.copy()
        mask_array = np.array(mask_array, dtype=np.uint8)
        mask_array = np.where(mask_array > 0, 255, 0)

        for h in range(mask_array.shape[0]):
            for w in range(mask_array.shape[1]):
                if mask_array[h, w] == 0:
                    for i in range(3):
                        image[h, w, i] = 0
        return image

    seg_image = save_segmented_image(image, masks[Position])

    return seg_image


""""

Building upon the pipeline developed using Yolo V8 and SAM,
to implement a function to  predict the weight of the sheep.

"""


def get_value(file):

    position_of_sheep = get_center_position(file)
    segmented_image = process_image(file, position_of_sheep)

    img = cv2.imread(file, 0)

    X_test = [calculate_area(segmented_image), compute_texture(img)]

    # ** Model Location **

    model_filename = get_path("model", "lasso_model.joblib")

    loaded_model = joblib.load(model_filename)

    array_2d = np.array(X_test).reshape(1, -1)

    y_pred = loaded_model.predict(array_2d)

    return y_pred.item()
