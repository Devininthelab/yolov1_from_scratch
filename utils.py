import torch
import numpy as np


def convert_to_yolo_format(target, class_mapping):
    """
        Convert annotation data from VOC format to YOLO format

        Parameters: target(dict): annotation data from VOCDetection dataset.
        class_mapping(dict): mapping from class names to integer IDs

        Returns:
        array of shape [N, 5] for N bounding boxes
        each with [class_idx, x_center, y_center, width, height]
    """
    real_width = int(target['annotation']['size']['width'])
    real_height = int(target['annotation']['size']['height'])

    boxes = []

    # trích từng object có trong ảnh: loop through each object in image
    for info in target['annotation']['object']:
        xmin = int(info['bndbox']['xmin']) / real_width #scale [0,1]
        xmax = int(info['bndbox']['xmax']) / real_width
        ymin = int(info['bndbox']['ymin']) / real_height
        ymax = int(info['bndbox']['ymax']) / real_height

        #calculate the center cooridnates, width, and height of the bouding box:
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        #get class name
        class_name = info['name']
        class_id = class_mapping[class_name]

        # append the yolo formated bbox to the list
        boxes.append([class_id, x_center, y_center, width, height])

    #convert the list of bboxes to tensor
    return np.array(boxes)


def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    """
        Calculate the Intersection over Union between bboxes

        Parameters:
        boxes_preds(tensor): Predicted bounding boxes (BATCH_SIZE, 4)  (x, y, w, h)
        boxes_labels(tensor): Ground truth bounding boxes (BATCH_SIZE, 4)
        box_format(str): Box_format, can be "midpoint" (x, y, w, h) or "corners" (x1, y1, x2, y2)

        Returns:
        tensor: Intersection over Union scores for each example
    """

    if box_format == "midpoint":
        # Calculate coordinates of top-left (x1, y1) and bottom-right (x2, y2) points for predicted boxes
        box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
        box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
        box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
        box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

        # Calculate coordinates of top-left (x1, y1) and bottom-right (x2, y2) points for ground truth boxes
        box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
        box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
        box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
        box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    # Check if the box format is "corners"
    if box_format == "corners": # coco_format: x_min, y_min, width, height
        # Extract coordinates for predicted boxes
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        # Extract coordinates for ground truth boxes
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    #compute the area of the intersection rectangle, clamp(0) to handle cases where they do not overlap (return 0)
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    #calculate the areas of the predicted and ground truth boxes
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # Calculate the Intersection over Union, adding a small epsilon to avoid division by zero
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppresion(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Perform Non Maximal Suppression of a list of bounding boxes
    Parameters:
        bboxes(list): List of bounding boxes, each represented as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold(float): IoU threshold to determine correctly_predicted bounding boxes
        threshold(float): threshold to discard predicted bouding boxes (independent of IoU)
        box_format(str): "midpoint" or "corners" to specify the format of bounding boxes
    Returns:
        list: list of bounding boxes after performing NMS with a specific IoU threshold
    """
    #check the data type of input parameter
    assert type(bboxes) == list

    #filter predicted bounding boxes based on probability threshold
    bboxes = [box for box in bboxes if box[1] > threshold]

    #sort the bounding boxes by probability in descending order
    bboxes = sorted(bboxes, key=lambda x:x[1], reverse=True)

    #list to store bounding boxes after NMS
    bboxes_after_nms = []

    #continue looping till the list of bboxes empty
    while bboxes:
        #get the bounding box with the highest probability
        chosen_box = bboxes.pop(0)

        # remove bounding boxes with IoU greater than the specified threshold with the chosen box
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold # the more iou_threshold, the more it overlaps
        ]

        # add the chosen bounding box to the list after NMS
        bboxes_after_nms.append(chosen_box)

    # return the list of bbox after NMS
    return bboxes_after_nms


bboxes = [[2, 0.8, 100, 200, 50, 60],
          [2, 0.7, 100, 200, 60, 80],
          [1, 0.5, 400, 500, 80, 100]]

print(len(non_max_suppresion(bboxes, iou_threshold = 0.61, threshold=0.6, box_format="midpoint")))

#box_a = torch.tensor([100, 200, 50, 60])
#box_b = torch.tensor([100, 200, 60, 80])
#print(intersection_over_union(boxes_preds=box_a, boxes_labels=box_b, box_format='midpoint'))