import torch
from torch import nn
import torchvision.datasets
import numpy as np
import albumentations as A  # for augmentation
from albumentations.pytorch import ToTensorV2
from utils import convert_to_yolo_format

class CustomVOCDatasets(torchvision.datasets.VOCDetection):
    def __init__(self, class_mapping, S=7, B=2, C=20, custom_transform=None, **kwargs):
        super().__init__(**kwargs)
        #initialize YOLO-specific configuration parameters
        self.S = S # grid size S x S
        self.B = B # Number of bounding box per gird
        self.C = C # number of classes
        self.class_mapping = class_mapping
        self.custom_transform = custom_transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        #custom
        boxes = convert_to_yolo_format(target, self.class_mapping)

        cords = boxes[:, 1:]
        labels = boxes[:, 0]

        #transform:resize
        if self.custom_transform:
            sample = {
                'image':np.array(image),
                'bboxes' : cords,
                'labels' : labels
            }

            sample = self.custom_transform(**sample)
            # equivalent
            # sample = self.custom_transform(image = np.array(image), bboxes = cords)...
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']

        #create an empty label matrix for YOLO ground truth
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        image = torch.as_tensor(image, dtype=torch.float32)

        #iterate through each bboxes in yolo format
        for box, class_label in zip(boxes, labels):
            #box: 4 numbers
            #class_label: 1 number

            x, y, width, height = box.tolist()
            class_label = int(class_label)

            #calculate the grid_cell(i, j) that this box belong to
            i, j = int(self.S * y), int(self.S * x) # hàng, cột; x [0, 1] all image -> x_cell[0, 7]
            x_cell, y_cell = self.S * x - j, self.S*y - i # toạ độ dạng 0.4 chẳng hạn so với vị trí của bbox (2,3)(hàng 2 cột 3)

            width_cell, height_cell = width * self.S, height * self.S

            #label_matrix: (7, 7, 30)
            #0->19: 20 class_id
            #20: conf
            # 21 -> 24:bbox 1
            # 25: conf
            # 26 -> 29: bbox2

            #if no object has been found in this specific cell (i, j) before
            if label_matrix[i, j, 20] == 0:
                #mark that an object exists in this cell
                label_matrix[i, j, 20] = 1

                #store the box coordinates as an offset from the cell boundaries
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                #set the box coordinates in the label matrix
                label_matrix[i, j, 21:25] = box_coordinates

                #set the one-hot encoding for the class label
                label_matrix[i, j, class_label] = 1

            #print(label_matrix[i,j,...])




        return image, label_matrix #target: (7, 7, 30)
    


#function for custom transform
WIDTH = 448
HEIGHT = 448

def get_transforms():
    return A.Compose([A.Resize(height=WIDTH, width=HEIGHT, p=1),
                      ToTensorV2(p=1.0)],
                     p=1.0,
                     bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0, label_fields=['labels'])
                     )


class_mapping = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


dataset = CustomVOCDatasets(root='./dataset',
                            year='2012',
                            image_set='trainval',
                            download=True,
                            class_mapping=class_mapping,
                            custom_transform=get_transforms(),
                            S=7,
                            B=2,
                            C=20)