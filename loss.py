import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 0.5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5) # batch_size, 7, 7, 30

        # Calculate Intersection over Union (IoU) for the two predicted bounding boxes
        # with the target bounding box.
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25]) #bboxes coord
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25]) #bboxes coord
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        
        # Get the box with the highest IoU among the two predictions.
        # Note that bestbox will have an index of 0 or 1, indicating which box is better.
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) #identity of object i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # Set the boxes with no objects to zero. Choose one of the two predictions
        # based on the bestbox index calculated earlier.
        box_predictions = exists_box * (
            (
                bestbox * predictions[..., 26:30] 
                + (1 - bestbox) * predictions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        # Take the square root of width and height to ensure positive values.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N,S,S,4) -> (N*S*S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box represents the confidence score of the box with the highest IoU.
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        #(N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        # N, S, S, 1 -> N, S*S
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        # N, S, S, 20 -> N*S*S, 20
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * target[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss #first two rows of loss in paper
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
