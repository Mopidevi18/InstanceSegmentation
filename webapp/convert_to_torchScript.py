import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import nms
from torch.jit import optimize_for_inference, freeze

# 1) Build your custom Mask R-CNN
def get_custom_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    return model

class JITWrapper(nn.Module):
    def __init__(self, model, score_thresh=0.5, nms_thresh=0.5):
        super().__init__()
        self.model        = model
        self.score_thresh = score_thresh
        self.nms_thresh   = nms_thresh

    def forward(self, img):
        # 1) Run the detection model: get (losses, detections)
        losses, dets = self.model([img], None)   

        # 2) Grab the first image's detections
        result = dets[0]

        # 3) Unpack
        boxes, scores, labels, masks = (
            result["boxes"], result["scores"],
            result["labels"], result["masks"]
        )

        # 4) Threshold + NMS (same as before)
        keep  = scores > self.score_thresh
        boxes = boxes[keep]; scores = scores[keep]
        labels = labels[keep]; masks = masks[keep]

        keep2 = nms(boxes, scores, self.nms_thresh)
        return boxes[keep2], labels[keep2], scores[keep2], masks[keep2]


if __name__ == "__main__":
    num_classes = 61
    model = get_custom_model(num_classes)
    model.load_state_dict(torch.load("model/best_model.pth", map_location="cpu"))
    model.eval()

    wrapper = JITWrapper(model)
    wrapper.eval()

    # 1) Script
    scripted = torch.jit.script(wrapper)

    # 2) Freeze only (no optimize_for_inference)
    scripted = torch.jit.freeze(scripted)

    # 3) Save
    torch.jit.save(scripted, "model/best_model_scripted.pt")
    print("✅ Saved frozen scripted model to best_model_scripted.pt")