import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
from PIL import Image

def get_model(num_classes, checkpoint_path, device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict(model, img_path, device, score_threshold=0.5):
    img = Image.open(img_path).convert("RGB")
    img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)

    masks = outputs[0]['masks'] > 0.5
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    img_np = np.array(img)

    for i in range(len(masks)):
        if scores[i] >= score_threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()
            color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
            colored_mask = np.stack([mask * color[0, c] for c in range(3)], axis=-1)

            img_np = np.where(colored_mask > 0, img_np * 0.5 + colored_mask * 0.5, img_np)

    return img_np
