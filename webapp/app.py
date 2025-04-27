import os
import io
import base64
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import uuid
from flask import Flask, request, render_template
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from flask import session, redirect, url_for

# ─── Helpers ──────────────────────────────────────────────────────────────

def get_model(num_classes, weights_path):
    model = maskrcnn_resnet50_fpn(pretrained=False)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    sd = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(sd)
    model.eval()
    return model

def load_category_map(ann_path):
    data = json.load(open(ann_path))
    return {c['id']: c['name'] for c in data['categories']}

def generate_vis_bytes(
    pil_img, outputs, category_map,
    score_thresh=0.5, mask_thresh=0.5
):
    img_np = np.array(pil_img.convert("RGB"))
    boxes  = outputs['boxes'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    masks  = outputs['masks'].cpu().numpy()[:, 0]  # [N,H,W]

    # 1) Build one overlay image, start as a copy of the original
    overlay = img_np.copy()

    # 2) For each mask above threshold:
    for box, lab, scr, m in zip(boxes, labels, scores, masks):
        if scr < score_thresh:
            continue

        mask_bool = m > mask_thresh

        # 2a) Fill the mask area in overlay with pure red
        overlay[mask_bool] = [255, 0, 0]

        # 2b) Find contours so we can draw an outline
        #    cv2.findContours expects uint8
        contours, _ = cv2.findContours(
            (mask_bool.astype(np.uint8)),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # Draw each contour in white, thickness=2 px
        cv2.drawContours(overlay, contours, -1, (255,255,255), 2)

    # 3) Blend overlay and original once
    blended = cv2.addWeighted(overlay, 0.6, img_np, 0.4, 0)

    # 4) Create your Matplotlib figure & draw it
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(blended)

    any_pred = False
    for box, lab, scr in zip(boxes, labels, scores):
        if scr < score_thresh:
            continue
        any_pred = True
        x1,y1,x2,y2 = box.astype(int)

        # yellow bbox
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor="yellow", facecolor="none"
        )
        ax.add_patch(rect)

        # caption
        cls_name = category_map.get(int(lab), f"cls{lab}")
        ax.text(
            x1, y1-5,
            f"{cls_name}: {scr:.2f}",
            color="yellow", fontsize=14,
            backgroundcolor="black", alpha=0.8
        )

    if not any_pred:
        ax.text(
            10, 20,
            "⚠️ No predictions above threshold",
            color="red", fontsize=16,
            backgroundcolor="white"
        )

    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()

# ─── Flask setup ──────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET", "change-me")

os.makedirs(os.path.join(app.static_folder, "results"), exist_ok=True)

WEIGHTS_PATH = "maskrcnn_taco_baseline.pth"
ANN_PATH     = "data/annotations_train.json"

print("🔄 Loading model and category names…")
category_map = load_category_map(ANN_PATH)
model        = get_model(len(category_map)+1, WEIGHTS_PATH)


@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        f = request.files.get("image")
        if f:
            # read all bytes up-front
            raw = f.read()

            # 1) save the original upload
            orig_name = f"{uuid.uuid4().hex}.png"
            orig_folder = os.path.join(app.static_folder, "uploads")
            os.makedirs(orig_folder, exist_ok=True)
            orig_path = os.path.join(orig_folder, orig_name)
            with open(orig_path, "wb") as out:
                out.write(raw)

            # 2) run inference on that same bytes buffer
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            img_t = to_tensor(img)
            with torch.no_grad():
                out = model([img_t])[0]

            # 3) build your Matplotlib‐style PNG bytes
            png = generate_vis_bytes(img, out, category_map)

            # 4) save it to disk under static/results
            pred_name = f"{uuid.uuid4().hex}.png"
            pred_folder = os.path.join(app.static_folder, "results")
            os.makedirs(pred_folder, exist_ok=True)
            pred_path = os.path.join(pred_folder, pred_name)
            with open(pred_path, "wb") as out:
                out.write(png)

            # 5) stash JUST the filenames in session
            session["orig_file"] = orig_name
            session["pred_file"] = pred_name

        # redirect to GET so the browser URL stays clean
        return redirect(url_for("index"))

    # GET: pull each out exactly once so that a browser refresh
    # clears them and you see only the upload form again.
    orig = session.pop("orig_file", None)
    pred = session.pop("pred_file", None)
    return render_template("index.html",
                           orig_file=orig,
                           pred_file=pred)

if __name__ == "__main__":
    app.run(debug=True)
