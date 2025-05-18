import os
import io
import base64
import uuid
import json
import time

from google.cloud import storage
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from PIL import ImageOps, Image 
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from flask import Flask, request, render_template, redirect, url_for

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

def generate_vis_bytes(pil_img, outputs, category_map,
                       score_thresh=0.5, mask_thresh=0.5):
    img_np = np.array(pil_img.convert("RGB"))
    boxes  = outputs['boxes'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    masks  = outputs['masks'].cpu().numpy()[:, 0]

    overlay = img_np.copy()
    for box,lab,scr,m in zip(boxes,labels,scores,masks):
        if scr<score_thresh: continue
        mask_bool = m>mask_thresh
        overlay[mask_bool] = [255,0,0]
        contours,_ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (255,255,255), 2)

    blended = cv2.addWeighted(overlay, 0.6, img_np, 0.4, 0)
    h,w = img_np.shape[:2]
    fig = plt.figure(figsize=(w/100,h/100), dpi=100)
    ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,w); ax.set_ylim(h,0); ax.axis("off")
    ax.imshow(blended)

    any_pred=False
    for box,lab,scr in zip(boxes,labels,scores):
        if scr<score_thresh: continue
        any_pred=True
        x1,y1,x2,y2 = box.astype(int)
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,
                                 edgecolor="yellow",facecolor="none")
        ax.add_patch(rect)

        label_y = y1-5 if y1-25>0 else y1+15
        txt = f"{category_map[int(lab)]}: {scr:.2f}"
        est_w = len(txt)*10
        label_x = x1 if x1+est_w<w else w-est_w-5
        ax.text(label_x, label_y, txt,
                color="yellow", fontsize=max(12,min(12,w//25)),
                backgroundcolor="black", alpha=0.8, weight="bold", clip_on=True)

    if not any_pred:
        ax.text(w//2,h//2,"⚠️ No predictions above threshold",
                color="red", fontsize=max(12,min(12,w//25)),
                ha="center",va="center", backgroundcolor="white",
                alpha=0.9,weight="bold")

    buf = io.BytesIO()
    fig.savefig(buf,format="png",bbox_inches=None,pad_inches=0)
    plt.close(fig)
    return buf.getvalue()

def download_model(bucket_name, blob_name, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

# ─── Flask setup ──────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET","change-me")
# Store results temporarily in memory for redirect
result_cache = {}

# At app startup
download_model("tacodataset", "checkpoints/best_model.pth", "model/best_model.pth")
download_model("tacodataset","checkpoints/best_model_scripted.pt", "model/best_model_scripted.pt")

# ─── Load models once ───────────────────────────────────────────────────────

WEIGHTS_PATH    = "model/best_model.pth"
QUANTIZED_PATH  = "model/best_model_scripted.pt"
ANN_PATH        = "data/annotations_train.json"

category_map    = load_category_map(ANN_PATH)


# Raw .pth
raw_model       = get_model(len(category_map)+1, WEIGHTS_PATH)

# INT8‐quantized TorchScript
quant_model     = torch.jit.load(QUANTIZED_PATH, map_location="cpu")
quant_model.eval()



# ─── Routes ────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        choice = request.form.get("model_choice", "pth")
        f = request.files.get("image")
        if not f:
            return redirect(url_for("index"))

        raw = f.read()
        mime_type = f.mimetype
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if max(img.size) / min(img.size) > 1.5:
            img = ImageOps.pad(img, (640, 640), method=Image.BICUBIC, color=(0, 0, 0))
        else:
            img = ImageOps.fit(img, (640, 640), method=Image.BICUBIC)

        img_t = to_tensor(img)
        t0 = time.time()
        with torch.no_grad():
            if choice == "pth":
                out = raw_model([img_t])[0]
            else:
                b, l, s, m = quant_model(img_t)
                out = {"boxes": b, "labels": l, "scores": s, "masks": m}
        latency = (time.time() - t0) * 1000

        # Encode images as base64 URIs
        buf1 = io.BytesIO()
        img.save(buf1, format="PNG")
        orig_uri = f"data:image/png;base64,{base64.b64encode(buf1.getvalue()).decode()}"
        buf2 = generate_vis_bytes(img, out, category_map)
        pred_uri = f"data:image/png;base64,{base64.b64encode(buf2).decode()}"

        # Store result in memory by UUID
        uid = uuid.uuid4().hex
        result_cache[uid] = {
            "orig_image": orig_uri,
            "pred_image": pred_uri,
            "latency": latency,
            "choice": choice
        }

        return redirect(url_for("index", id=uid))

    # GET
    uid = request.args.get("id")
    data = result_cache.pop(uid, {}) if uid in result_cache else {}
    return render_template("index.html",
                           orig_image=data.get("orig_image"),
                           pred_image=data.get("pred_image"),
                           latency=data.get("latency"),
                           choice=data.get("choice", "pth"))


if __name__=="__main__":
    app.run(debug=True)
