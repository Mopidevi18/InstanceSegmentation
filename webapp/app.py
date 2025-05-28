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
from google.auth import default
from google.auth import impersonated_credentials

import torch
from PIL import ImageOps, Image 
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from flask import Flask, request, render_template, redirect, url_for

# ─── Helpers ───

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
    if max(w, h) > 1000:
        fontsize = min(64, max(36, w // 30))
    else:
        fontsize = max(8, min(20, w // 50))

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
                color="yellow", fontsize = fontsize,
                backgroundcolor="black", alpha=0.8, weight="bold", clip_on=True)

    if not any_pred:
        ax.text(w//2,h//2,"⚠️ No predictions above threshold",
                color="red", fontsize = fontsize,
                ha="center",va="center", backgroundcolor="white",
                alpha=0.9,weight="bold")

    buf = io.BytesIO()
    # Step 1: Save figure as PNG to a temporary buffer
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", bbox_inches=None, pad_inches=0)
    plt.close(fig)

    # Step 2: Convert to JPEG using Pillow
    buf_png.seek(0)
    img = Image.open(buf_png).convert("RGB")  # Ensure no alpha
    buf_jpeg = io.BytesIO()
    img.save(buf_jpeg, format="JPEG", quality=85)

    return buf_jpeg.getvalue()


def download_model(bucket_name, blob_name, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def upload_to_bucket(bucket_name, blob_name, data):
    # Get default credentials and impersonate the service account
    source_credentials, _ = default()

    target_service_account = "segwaste-mlops@instancesegmentation-456922.iam.gserviceaccount.com"
    target_credentials = impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=target_service_account,
        target_scopes=["https://www.googleapis.com/auth/devstorage.read_write"],
        lifetime=300
    )

    client = storage.Client(credentials=target_credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type="image/jpeg")

    return blob.generate_signed_url(
        version="v4",
        expiration=3600,
        method="GET",
        credentials=target_credentials
    )

# ─── Flask setup ───

app = Flask(__name__, template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET","change-me")
result_cache = {}

download_model("tacodataset", "checkpoints/best_model.pth", "model/best_model.pth")
download_model("tacodataset","checkpoints/best_model_scripted.pt", "model/best_model_scripted.pt")

WEIGHTS_PATH    = "model/best_model.pth"
QUANTIZED_PATH  = "model/best_model_scripted.pt"
ANN_PATH        = "data/annotations_train.json"

category_map    = load_category_map(ANN_PATH)
raw_model       = get_model(len(category_map)+1, WEIGHTS_PATH)
quant_model     = torch.jit.load(QUANTIZED_PATH, map_location="cpu")
quant_model.eval()

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

        if choice == "pth":
            img_input = img.copy()
            img_t = to_tensor(img_input)
        else:
            if max(img.size) / min(img.size) > 1.5:
                img_resized = ImageOps.pad(img, (640, 640), method=Image.BICUBIC, color=(0, 0, 0))
            else:
                img_resized = ImageOps.fit(img, (640, 640), method=Image.BICUBIC)
            img_input = img_resized
            img_t = to_tensor(img_resized)

        t0 = time.time()
        with torch.no_grad():
            if choice == "pth":
                out = raw_model([img_t])[0]
            else:
                b, l, s, m = quant_model(img_t)
                out = {"boxes": b, "labels": l, "scores": s, "masks": m}
        latency = (time.time() - t0) * 1000

        # Upload images to GCS
        pred_buf = generate_vis_bytes(img_input, out, category_map)
        uid = uuid.uuid4().hex
        pred_uri = upload_to_bucket("tacodataset", f"results/{uid}_pred.jpg", pred_buf)

        orig_buf = io.BytesIO()
        img_input.save(orig_buf, format="JPEG", quality=85)
        orig_uri = upload_to_bucket("tacodataset", f"results/{uid}_orig.jpg", orig_buf.getvalue())

        result_cache[uid] = {
            "orig_image": orig_uri,
            "pred_image": pred_uri,
            "latency": latency,
            "choice": choice
        }

        return redirect(url_for("index", id=uid) + "#inference-result")

    uid = request.args.get("id")
    data = result_cache.pop(uid, {}) if uid in result_cache else {}
    return render_template("index.html",
                           orig_image=data.get("orig_image"),
                           pred_image=data.get("pred_image"),
                           latency=data.get("latency"),
                           choice=data.get("choice", "pth"))

if __name__=="__main__":
    app.run(debug=True)
