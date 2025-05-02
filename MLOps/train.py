#!/usr/bin/env python3
# train.py

import os
import argparse
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from google.cloud import storage
import numpy as np

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

def parse_args():
    p = argparse.ArgumentParser("MaskRCNN Training Pipeline")
    p.add_argument("--train-annotations", type=str, required=True)
    p.add_argument("--val-annotations", type=str, required=True)
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--mask-thresh", type=float, default=0.5)
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--resume_from", type=str, default=None)
    return p.parse_args()

def clip_boxes_to_image(boxes, img_w, img_h):
    clipped = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        x_min = max(0, min(x_min, img_w - 1))
        y_min = max(0, min(y_min, img_h - 1))
        x_max = max(0, min(x_max, img_w - 1))
        y_max = max(0, min(y_max, img_h - 1))
        clipped.append([x_min, y_min, x_max, y_max])
    return clipped

class TacoDataset(Dataset):
    def __init__(self, images_dir, annotation_path, transforms=None, max_resolution=(4000, 4000)):
        self.images_dir = images_dir
        self.coco = COCO(annotation_path)
        self.transforms = transforms
        self.max_resolution = max_resolution
        self.image_ids = [
            img_id for img_id, info in self.coco.imgs.items()
            if info['width'] <= max_resolution[0] and info['height'] <= max_resolution[1]
        ]
        print(f"✅ Loaded {len(self.image_ids)} images under resolution {max_resolution}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_id   = self.image_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image_np = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = image_np.shape[:2]

        # load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns    = self.coco.loadAnns(ann_ids)

        # build raw boxes, labels, masks lists
        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h
            if x_max <= x_min or y_max <= y_min:
                continue
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
            masks.append(self.coco.annToMask(ann))

        # clamp boxes
        boxes = clip_boxes_to_image(boxes, img_w, img_h)

        # resize masks
        resized_masks = []
        for m in masks:
            if m.shape != (img_h, img_w):
                m_img     = Image.fromarray(m.astype(np.uint8))
                m_resized = F.resize(m_img, (img_h, img_w), interpolation=Image.NEAREST)
                m         = np.array(m_resized)
            resized_masks.append(m)

        # apply transforms if any
        if self.transforms:
            try:
                transformed = self.transforms(
                    image=image_np,
                    masks=resized_masks,
                    bboxes=boxes,
                    category_ids=labels
                )
                image = transformed['image'].float() / 255.0

                boxes  = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.as_tensor(transformed['category_ids'], dtype=torch.int64)
                masks  = torch.stack([torch.tensor(m, dtype=torch.uint8)
                                      for m in transformed['masks']])

                # filter out tiny/invalid
                keep = []
                for i, box in enumerate(boxes):
                    if (box[2]-box[0] > 1) and (box[3]-box[1] > 1):
                        keep.append(i)
                if len(keep) < boxes.size(0):
                    boxes  = boxes[keep]
                    labels = labels[keep]
                    masks  = masks[keep]

            except Exception as e:
                print(f"⚠️ Transform failed on image_id {img_id}: {e}")
                image  = F.to_tensor(Image.fromarray(image_np))
                boxes  = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                masks  = torch.as_tensor(np.stack(resized_masks), dtype=torch.uint8)
        else:
            image  = F.to_tensor(Image.fromarray(image_np))
            boxes  = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks  = torch.as_tensor(np.stack(resized_masks), dtype=torch.uint8)

        # —— Remove any degenerate boxes (zero width or height) ——
        if boxes.numel() > 0:
            widths  = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            keep    = (widths > 0) & (heights > 0)
            if keep.sum() < boxes.size(0):
                boxes  = boxes[keep]
                labels = labels[keep]
                masks  = masks[keep]

        # —— Handle empty targets ——
        if boxes.numel() == 0:
            boxes   = torch.zeros((0, 4), dtype=torch.float32)
            labels  = torch.zeros((0,),    dtype=torch.int64)
            masks   = torch.zeros((0, img_h, img_w), dtype=torch.uint8)
            areas   = torch.zeros((0,),    dtype=torch.float32)
            iscrowd = torch.zeros((0,),    dtype=torch.int64)
        else:
            areas   = torch.as_tensor([ann['area']      for ann in anns], dtype=torch.float32)
            iscrowd = torch.as_tensor([ann.get('iscrowd', 0) for ann in anns], dtype=torch.int64)

        target = {
            'boxes':    boxes,
            'labels':   labels,
            'masks':    masks,
            'image_id': torch.tensor([img_id]),
            'area':     areas,
            'iscrowd':  iscrowd
        }

        return image, target


def get_train_transform():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        ToTensorV2()
    ])

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes)
    in_feats_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feats_m, 256, num_classes)
    return model

def calculate_iou(m1, m2):
    i = (m1 & m2).sum().item()
    u = (m1 | m2).sum().item()
    return i / u if u > 0 else 0.0

def evaluate(model, loader, device, score_thresh, mask_thresh, iou_thresh):
    model.eval()
    tp = fp = fn = 0
    ious = []
    cls_correct = 0
    total_preds = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            outs = model(imgs)
            for out, tgt in zip(outs, targets):
                gt_masks = tgt['masks'].bool().to(device)
                gt_labels = tgt['labels'].to(device)
                keep = out['scores'] > score_thresh
                pmasks = (out['masks'][keep, 0] > mask_thresh)
                plabs = out['labels'][keep]
                total_preds += plabs.shape[0]
                matched = set()
                for gm, gl in zip(gt_masks, gt_labels):
                    best_i, best_iou = -1, 0
                    for pi, (pm, pl) in enumerate(zip(pmasks, plabs)):
                        if pi in matched or pl != gl:
                            continue
                        iou = calculate_iou(gm, pm)
                        if iou > best_iou:
                            best_iou, i = iou, pi
                    if best_iou >= iou_thresh:
                        tp += 1
                        ious.append(best_iou)
                        matched.add(i)
                        cls_correct += 1
                    else:
                        fn += 1
                fp += pmasks.shape[0] - len(matched)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    cls_accuracy = cls_correct / (total_preds + 1e-6)
    return prec, rec, (np.mean(ious) if ious else 0.0), cls_accuracy

def train_one_epoch(model, opt, loader, device, print_every=50):
    model.train()
    scaler = GradScaler()
    total_loss = 0
    for i, (imgs, targets) in enumerate(loader, 1):
        imgs = [img.to(device) for img in imgs]
        tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            losses = model(imgs, tgts)
            loss = sum(losses.values())

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item()
        if i % print_every == 0:
            print(f"  [batch {i:4d}/{len(loader)}] loss={loss.item():.4f}")
    return total_loss / len(loader)

from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"✅ Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_ds = TacoDataset(args.images_dir, args.train_annotations, transforms=get_train_transform())
    val_ds = TacoDataset(args.images_dir, args.val_annotations, transforms=get_val_transform())
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    num_classes = len(COCO(args.train_annotations).getCatIds()) + 1
    model = get_model(num_classes).to(device)
    if args.resume_from and os.path.isfile(args.resume_from):
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print("✅ Resumed from", args.resume_from)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_miou = 0.0

    for e in range(1, args.epochs + 1):
        print(f"\n=== Epoch {e}/{args.epochs} @ {datetime.now().strftime('%H:%M:%S')} ===")
        train_loss = train_one_epoch(model, optimizer, train_ld, device)
        scheduler.step()
        print(f"=> avg train loss: {train_loss:.4f}")

        if e % 2 == 0:
            prec_train, rec_train, miou_train, cls_acc_train = evaluate(model, train_ld, device, args.score_thresh, args.mask_thresh, args.iou_thresh)
            print(f" TRAIN  | P={prec_train:.3f} R={rec_train:.3f} mIoU={miou_train:.3f} class Accuracy={cls_acc_train:.3f}")

            
            prec, rec, miou, cls_acc = evaluate(model, val_ld, device, args.score_thresh, args.mask_thresh, args.iou_thresh)
            print(f" VALID  | P={prec:.3f} R={rec:.3f} mIoU={miou:.3f} class Accuracy={cls_acc:.3f}")



            if miou > best_miou:
                best_miou = miou
                best_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"📈 New best model saved with mIoU={miou:.4f}")

    final_path = os.path.join(args.output_dir, "best_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✔ Saved final model to {final_path}")
    # Usage
    upload_blob("tacodataset", "/workspace/data/outputs/best_model.pth", "checkpoints/best_model.pth")

if __name__ == "__main__":
    main()
