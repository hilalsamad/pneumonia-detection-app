import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform

import config

plt.switch_backend("agg")


def _as_pil(img):
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    return img


def _select_target_layer(model):
    """Pick the last convolutional layer of the Faster R-CNN backbone."""
    bb = getattr(model, "backbone", None)
    if bb is None:
        return None
    if hasattr(bb, "body") and hasattr(bb.body, "layer4"):
        layer4 = bb.body.layer4
        if hasattr(layer4, "__getitem__"):
            return layer4[-1].conv3 if hasattr(layer4[-1], "conv3") else layer4[-1]
        return layer4
    return bb


def draw_boxes(img, det, score_th=0.3, return_image=False):
    """Draw detection boxes on image."""
    pil = _as_pil(img)
    fig, ax = plt.subplots(1)
    ax.imshow(pil)
    ax.axis("off")

    boxes = det.get("boxes", torch.empty(0))
    labels = det.get("labels", torch.empty(0))
    scores = det.get("scores", torch.empty(0))

    if not all(isinstance(x, torch.Tensor) for x in [boxes, labels, scores]):
        st.error("Detection outputs are not tensors; check model output.")
        plt.close(fig)
        return None if not return_image else pil

    for box, label, score in zip(boxes, labels, scores):
        if float(score) >= float(score_th):
            x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            name = config.LABEL_MAP.get(int(label.item()), "N/A")
            ax.text(x1, y1 - 8, f"{name}: {float(score):.2f}",
                    color="lime", fontsize=12,
                    bbox=dict(facecolor="black", alpha=0.5))

    plt.tight_layout(pad=0)
    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)
    else:
        st.pyplot(fig)
        plt.close(fig)
        return None


def gradcam_overlay(tensor_CHW, img_resized_HWC, model, det, score_th=0.3, image_weight=0.6):
    """Generate Grad-CAM++ heatmap overlay."""
    try:
        model.eval()

        # Ensure valid inputs
        if isinstance(img_resized_HWC, Image.Image):
            img_resized_HWC = np.array(img_resized_HWC)
        if not isinstance(img_resized_HWC, np.ndarray):
            raise TypeError(f"Expected numpy array for image, got {type(img_resized_HWC)}")
        if not isinstance(tensor_CHW, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for tensor, got {type(tensor_CHW)}")

        boxes = det.get("boxes")
        labels = det.get("labels")
        scores = det.get("scores")
        if not all(isinstance(x, torch.Tensor) for x in [boxes, labels, scores]):
            st.error("Detection outputs are not tensors; check model output.")
            return img_resized_HWC

        keep = scores > float(score_th)
        if not torch.any(keep):
            st.warning("No high-confidence detections for Grad-CAM.")
            return img_resized_HWC

        tgt_boxes = boxes[keep].detach().cpu()
        tgt_labels = labels[keep].detach().cpu().tolist()

        targets = [FasterRCNNBoxScoreTarget(labels=tgt_labels, bounding_boxes=tgt_boxes)]
        target_layer = _select_target_layer(model)

        cam = GradCAMPlusPlus(
            model=model,
            target_layers=[target_layer],
            reshape_transform=fasterrcnn_reshape_transform
        )

        # Ensure tensor is float32
        if tensor_CHW.dtype != torch.float32:
            tensor_CHW = tensor_CHW.float()

        grayscale_cam = cam(input_tensor=tensor_CHW.unsqueeze(0), targets=targets)
        if grayscale_cam is None or len(grayscale_cam) == 0:
            st.warning("Grad-CAM returned empty result.")
            return img_resized_HWC

        cam_map = grayscale_cam[0, :]
        base = (img_resized_HWC / 255.0).astype(np.float32)
        cam_img = show_cam_on_image(base, cam_map, use_rgb=True, image_weight=image_weight)
        return cam_img

    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return img_resized_HWC
