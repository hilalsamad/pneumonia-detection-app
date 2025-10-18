import io
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
import config

plt.switch_backend("agg")


def draw_boxes(img, det, score_th=0.3, return_image=False):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis("off")

    for box, label, score in zip(det["boxes"], det["labels"], det["scores"]):
        if score >= score_th:
            box = box.detach().cpu().numpy()
            label_name = config.LABEL_MAP.get(label.item(), "N/A")
            x, y, x2, y2 = box
            rect = plt.Rectangle((x, y), x2 - x, y2 - y, linewidth=2,
                                 edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 10, f"{label_name}: {score:.2f}", color='lime', fontsize=12,
                    bbox=dict(facecolor='black', alpha=0.5))

    plt.tight_layout(pad=0)
    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        return Image.open(buf)
    else:
        st.pyplot(fig)
        plt.close(fig)


def gradcam_overlay(tensor, img_resized, model, det, score_th=0.3, image_weight=0.5):
    try:
        model.eval()
        if isinstance(img_resized, Image.Image):
            img_resized = np.array(img_resized)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Tensor input must be a torch.Tensor")

        boxes, labels, scores = det["boxes"], det["labels"], det["scores"]
        keep = scores > score_th
        if not torch.any(keep):
            st.warning("No detections above threshold.")
            return img_resized

        boxes, labels = boxes[keep], labels[keep]
        targets = [FasterRCNNBoxScoreTarget(labels=labels.cpu().tolist(),
                                            bounding_boxes=boxes.cpu())]

        from torch import nn
        layer = model.backbone.body.layer4[-1].conv3 if hasattr(model.backbone.body.layer4[-1], "conv3") else model.backbone.body.layer4[-1]

        cam = GradCAMPlusPlus(model=model,
                              target_layers=[layer],
                              reshape_transform=fasterrcnn_reshape_transform)

        tensor = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
        tensor = tensor.float()

        grayscale_cam = cam(input_tensor=tensor, targets=targets)
        if grayscale_cam is None or len(grayscale_cam) == 0:
            st.warning("Grad-CAM produced no output.")
            return img_resized

        grayscale_cam = grayscale_cam[0, :]
        cam_img = show_cam_on_image((img_resized / 255.0).astype(np.float32),
                                    grayscale_cam, use_rgb=True, image_weight=image_weight)
        return cam_img
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return img_resized
