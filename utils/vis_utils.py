import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import config
import io

# Use a non-interactive backend for Matplotlib
plt.switch_backend('agg')

def draw_boxes(img, det, score_th, return_image=False):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis("off")

    for box, label, score in zip(det["boxes"], det["labels"], det["scores"]):
        if score > score_th:
            box = box.cpu().numpy()
            label_name = config.LABEL_MAP.get(label.item(), "N/A")
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-10, f"{label_name}: {score:.2f}", color='lime', fontsize=12,
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
        return None


def gradcam_overlay(tensor, img_resized, model, det, image_weight=0.5):
    model.eval()
    target_layers = [model.backbone]
    
    # --- THIS IS THE FIX ---
    # Find detections with scores above a threshold
    high_conf_indices = det['scores'] > 0.3
    
    # If there are no high-confidence detections, we can't create a CAM.
    # Just return the original image.
    if not torch.any(high_conf_indices):
        return img_resized

    # Filter the labels and boxes using the high-confidence indices
    high_conf_labels = det['labels'][high_conf_indices]
    high_conf_boxes = det['boxes'][high_conf_indices]

    # Create the target for Grad-CAM using only the filtered detections
    targets = [FasterRCNNBoxScoreTarget(labels=high_conf_labels, boxes=high_conf_boxes)]
    # --- END OF FIX ---

    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor.unsqueeze(0), targets=targets)
    
    if grayscale_cam is None:
        return img_resized
        
    grayscale_cam = grayscale_cam[0, :]
    
    return show_cam_on_image((img_resized / 255.0).astype(np.float32), grayscale_cam, use_rgb=True, image_weight=image_weight)

