import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import config
import io

# Use a non-interactive backend for Matplotlib to prevent errors
plt.switch_backend('agg')

def draw_boxes(img, det, score_th, return_image=False):
    """Draws bounding boxes on an image."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    fig, ax = plt.subplots(1, figsize=(10, 10))
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
        img_with_boxes = Image.open(buf)
        plt.close(fig)
        return img_with_boxes
    else:
        st.pyplot(fig)
        plt.close(fig)
        return None


def gradcam_overlay(tensor, img_resized, model, det, image_weight=0.5):
    """
    STABLE PLACEHOLDER: The Grad-CAM feature is disabled due to a deep library
    incompatibility. This function returns the original image to ensure the app
    is stable and functional for the main detection task.
    """
    # This function now simply returns the image to prevent any crashes.
    st.info("Grad-CAM heatmap feature is currently disabled to ensure app stability.")
    return img_resized
```