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

# Use a non-interactive backend for Matplotlib (important for Streamlit)
plt.switch_backend('agg')


def draw_boxes(img, det, score_th, return_image=False):
    """Draw bounding boxes and labels for detections above threshold."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")

    if det and "boxes" in det:
        for box, label, score in zip(det["boxes"], det["labels"], det["scores"]):
            if score > score_th:
                box = box.detach().cpu().numpy()
                label_name = config.LABEL_MAP.get(label.item(), "N/A")
                x, y, x2, y2 = box
                w, h = x2 - x, y2 - y
                rect = plt.Rectangle((x, y), w, h,
                                     linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y - 10, f"{label_name}: {score:.2f}",
                        color='lime', fontsize=12,
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
    Generate a Grad-CAM++ overlay for Faster R-CNN detections.
    Returns a NumPy RGB array with heatmap overlay.
    """
    try:
        model.eval()

        # Handle possible list or OrderedDict outputs
        if not isinstance(det, dict):
            if isinstance(det, (list, tuple)) and len(det) > 0:
                det = det[0]
            else:
                st.warning("Invalid detection output format.")
                return img_resized

        target_layers = [model.backbone]

        if "scores" not in det or "boxes" not in det:
            st.warning("No valid detection fields found.")
            return img_resized

        # Select only high-confidence detections
        high_conf_indices = det['scores'] > 0.3
        if not torch.any(high_conf_indices):
            st.warning("No high-confidence detections found to generate a heatmap.")
            return img_resized

        high_conf_labels = det['labels'][high_conf_indices]
        high_conf_boxes = det['boxes'][high_conf_indices]

        # Ensure tensors are tensors (and not lists or OrderedDicts)
        labels_t = (high_conf_labels
                    if torch.is_tensor(high_conf_labels)
                    else torch.tensor(high_conf_labels))
        boxes_t = (high_conf_boxes
                   if torch.is_tensor(high_conf_boxes)
                   else torch.tensor(high_conf_boxes))

        # Safely detach and move to CPU
        targets = [FasterRCNNBoxScoreTarget(
            labels=labels_t.detach().cpu().tolist(),
            bounding_boxes=boxes_t.detach().cpu()
        )]

        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=tensor.unsqueeze(0), targets=targets)

        if grayscale_cam is None or len(grayscale_cam) == 0:
            st.warning("Grad-CAM generation returned an empty result.")
            return img_resized

        grayscale_cam = grayscale_cam[0, :]

        # Overlay CAM on image
        heatmap_img = show_cam_on_image(
            (img_resized / 255.0).astype(np.float32),
            grayscale_cam,
            use_rgb=True,
            image_weight=image_weight
        )

        return heatmap_img

    except Exception as e:
        st.error(f"Could not generate Grad-CAM heatmap due to an internal error: {e}")
        return img_resized
