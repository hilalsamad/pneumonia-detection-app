import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import config
import io

plt.switch_backend("agg")


def draw_boxes(img, det, score_th, return_image=False):
    """Draw detection boxes and labels."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    ax.axis("off")

    if det and "boxes" in det:
        for box, label, score in zip(det["boxes"], det["labels"], det["scores"]):
            if score > score_th:
                box = box.detach().cpu().numpy()
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                name = config.LABEL_MAP.get(label.item(), str(label.item()))
                rect = plt.Rectangle((x1, y1), w, h,
                                     linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 10, f"{name}: {score:.2f}",
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
    Generate a working Grad-CAM-like heatmap for Faster R-CNN.
    Uses gradients from the backbone feature map with respect
    to the top detection score.
    """
    try:
        model.eval()

        # Forward through backbone (keep gradients)
        tensor = tensor.unsqueeze(0)
        for p in model.parameters():
            p.requires_grad_(False)
        model.backbone.requires_grad_(True)

        features = model.backbone(tensor)
        if isinstance(features, dict):
            features = list(features.values())[-1]
        features.retain_grad()

        # Forward through the detector head
        outputs = model(tensor)
        detections = outputs[0]
        if len(detections["boxes"]) == 0:
            st.warning("No detections found.")
            return img_resized

        # Use the top detection score for CAM
        top_idx = torch.argmax(detections["scores"])
        top_score = detections["scores"][top_idx]

        # Backward pass to get gradients
        model.zero_grad(set_to_none=True)
        top_score.backward(retain_graph=True)

        grads = features.grad
        if grads is None:
            st.warning("Gradients not found — CAM skipped.")
            return img_resized

        # Compute Grad-CAM
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * features, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.detach().cpu().numpy()

        # Resize & overlay
        cam = np.uint8(255 * cam)
        cam = np.uint8(Image.fromarray(cam).resize(img_resized.shape[:2][::-1]))
        cam = cam.astype(np.float32) / 255.0
        heatmap = plt.cm.jet(cam)[..., :3]
        overlay = (1 - image_weight) * (img_resized / 255.0) + image_weight * heatmap
        overlay = np.clip(overlay, 0, 1)
        overlay = (overlay * 255).astype(np.uint8)
        return overlay

    except Exception as e:
        st.error(f"Could not generate Grad-CAM heatmap due to an internal error: {e}")
        return img_resized
