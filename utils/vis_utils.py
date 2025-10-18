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


from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def gradcam_overlay(tensor, img_resized, model, det, image_weight=0.5):
    try:
        model.eval()

        idxs = det["scores"] > 0.3
        if not torch.any(idxs):
            st.warning("No high-confidence detections found.")
            return img_resized

        labels = det["labels"][idxs].cpu().tolist()
        boxes = det["boxes"][idxs].cpu()

        targets = [FasterRCNNBoxScoreTarget(labels=labels,
                                            bounding_boxes=boxes)]

        cam = GradCAMPlusPlus(model=model,
                              target_layers=[model.backbone.body.layer4])
        grayscale_cam = cam(input_tensor=tensor.unsqueeze(0),
                            targets=targets)

        grayscale_cam = grayscale_cam[0, :]
        return show_cam_on_image(
            (img_resized / 255.0).astype(np.float32),
            grayscale_cam,
            use_rgb=True,
            image_weight=image_weight,
        )

    except Exception as e:
        st.error(f"Could not generate Grad-CAM heatmap: {e}")
        return img_resized
