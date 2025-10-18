import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import config
import io

plt.switch_backend("agg")


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
    Generate a Grad-CAM++ overlay for Faster R-CNN models.
    Ensures Grad-CAM sees a plain tensor, not an OrderedDict.
    """
    try:
        model.eval()

        # 1️⃣ Wrap backbone to ensure tensor output
        class BackboneTensor(torch.nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

            def forward(self, x):
                feats = self.backbone(x)
                if isinstance(feats, (dict, torch.nn.modules.container.OrderedDict)):
                    feats = list(feats.values())[-1]
                if not torch.is_tensor(feats):
                    feats = torch.as_tensor(feats, dtype=torch.float32)
                return feats

        wrapped_backbone = BackboneTensor(model.backbone)

        # 2️⃣ Custom forward accepting any args Grad-CAM passes
        def tensor_forward(x, *args, **kwargs):
            return wrapped_backbone(x)

        # 3️⃣ Instantiate Grad-CAM with safe wrapper
        cam = GradCAMPlusPlus(model=wrapped_backbone,
                              target_layers=[wrapped_backbone.backbone])
        cam.forward = tensor_forward  # override safely

        # 4️⃣ Compute Grad-CAM
        grayscale_cam = cam(input_tensor=tensor.unsqueeze(0))
        if grayscale_cam is None or len(grayscale_cam) == 0:
            st.warning("Grad-CAM generation returned an empty result.")
            return img_resized

        grayscale_cam = grayscale_cam[0, :]

        # 5️⃣ Overlay heatmap on image
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
