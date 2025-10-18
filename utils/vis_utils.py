import torch, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import config

# Use a non-interactive backend for Matplotlib
plt.switch_backend('agg')

def draw_boxes(img, det, score_th):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    ax.axis("off")
    for box, label, score in zip(det["boxes"], det["labels"], det["scores"]):
        if score > score_th:
            box = box.cpu().numpy()
            label_name = config.LABEL_MAP[label.item()]
            x, y, x2, y2 = box
            w, h = x2 - x, y2 - y
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            plt.text(x, y-10, f"{label_name}: {score:.2f}", color='lime', fontsize=12,
                     bbox=dict(facecolor='black', alpha=0.5))
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

def gradcam_overlay(tensor, img_resized, model, det, image_weight=0.5):
    target_layers = [model.backbone]
    targets = [FasterRCNNBoxScoreTarget(labels=det["labels"], boxes=det["boxes"])]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor.unsqueeze(0), targets=targets)[0, :]
    return show_cam_on_image(img_resized / 255.0, grayscale_cam, use_rgb=True, image_weight=image_weight)