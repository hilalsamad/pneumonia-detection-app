import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import model
import utils.io_utils as io_utils
import utils.vis_utils as vis_utils
import config
from utils.dicom_utils import list_dicom_files, filter_by_patient_id, read_dicom_raw, read_dicom_metadata

st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("Pneumonia Detection (Faster R-CNN + Grad-CAM++)")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running inference on: **{device}**")
score_th = st.slider("Detection threshold", 0.0, 1.0, 0.5, 0.05)

# ---- Model ----
detector_model = None
with st.sidebar:
    path = st.text_input("Model checkpoint", "fasterrcnn_resnet50_fpn.pth")
    try:
        detector_model = model.build_model(path, device)
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

# ---- Image upload ----
uploaded = st.file_uploader("Upload image or DICOM", type=["png", "jpg", "jpeg", "dcm"])
if not uploaded:
    st.info("Please upload a file to start inference.")
    st.stop()

raw = io.BytesIO(uploaded.read())
rgb = io_utils.raw_bytes_to_rgb(raw, uploaded.name)
rgb_resized = np.array(Image.fromarray(rgb).resize(config.IMAGE_SIZE))
tensor = io_utils.preprocess(rgb_resized)
if isinstance(tensor, np.ndarray):
    tensor = torch.tensor(tensor, dtype=torch.float32)
tensor = tensor.to(device)

# ---- Run inference ----
if st.button("Run inference"):
    with torch.no_grad():
        det = detector_model([tensor])[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detections")
        vis_utils.draw_boxes(rgb_resized, det, score_th)
    with col2:
        st.subheader("Grad-CAM++")
        cam_img = vis_utils.gradcam_overlay(tensor, rgb_resized, detector_model, det, score_th=score_th)
        st.image(cam_img, use_container_width=True)

    st.subheader("Combined Overlay")
    overlay = vis_utils.gradcam_overlay(tensor, rgb_resized, detector_model, det, score_th=score_th, image_weight=0.8)
    final = vis_utils.draw_boxes(overlay, det, score_th, return_image=True)
    st.image(final, use_container_width=True)
