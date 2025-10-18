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

# -------------------- App Configuration --------------------
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.title("Pneumonia Detection")
st.markdown("(Faster R-CNN + Grad-CAM++)")
st.caption("Upload or select a chest X-ray to detect pneumonia, view bounding boxes, and Grad-CAM++ heatmaps.")

device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Running inference on: **{device}**")

score_th = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

# -------------------- Model Loading --------------------
raw = None
name = None
model_missing = False
detector_model = None

with st.sidebar:
    model_path = st.text_input("Checkpoint path", "fasterrcnn_resnet50_fpn.pth")

    try:
        detector_model = model.build_model(model_path, device)
    except Exception as e:
        model_missing = True
        st.error(f"❌ Failed to load model: {e}")

if model_missing:
    st.warning("⚠️ Could not load the model. Check that the checkpoint path is correct.")
    st.stop()

# -------------------- File Input --------------------
use_local = st.checkbox("Use local DICOM folder", value=False)
if use_local:
    data_dir = st.text_input("Path to DICOM folder", "./inference_data/dcm")
    all_paths = list_dicom_files(data_dir)
    if not all_paths:
        st.error(f"No .dcm files found in {data_dir}")
        st.stop()

    query = st.text_input("🔍 Search Patient ID", "")
    dicom_paths = filter_by_patient_id(all_paths, query) if query else all_paths
    if not dicom_paths:
        st.warning("No files matched your search.")
        st.stop()

    selected_path = st.selectbox("Choose a DICOM file", dicom_paths)
    if selected_path:
        raw, name = read_dicom_raw(selected_path)
        md = read_dicom_metadata(selected_path)
        st.markdown("#### Selected File Metadata")
        for k, v in md.items():
            st.markdown(f"- **{k}:** {v or '—'}")
else:
    uploaded = st.file_uploader("Upload DICOM / PNG / JPG", type=["dcm", "png", "jpg", "jpeg"])
    if uploaded:
        raw = io.BytesIO(uploaded.read())
        name = uploaded.name

# -------------------- Inference + Visualization --------------------
if st.button("Run inference") and raw and detector_model:
    rgb = io_utils.raw_bytes_to_rgb(raw, name)
    rgb_resized = np.array(Image.fromarray(rgb).resize(config.IMAGE_SIZE))

    # Ensure numeric tensor
    tensor = io_utils.preprocess(rgb_resized)
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    tensor = tensor.to(device)

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

    st.subheader("Detections + Heatmaps")
    cam_img_overlay = vis_utils.gradcam_overlay(tensor, rgb_resized, detector_model, det, score_th=score_th, image_weight=0.8)
    final_overlay = vis_utils.draw_boxes(cam_img_overlay, det, score_th, return_image=True)
    st.image(final_overlay, use_container_width=True)

elif not raw:
    st.info("📂 Please upload a file or enable local folder mode before running inference.")
