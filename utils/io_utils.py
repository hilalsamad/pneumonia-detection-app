import pydicom, numpy as np, torch
from PIL import Image
from torchvision.transforms import functional as F

def read_dicom_image(path):
    dcm = pydicom.dcmread(path)
    arr = dcm.pixel_array
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        arr = np.invert(arr)
    return arr

def raw_bytes_to_rgb(raw, name):
    name = name.lower()
    if name.endswith(".dcm"):
        dcm = pydicom.dcmread(raw)
        arr = dcm.pixel_array
        if dcm.PhotometricInterpretation == "MONOCHROME1":
            arr = np.invert(arr)
    else:
        arr = np.array(Image.open(raw))

    # Ensure array is 2D
    if len(arr.shape) > 2:
        arr = arr[:, :, 0]

    # Convert to 8-bit, stack to 3 channels for RGB
    im_8bit = (arr / arr.max() * 255).astype(np.uint8)
    return np.stack([im_8bit] * 3, axis=-1)

def preprocess(img: np.ndarray):
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img.float() / 255.0