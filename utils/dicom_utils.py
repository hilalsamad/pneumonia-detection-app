import pydicom, io
from pathlib import Path

def list_dicom_files(data_dir):
    return sorted(list(Path(data_dir).rglob("*.dcm")))

def filter_by_patient_id(paths, query):
    return [p for p in paths if query.lower() in p.stem.lower()]

def read_dicom_raw(path):
    with open(path, "rb") as f:
        raw = io.BytesIO(f.read())
    return raw, path.name

def read_dicom_metadata(path):
    dcm = pydicom.dcmread(path)
    return {
        "Patient ID": dcm.get("PatientID"),
        "Patient Name": dcm.get("PatientName"),
        "Patient Sex": dcm.get("PatientSex"),
        "Patient Age": dcm.get("PatientAge"),
        "Study Description": dcm.get("StudyDescription"),
    }