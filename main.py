from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import torch
from torchvision import transforms

from add_persons import add_persons
from face_detection.scrfd.detector import SCRFD
from face_alignment.alignment import norm_crop
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features

# ----------------------------------------------------
# INITIALIZATION (load models once at startup)
# ----------------------------------------------------

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face Detector
detector = SCRFD("face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# ArcFace Recognizer
recognizer = iresnet_inference(
    model_name="r100",
    path="face_recognition/arcface/weights/arcface_r100.pth",
    device=device
)

# Load DB features
images_names, images_embs = read_features("./datasets/face_features/feature")


# ----------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------
@torch.no_grad()
def get_feature(face_image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_tensor = transform(face_image).unsqueeze(0).to(device)

    emb = recognizer(face_tensor).cpu().numpy()
    emb = emb / np.linalg.norm(emb)

    return emb


def recognize(face_image):
    query_emb = get_feature(face_image)
    score, idx = compare_encodings(query_emb, images_embs)
    return images_names[idx], float(score[0])


# ----------------------------------------------------
# API 1: ADD NEW PERSONS (existing)
# ----------------------------------------------------
@app.post("/add_new_persons")
def add_new_persons():
    backup_dir = "./datasets/backup"
    add_persons_dir = "./datasets/new_persons"
    faces_save_dir = "./datasets/data"
    features_path = "./datasets/face_features/feature"

    add_persons(
        backup_dir=backup_dir,
        add_persons_dir=add_persons_dir,
        faces_save_dir=faces_save_dir,
        features_path=features_path,
    )

    return {"status": "ok", "message": "New persons added successfully!"}


# ----------------------------------------------------
# API 2: RECOGNIZE PERSON FROM IMAGE
# ----------------------------------------------------
@app.post("/recognize_image")
async def recognize_image(file: UploadFile = File(...)):
    """
    Upload an image → detect face → extract embedding → compare → return name
    """

    # Read uploaded file as bytes
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "message": "Invalid image!"}

    # Face detection
    outputs, img_info, bboxes, landmarks = detector.detect_tracking(img)

    if bboxes is None or len(bboxes) == 0:
        return {"status": "error", "message": "No face detected"}

    # Take first detected face
    face_crop = norm_crop(img, landmarks[0])

    # Recognize
    name, score = recognize(face_crop)

    return {
        "status": "ok",
        "name": name,
        "score": score
    }
