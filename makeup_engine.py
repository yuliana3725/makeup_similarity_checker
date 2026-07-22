import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================
# PRETRAINED MODEL
# =========================
cnn_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)


# =========================
# MEDIAPIPE FACE MESH
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# =========================
# LANDMARK AREA MAKEUP
# =========================
REGIONS = {
    "Lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "Left Eye": [33, 7, 163, 144, 145, 153, 154, 155, 133],
    "Right Eye": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "Left Eyebrow": [70, 63, 105, 66, 107],
    "Right Eyebrow": [336, 296, 334, 293, 300],
    "Left Cheek": [50, 101, 118, 117, 111, 123],
    "Right Cheek": [280, 330, 347, 346, 340, 352]
}


# Bobot area untuk nilai total
WEIGHTS = {
    "Lips": 0.20,
    "Left Eye": 0.15,
    "Right Eye": 0.15,
    "Left Eyebrow": 0.10,
    "Right Eyebrow": 0.10,
    "Left Cheek": 0.15,
    "Right Cheek": 0.15
}


# Nama area versi user-friendly
REGION_LABELS = {
    "Lips": "Bibir",
    "Left Eye": "Mata kiri",
    "Right Eye": "Mata kanan",
    "Left Eyebrow": "Alis kiri",
    "Right Eyebrow": "Alis kanan",
    "Left Cheek": "Pipi kiri",
    "Right Cheek": "Pipi kanan"
}


# =========================
# PREPROCESSING
# =========================
def preprocess_image(image, prefix="referensi"):

    if image is None:
        os.makedirs("hasil_preprocessing", exist_ok=True)
        raise ValueError("Gambar tidak ditemukan.")

    # Pastikan format gambar uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
        

    # =========================
    # 1. Resize Image
    # =========================
    # Menyeragamkan ukuran gambar
    image = cv2.resize(image, (600, 600))
    

    # =========================
    # 2. Noise Reduction
    # =========================
    # Mengurangi noise pada citra
    image = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )
    

    # =========================
    # 3. Contrast Enhancement
    # =========================
    # Menggunakan CLAHE untuk
    # meningkatkan kontras citra
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced_lab = cv2.merge((l, a, b))

    image = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )
    

    # =========================
    # 4. Sharpening
    # =========================
    # Mempertajam detail makeup
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    image = cv2.filter2D(
        image,
        -1,
        sharpening_kernel
    )

    return image
    


# =========================
# DETEKSI LANDMARK
# =========================
def get_landmarks(image, prefix="referensi"):

    results = face_mesh.process(image)

    if not results.multi_face_landmarks:
        return None


    # Salin gambar asli
    drawing = image.copy()

    # ==========================
    # GAMBAR DETEKSI WAJAH
    # ==========================
    mp.solutions.drawing_utils.draw_landmarks(
        image=drawing,
        landmark_list=results.multi_face_landmarks[0],
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
    )


    h, w, _ = image.shape
    landmarks = []

    for lm in results.multi_face_landmarks[0].landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks.append((x, y))

    return landmarks


# =========================
# ROI + MASK
# =========================
def extract_roi_and_mask(image, landmarks, indices):
    points = np.array([landmarks[i] for i in indices], dtype=np.int32)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)

    masked_image = cv2.bitwise_and(image, image, mask=mask)

    x, y, w, h = cv2.boundingRect(points)
    roi = masked_image[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]

    if roi.size == 0 or roi_mask.size == 0:
        return None, None

    roi = cv2.resize(roi, (224, 224))
    roi_mask = cv2.resize(roi_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    return roi, roi_mask


# =========================
# FEATURE EXTRACTION CNN
# =========================
def extract_feature(roi):
    if roi is None:
        return None

    roi = roi.astype(np.float32)
    roi = np.expand_dims(roi, axis=0)
    roi = preprocess_input(roi)

    feature = cnn_model.predict(roi, verbose=0)
    return feature.flatten()


# =========================
# CNN SIMILARITY
# =========================
def calculate_cnn_similarity(feature_ref, feature_user):
    if feature_ref is None or feature_user is None:
        return 0.0

    score = cosine_similarity([feature_ref], [feature_user])[0][0]
    score = max(0.0, min(1.0, score))

    return round(score * 100, 2)


# =========================
# COLOR SIMILARITY
# =========================
def calculate_color_similarity(ref_roi, user_roi, ref_mask, user_mask):
    ref_lab = cv2.cvtColor(ref_roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    user_lab = cv2.cvtColor(user_roi, cv2.COLOR_RGB2LAB).astype(np.float32)

    valid = (ref_mask > 0) & (user_mask > 0)

    if np.sum(valid) == 0:
        return 0.0

    ref_mean = np.mean(ref_lab[valid], axis=0)
    user_mean = np.mean(user_lab[valid], axis=0)

    distance = np.linalg.norm(ref_mean - user_mean)

    # Semakin kecil jarak warna, semakin tinggi similarity
    score = max(0.0, 100.0 - distance)
    return round(score, 2)


# =========================
# HISTOGRAM SIMILARITY
# =========================
def calculate_histogram_similarity(ref_roi, user_roi, ref_mask, user_mask):
    ref_lab = cv2.cvtColor(ref_roi, cv2.COLOR_RGB2LAB)
    user_lab = cv2.cvtColor(user_roi, cv2.COLOR_RGB2LAB)

    total_score = 0.0

    for channel in range(3):
        ref_hist = cv2.calcHist([ref_lab], [channel], ref_mask, [32], [0, 256])
        user_hist = cv2.calcHist([user_lab], [channel], user_mask, [32], [0, 256])

        ref_hist = cv2.normalize(ref_hist, ref_hist).flatten()
        user_hist = cv2.normalize(user_hist, user_hist).flatten()

        corr = cv2.compareHist(
            ref_hist.astype(np.float32),
            user_hist.astype(np.float32),
            cv2.HISTCMP_CORREL
        )

        corr = max(0.0, min(1.0, (corr + 1) / 2))
        total_score += corr

    return round((total_score / 3) * 100, 2)


# =========================
# TEXTURE SIMILARITY
# =========================
def calculate_texture_similarity(ref_roi, user_roi):
    ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_RGB2GRAY)
    user_gray = cv2.cvtColor(user_roi, cv2.COLOR_RGB2GRAY)

    score = ssim(ref_gray, user_gray, data_range=255)
    score = max(0.0, min(1.0, score))

    return round(score * 100, 2)


# =========================
# SHAPE SIMILARITY
# =========================
def calculate_shape_similarity(ref_mask, user_mask):
    ref_bin = ref_mask > 0
    user_bin = user_mask > 0

    intersection = np.logical_and(ref_bin, user_bin).sum()
    union = np.logical_or(ref_bin, user_bin).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return round(iou * 100, 2)


# =========================
# STATUS SIMILARITY
# =========================
def get_status(score):
    if score >= 85:
        return "Sangat mirip"
    elif score >= 70:
        return "Cukup mirip"
    else:
        return "Perlu disempurnakan"


# =========================
# ANALISIS PER AREA
# =========================
def analyze_region(ref_img, user_img, ref_landmarks, user_landmarks, region_name, indices):
    ref_roi, ref_mask = extract_roi_and_mask(ref_img, ref_landmarks, indices)
    user_roi, user_mask = extract_roi_and_mask(user_img, user_landmarks, indices)

    if ref_roi is None or user_roi is None:
        return {
            "region": region_name,
            "area": REGION_LABELS.get(region_name, region_name),
            "cnn_similarity": 0.0,
            "color_similarity": 0.0,
            "histogram_similarity": 0.0,
            "texture_similarity": 0.0,
            "shape_similarity": 0.0,
            "final_similarity": 0.0,
            "status": "Tidak terdeteksi"
        }

    ref_feature = extract_feature(ref_roi)
    user_feature = extract_feature(user_roi)

    cnn_score = calculate_cnn_similarity(ref_feature, user_feature)
    color_score = calculate_color_similarity(ref_roi, user_roi, ref_mask, user_mask)
    hist_score = calculate_histogram_similarity(ref_roi, user_roi, ref_mask, user_mask)
    texture_score = calculate_texture_similarity(ref_roi, user_roi)
    shape_score = calculate_shape_similarity(ref_mask, user_mask)

    # Gabungan fitur
    final_score = (
        0.15 * cnn_score +
        0.30 * color_score +
        0.25 * hist_score +
        0.20 * texture_score +
        0.10 * shape_score
    )

    final_score = round(final_score, 2)

    return {
        "region": region_name,
        "area": REGION_LABELS.get(region_name, region_name),
        "cnn_similarity": cnn_score,
        "color_similarity": color_score,
        "histogram_similarity": hist_score,
        "texture_similarity": texture_score,
        "shape_similarity": shape_score,
        "final_similarity": final_score,
        "status": get_status(final_score)
    }


# =========================
# TOTAL SCORE
# =========================
def calculate_total_score(results):
    total = 0.0

    for item in results:
        region = item["region"]
        score = item["final_similarity"]
        weight = WEIGHTS[region]
        total += score * weight

    return round(total, 2)


# =========================
# FEEDBACK ALA MUA
# =========================
def generate_feedback(results):
    feedback = []

    for item in results:
        region = item["region"]
        area = item["area"]

        final_score = item["final_similarity"]
        color_score = item["color_similarity"]
        texture_score = item["texture_similarity"]
        shape_score = item["shape_similarity"]

        issues = []

        if color_score < 70:
            issues.append(("color", color_score))

        if texture_score < 70:
            issues.append(("texture", texture_score))

        if shape_score < 70:
            issues.append(("shape", shape_score))

        # Ambil maksimal 2 masalah utama agar feedback tidak terlalu ramai
        issues = sorted(issues, key=lambda x: x[1])[:2]

        tips = []

        for issue, _ in issues:
            if issue == "color":
                if region == "Lips":
                    tips.append("warna lipstik bisa dibuat lebih mendekati referensi")
                elif region in ["Left Eye", "Right Eye"]:
                    tips.append("warna eyeshadow dapat disesuaikan agar lebih serupa")

                elif region in ["Left Eyebrow", "Right Eyebrow"]:
                    tips.append("warna alis dapat disesuaikan agar lebih menyatu dengan tampilan referensi")

                elif region in ["Left Cheek", "Right Cheek"]:
                    tips.append("warna blush dapat dibuat lebih lembut atau lebih dekat dengan referensi")

            elif issue == "texture":
                if region == "Lips":
                    tips.append("ratakan aplikasi lipstik agar hasilnya terlihat lebih halus")
                elif region in ["Left Eye", "Right Eye"]:
                    tips.append("baurkan eyeshadow agar transisinya terlihat lebih halus")

                elif region in ["Left Eyebrow", "Right Eyebrow"]:
                    tips.append("rapikan arsiran alis agar hasilnya terlihat lebih natural")

                elif region in ["Left Cheek", "Right Cheek"]:
                    tips.append("baurkan blush agar terlihat lebih menyatu dengan kulit")

            elif issue == "shape":
                if region == "Lips":
                    tips.append("rapikan garis bibir agar bentuknya lebih mendekati referensi")
                elif region in ["Left Eye", "Right Eye"]:
                    tips.append("rapikan batas eyeshadow atau eyeliner agar bentuknya lebih sesuai")

                elif region in ["Left Eyebrow", "Right Eyebrow"]:
                    tips.append("sesuaikan bentuk dan ketebalan alis agar lebih seimbang")

                elif region in ["Left Cheek", "Right Cheek"]:
                    tips.append("atur posisi blush agar mengikuti area pipi pada referensi")

        if final_score >= 85:
            feedback.append(
                f"{area}: hasilnya sudah sangat mendekati referensi. Pertahankan keseimbangan warna dan bentuknya."
            )

        elif final_score >= 70:
            if tips:
                feedback.append(
                    f"{area}: sudah cukup mirip. Untuk hasil yang lebih rapi, {tips[0]}."
                )
            else:
                feedback.append(
                    f"{area}: sudah cukup mirip, hanya perlu sedikit penyempurnaan agar lebih menyerupai referensi."
                )

        else:
            if len(tips) == 0:
                feedback.append(
                    f"{area}: masih perlu disesuaikan agar lebih mendekati referensi."
                )
            elif len(tips) == 1:
                feedback.append(
                    f"{area}: masih perlu disempurnakan. Saran: {tips[0]}."
                )
            else:
                feedback.append(
                    f"{area}: masih perlu disempurnakan. Saran: {tips[0]}, lalu {tips[1]}."
                )

    return feedback


# =========================
# OVERLAY VISUALISASI
# =========================
def create_overlay(image, landmarks, prefix="referensi"):
    overlay = image.copy()

    colors = {
        "Lips": (255, 0, 0),
        "Left Eye": (0, 255, 0),
        "Right Eye": (0, 255, 0),
        "Left Eyebrow": (255, 255, 0),
        "Right Eyebrow": (255, 255, 0),
        "Left Cheek": (255, 105, 180),
        "Right Cheek": (255, 105, 180)
    }

    for region_name, indices in REGIONS.items():
        points = np.array([landmarks[i] for i in indices], dtype=np.int32)
        color = colors.get(region_name, (255, 0, 0))
        cv2.fillPoly(overlay, [points], color)

    result = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)
    return result


# =========================
# MAIN PIPELINE
# =========================
def analyze_makeup(reference_img, user_img):
    reference_img = preprocess_image(reference_img,"referensi")
    user_img = preprocess_image(user_img,"user")

    ref_landmarks = get_landmarks(reference_img,"referensi")
    user_landmarks = get_landmarks(user_img,"user")

    if ref_landmarks is None:
        raise ValueError("Wajah pada gambar referensi tidak terdeteksi.")

    if user_landmarks is None:
        raise ValueError("Wajah pada gambar pengguna tidak terdeteksi.")

    results = []

    for region_name, indices in REGIONS.items():
        result = analyze_region(
            reference_img,
            user_img,
            ref_landmarks,
            user_landmarks,
            region_name,
            indices
        )
        results.append(result)

    total_score = calculate_total_score(results)
    feedback = generate_feedback(results)

    ref_overlay = create_overlay(reference_img, ref_landmarks,"referensi")
    user_overlay = create_overlay(user_img, user_landmarks,"user")

    return {
        "total_score": total_score,
        "results": results,
        "feedback": feedback,
        "reference_overlay": ref_overlay,
        "user_overlay": user_overlay
    }
