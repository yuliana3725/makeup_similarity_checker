import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from makeup_engine import analyze_makeup

st.set_page_config(
    page_title="Makeup Similarity Checker",
    page_icon="💄",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: #fdeaf1;
    }

    .block-container {
        padding-top: 3rem;
        max-width: 1180px;
    }

    h1, h2, h3 {
        color: #2f2f3a;
    }

    .hero {
        background: linear-gradient(135deg, #fff7fa, #ffe1ec);
        padding: 34px 38px;
        border-radius: 28px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(180, 60, 100, 0.12);
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        color: #b43b65;
        margin-bottom: 10px;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #7a4b5c;
        line-height: 1.6;
    }

    .info-card {
        background: white;
        padding: 26px;
        border-radius: 24px;
        box-shadow: 0 8px 24px rgba(180, 60, 100, 0.10);
        margin-bottom: 24px;
    }

    .score-box {
        background: linear-gradient(135deg, #ffcad8, #fff1f6);
        padding: 32px;
        border-radius: 26px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(180, 60, 100, 0.14);
        margin-bottom: 18px;
    }

    .score-label {
        font-size: 17px;
        color: #7c3d55;
        margin-bottom: 8px;
    }

    .score-number {
        font-size: 60px;
        font-weight: 800;
        color: #c33768;
    }

    .feedback-box {
        background: #fff7fa;
        padding: 15px 18px;
        border-left: 5px solid #ef7fa3;
        border-radius: 14px;
        margin-bottom: 10px;
        color: #4b3a40;
    }

    section[data-testid="stSidebar"] {
        background: #fff3f7;
    }

    div.stButton > button {
        background: #c94f7c;
        color: white;
        border-radius: 14px;
        padding: 12px 30px;
        border: none;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background: #ad3d68;
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background: #fff;
        padding: 18px;
        border-radius: 18px;
        border: 1px solid #f3c6d5;
    }

    div[data-testid="stMetric"] {
        background: white;
        padding: 18px;
        border-radius: 18px;
    }
</style>
""", unsafe_allow_html=True)


def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


with st.sidebar:
    st.markdown("## Tentang Sistem")
    st.write("Aplikasi ini membandingkan hasil makeup pengguna dengan gambar referensi.")

    st.markdown("### Metode")
    st.write("""
    - MediaPipe Face Mesh
    - MobileNetV2
    - ROI Extraction
    - Cosine Similarity
    - Color, Texture, Shape Analysis
    - Weighted Scoring
    """)

    st.info("Gunakan foto wajah frontal, jelas, dan pencahayaan cukup.")


st.markdown("""
<div class="hero">
    <div class="hero-title">💄 Makeup Similarity Checker</div>
    <div class="hero-subtitle">
        Evaluasi kemiripan hasil makeup berdasarkan gambar referensi dan foto pengguna.
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("## Upload Gambar")

col1, col2 = st.columns(2, gap="large")

with col1:
    ref_file = st.file_uploader(
        "Gambar referensi",
        type=["jpg", "jpeg", "png"]
    )

with col2:
    user_file = st.file_uploader(
        "Gambar pengguna",
        type=["jpg", "jpeg", "png"]
    )


if ref_file is not None and user_file is not None:
    ref_img = load_image(ref_file)
    user_img = load_image(user_file)

    st.markdown("## Preview Gambar")

    p1, p2 = st.columns(2, gap="large")

    with p1:
        st.image(ref_img, caption="Gambar Referensi", use_container_width=True)

    with p2:
        st.image(user_img, caption="Gambar Pengguna", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Mulai Analisis"):
        try:
            with st.spinner("Sedang menganalisis makeup..."):
                output = analyze_makeup(ref_img, user_img)

            total_score = output["total_score"]

            st.markdown("## Hasil Analisis")

            st.markdown(f"""
            <div class="score-box">
                <div class="score-label">Total Similarity</div>
                <div class="score-number">{total_score}%</div>
            </div>
            """, unsafe_allow_html=True)

            if total_score >= 85:
                st.success("Hasil makeup sangat mendekati gambar referensi.")
            elif total_score >= 70:
                st.warning("Hasil makeup cukup mirip, namun masih dapat disempurnakan.")
            else:
                st.error("Hasil makeup masih berbeda dari gambar referensi.")

            st.markdown("## Area yang Dianalisis")

            v1, v2 = st.columns(2, gap="large")

            with v1:
                st.image(
                    output["reference_overlay"],
                    caption="Area Analisis Referensi",
                    use_container_width=True
                )

            with v2:
                st.image(
                    output["user_overlay"],
                    caption="Area Analisis Pengguna",
                    use_container_width=True
                )

            df = pd.DataFrame(output["results"])

            st.markdown("## Hasil Per Area")

            df_simple = df[["area", "final_similarity", "status"]].copy()
            df_simple.columns = ["Area", "Similarity (%)", "Status"]

            st.dataframe(
                df_simple,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("## Grafik Similarity Per Area")

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(df_simple["Area"], df_simple["Similarity (%)"])
            ax.set_ylim(0, 100)
            ax.set_ylabel("Similarity (%)")
            ax.set_xlabel("Area Wajah")
            ax.set_title("Nilai Similarity Per Area")
            plt.xticks(rotation=25, ha="right")
            st.pyplot(fig)

            st.markdown("## Saran Perbaikan")

            for fb in output["feedback"]:
                st.markdown(
                    f'<div class="feedback-box">{fb}</div>',
                    unsafe_allow_html=True
                )

            with st.expander("Lihat Detail Teknis Analisis"):
                df_detail = df[
                    [
                        "area",
                        "cnn_similarity",
                        "color_similarity",
                        "histogram_similarity",
                        "texture_similarity",
                        "shape_similarity",
                        "final_similarity"
                    ]
                ]

                st.dataframe(
                    df_detail,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

else:
    st.info("Silakan upload gambar referensi dan gambar pengguna terlebih dahulu.")