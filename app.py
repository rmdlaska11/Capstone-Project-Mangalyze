import streamlit as st
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="MANGALYZE - Analisis Daun Mangga", layout="centered", page_icon="🍃")

# Load model .h5
@st.cache_resource
def load_keras_model():
    model = tf.keras.models.load_model("Model/densenet201.keras")
    return model

model = load_keras_model()

label_map = {
    0: 'Anthracnose',
    1: 'Bacterial Canker',
    2: 'Cutting Weevil',
    3: 'Die Back',
    4: 'Gall Midge',
    5: 'Healthy',
    6: 'Powdery Mildew',
    7: 'Sooty Mould'
}

recommendation_map = {
    'Anthracnose': 'Gunakan fungisida berbahan aktif (mankozeb, tembaga hidroksida, atau propineb) sesuai dosis anjuran.',
    'Bacterial Canker': 'Potong bagian yang terinfeksi dan gunakan bakterisida berbahan tembaga (copper-based).',
    'Cutting Weevil': 'Gunakan insektisida berbahan aktif (imidakloprid, lambda-cyhalothrin) dan periksa kebersihan lingkungan sekitar tanaman.',
    'Die Back': 'Lakukan pemangkasan daun mati dan semprotkan fungisida sistemik (benomil, karbendazim, tebuconazole).',
    'Gall Midge': 'Pangkas dan bakar daun/bunga yang terinfestasi dan aplikasikan insektisida sistematik (imidakloprid, abamektin, spinosad).',
    'Healthy': 'Tanaman sehat! Lanjutkan pemupukan dan penyiraman rutin.',
    'Powdery Mildew': 'Semprot dengan fungisida sistemik dan preventif (karathane, hexaconazole, sulfur, miklobutanil).',
    'Sooty Mould': 'Pangkas ranting yang terlalu rimbun dan semprot air sabun ringan atau campuran air + fungisida ringan.'
}

def preprocess(image_file):
    target_size = (224, 224)
    # load_img menerima file-like object, cocok untuk uploaded_file Streamlit
    img = load_img(image_file, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ----------------- CSS Styling -----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    body {
        font-family: 'Poppins', sans-serif;
        background: #f0f5f1;
        margin: 0;
        padding: 0;
    }
    .topnav {
    overflow: hidden;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 12px 24px;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    justify-content: flex-end;
    position: sticky;
    top: 0;
    z-index: 999;
    }
    .topnav a {
        color: #2c3e50;
        text-decoration: none;
        padding: 14px 16px;
        font-weight: 500;
    }
    .topnav a:hover { color: #4CAF50; }
    .hero {
    background: linear-gradient(to right, rgba(70, 129, 92, 0.8), rgba(11, 87, 35, 0.8)), 
    url("https://images.unsplash.com/photo-1724565923616-efc1864bbacf?q=80&w=2070&auto=format&fit=crop") center/cover no-repeat;
    color: white;
    padding: 80px 20px;
    text-align: center;
    border-radius: 8px;
    }
    .hero h1 { font-size: 3rem; margin-bottom: 10px; }
    .hero p { font-size: 1.2rem; }
    .hero button {
    background-color:#4CAF50;
    color:white;
    padding:12px 24px;
    border:none;
    border-radius:8px;
    font-size:16px;
    margin-top:20px;
    cursor:pointer;
    }

    .disease-card {
        background: white;
        border-radius: 16px;
        padding: 15px;
        margin: 12px 0;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: default;
        color: #2e7d32;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .disease-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
    }
    .disease-card h4 {
        font-weight: 600;
    }
    .disease-card p {
        font-size: 1rem;
        color: #4a784a;
    }

    .stButton > button {
        background: linear-gradient(90deg, #4caf50 0%, #81c784 100%);
        color: white;
        padding: 14px 35px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 14px;
        transition: background 0.3s ease, transform 0.2s ease;
        box-shadow: 0 6px 15px rgba(76, 175, 80, 0.5);
        margin-top: 10px;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #388e3c 0%, #66bb6a 100%);
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(56, 142, 60, 0.7);
    }

    .result-card {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        border-radius: 16px;
        padding: 25px;
        margin-top: 25px;
        box-shadow: 0 8px 20px rgba(76, 175, 80, 0.3);
        color: #1b5e20;
        font-family: 'Poppins', sans-serif;
    }
    .result-card h3 {
        margin-bottom: 10px;
        font-weight: 700;
    }
    .result-card h4 {
        margin-top: 15px;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .result-card p {
        font-size: 1rem;
        line-height: 1.5;
    }

    .upload-area {
        border: 2px dashed #4caf50;
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        color: #4caf50;
        font-weight: 600;
        font-size: 1.3rem;
        transition: background-color 0.3s ease;
        cursor: pointer;
        margin-bottom: 30px;
    }
    .upload-area:hover {
        background-color: #e8f5e9;
    }

    .footer {
        text-align: center;
        color: #888;
        padding: 20px 0;
        font-size: 0.9rem;
        margin-top: 50px;
        border-top: 1px solid #ddd;
        user-select: none;
    }
    </style>
""", unsafe_allow_html=True)

# ----------- Navbar -----------
# st.markdown("""
# <div class="topnav">
#     <a href="#beranda">Beranda</a>
#     <a href="#jenis-penyakit">Jenis Penyakit</a>
#     <a href="#prediksi">Prediksi</a>
#     <a href="#panduan">Panduan</a>
# </div>
# """, unsafe_allow_html=True)

st.sidebar.markdown("""
    <style>
    .css-1d391kg {  /* class default sidebar container pada Streamlit (cek classnya di Inspect Element) */
        padding: 20px 15px;
        background: #F7F7F7;
        box-shadow: 2px 0 10px rgb(76 175 80 / 0.15);
        border-radius: 0 10px 10px 0;
    }

    .sidebar-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 25px;
        text-align: center;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        user-select: none;
        box-shadow: 0 5px 5px rgb(0 0 0 / 0.1);
    }
    .sidebar-description {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1rem;
        color: #4a4a4a;
        margin-bottom: 30px;
        text-align: center;
        font-style: italic;
        user-select: none;
        border-radius: 8px;
    }
    .sidebar-link {
        display: block;
        padding: 14px 12px;
        color: #2e7d32;
        text-decoration: none;
        border-radius: 8px;
        transition: background-color 0.3s ease, color 0.3s ease;
        margin: 8px 0;
        font-weight: 300;
        font-size: 1.05rem;
        text-align: left;
        user-select: none;
    }
    .sidebar-link:hover {
        background-color: #81c784;
        color: #fff;
        box-shadow: 0 4px 10px rgb(0 0 0 / 0.2);
    }
    .sidebar-footer {
        margin-top: 40px;
        font-size: 0.85rem;
        color: #999;
        text-align: center;
        user-select: none;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-title">Mangalyze</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-description">Identifikasi Penyakit Daun Mangga</div>', unsafe_allow_html=True)

st.sidebar.markdown('<a href="#beranda" class="sidebar-link" style="text-decoration:none; color:black;">Beranda</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#jenis-penyakit" class="sidebar-link" style="text-decoration:none; color:black;">Jenis Penyakit</a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="#prediksi" class="sidebar-link" style="text-decoration:none; color:black;">Prediksi</a>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-footer">© 2025 Mangalyze</div>', unsafe_allow_html=True)

# ----------- Hero -----------
# if selected_section == "Beranda":
st.markdown("<div id='beranda'></div>", unsafe_allow_html=True)
st.markdown("""
    <div class='hero'>
        <h1>Deteksi Penyakit Daun Mangga</h1>
        <p>Gunakan AI untuk mengidentifikasi penyakit pada daun mangga dengan cepat dan akurat.</p>
        <a href="#prediksi"><button>Mulai Deteksi</button></a>
    </div>
    """, unsafe_allow_html=True)

# -------- Jenis Penyakit --------
# if selected_section == "Jenis Penyakit":
st.markdown("<div id='jenis-penyakit'></div>", unsafe_allow_html=True)
st.subheader("🌿 Jenis Penyakit Daun Mangga")

diseases = [
        ("Anthracnose", "Penyakit jamur (<b>Colletotrichum gloeosporioides</b>) yang ditandai dengan munculnya bercak-bercak coklat tua hingga hitam."),
        ("Bacterial Canker", "Penyakit berbasis bakteri yang ditandai dengan munculnya bercak nekrotik (mati) berair atau kering pada daun, biasanya dikelilingi oleh tepi kuning."),
        ("Cutting Weevil", "Serangan hama dari kelompok kumbang kecil yang menyerang tunas muda, tangkai bunga, dan cabang kecil dengan cara memotong atau menggigit pangkal jaringan."),
        ("Die Back", "Penyakit jamur yang menyebabkan kematian jaringan tanaman dimulai dari ujung (pucuk) ranting atau cabang, yang kemudian menyebar ke arah batang utama."),
        ("Gall Midge", 'Serangan hama yang menyebabkan terbentuknya "gall" (bengkokan atau benjolan abnormal) pada daun atau jaringan tanaman lainnya.'),
        ("Powdery Mildew", "Penyakit jamur (<b>Oidium mangiferae</b>) yang ditandai dengan munculnya lapisan putih seperti tepung pada permukaan daun."),
        ("Sooty Mould", "Penyakit jamur yang menyebabkan lapisan berwarna hitam pekat seperti arang atau jelaga muncul di permukaan daun."),
    ]

    # Tampilkan card penyakit dalam 3 kolom
cols = st.columns(3)
for idx, (name, desc) in enumerate(diseases):
        with cols[idx % 3]:
            st.markdown(f"""
                <div class="disease-card">
                    <h4>{name}</h4>
                    <p>{desc}</p>
                </div>
            """, unsafe_allow_html=True)

# -------- Upload Section --------
# if selected_section == "Prediksi":
st.markdown("<div id='prediksi'></div>", unsafe_allow_html=True)
st.subheader("📤 Unggah Gambar Daun Mangga")

uploaded_file = st.file_uploader("Pilih gambar berformat JPG / PNG", type=["jpg", "jpeg", "png"], label_visibility="visible")

if uploaded_file:
        image = Image.open(uploaded_file)
        # Set a fixed width for the image display
        col1, col2, col3 = st.columns(3)
        with col2:
            st.image(image, caption="Pratinjau Gambar Daun", width=200)

else:
        image = None

    # -------- Analisis Tombol --------
analyze_btn = st.button("🔍 Analisis Daun")

if analyze_btn:
        if image is None:
            st.warning("⚠️ Silakan unggah gambar terlebih dahulu.")
        else:
            with st.spinner("Menganalisis gambar..."):
                try:
                    input_data = preprocess(uploaded_file)
                    output_data = model.predict(input_data)
                    predicted_label = np.argmax(output_data)
                    confidence = output_data[0][predicted_label] * 100
                    label_name = label_map.get(predicted_label, "Unknown")
                    recommendation = recommendation_map.get(label_name, "Tidak ada rekomendasi.")

                    st.markdown(f"""
                        <div class="result-card">
                            <h3>✅ Hasil Prediksi: <span style="font-weight:bold;">{label_name}</span> ({confidence:.2f}%)</h3>
                            <h4>📌 Rekomendasi Penanganan:</h4>
                            <p>{recommendation}</p>
                        </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menganalisis: {e}")

# ------------------ Panduan ------------------
# if selected_section == "Panduan":
st.markdown("<div id='panduan' style='margin-top: 20px;'>", unsafe_allow_html=True)

with st.expander("❓ Cara Menggunakan Aplikasi"):
        st.markdown("""
        1. 📷 Unggah gambar daun mangga yang ingin dianalisis.
        2. 🔍 Klik tombol **Analisis Daun**.
        3. ✅ Tunggu hasil prediksi dan baca rekomendasi penanganan.
        """)
st.markdown('</div>', unsafe_allow_html=True)

# -------- Footer --------
st.markdown("""
    <div class="footer">
        &copy; 2025 Mangalyze. Dibuat oleh Tim Mangalyze.
    </div>
""", unsafe_allow_html=True)
