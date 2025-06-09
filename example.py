import time
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input

# Set page config
st.set_page_config(
    page_title="Deteksi Penyakit Daun Mangga",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model
@st.cache_resource
def load_keras_model():
    model = tf.keras.models.load_model("model/densenet201.keras")
    return model

model = load_keras_model()

# Label and recommendation maps
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

def preprocess(image_path):
    target_size = (224, 224)
    # load_img menerima file-like object, cocok untuk uploaded_file Streamlit
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Custom CSS to style the app
st.markdown("""
    <style>
        /* Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Navbar Fix */
        .navbar {
            position: fixed;
            top: 0;
            width: 100%;
            left: 0;
            z-index: 1000;
            background-color: white;
            padding: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .navbar-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0 20px;
        }
        
        .navbar-brand {
            font-weight: bold;
            font-size: 1.5rem;
            color: black !important;
            text-decoration: none !important;
        }
        
        .navbar-brand span {
            color: #09d65f !important;
        }
        
        .nav-links {
            display: flex;
            gap: 20px;
        }
        
        .nav-link {
            color: #555555 !important;
            text-decoration: none !important;
            font-weight: 500;
            padding: 5px 0;
            position: relative;
            transition: color 0.3s ease;
        }
        
        .nav-link:hover {
            color: #178011 !important;
        }
        
        .nav-link::after {
            content: '';
            position: absolute;
            width: 0;
            height: 2px;
            bottom: 0;
            left: 0;
            background-color: #178011;
            transition: width 0.3s ease;
        }
        
        .nav-link:hover::after {
            width: 100%;
        }
        
        .nav-link.active {
            color: #178011 !important;
            font-weight: bold;
        }
        
        .nav-link.active::after {
            width: 100%;
        }
        
        /* Pastikan konten tidak tertutup navbar */
        .main-container {
            padding-top: 0px ;
            margin-top: 0px;
        }
        
        /* Hero section */
        .hero {
            background: linear-gradient(to right, rgba(70, 129, 92, 0.689), rgba(11, 87, 35, 0.667)), 
                        url("https://images.unsplash.com/photo-1724565923616-efc1864bbacf?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D") center/cover no-repeat;
            position: relative;
            width: 100vw;
            margin-left: calc(-50vw + 50%);
            margin-right: calc(-50vw + 50%);
            color: white;
            padding: 150px 0 100px;
            text-align: center;
            border-radius: 10px;
            margin-top: -100px; 
            margin-bottom: 30px;
        }
        
        .hero h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 20px;
        }
        
        .hero p {
            font-size: 1.2rem;
            margin-bottom: 30px;
        }
        
        .hero-btn {
            background-color: white;
            color: black !important;
            padding: 12px 30px;
            border-radius: 5px;
            text-decoration: none !important;
            font-weight: 600;
            transition: all 0.3s ease;
            display: inline-block;
        }
        
        .hero-btn:hover {
            background-color: #f8f9fa;
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        /* Disease cards */
        .card-disease {
            height: 100%;
            display: flex;
            flex-direction: column;
            transition: all 0.3s ease;
            cursor: pointer;
            border-radius: 1rem !important;
            box-shadow: 0 0.5rem 1rem rgb(0 0 0 / 0.15);
            padding: 20px;
            margin-bottom: 20px;
            background-color: white;
            border: 1px solid #e9ecef;
            justify-content: center !important;
            align-items: center !important;
            text-align: center !important;
        }
        
        .card-disease:hover {
            transform: translateY(-10px);
            box-shadow: 0 1rem 2rem rgb(0 0 0 / 0.25);
        }
        
        .card-disease i {
            font-size: 3rem;
            color: #0f8835;
            margin-bottom: 1rem;
        }
        
        .card-disease h3 {
            color: #212529;
            margin-bottom: 15px;
        }
        
        .card-disease p {
            color: #6c757d;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #178011;
            color: white;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
            padding: 12px;
        }
        
        .stButton>button:hover {
            background-color: #0c6c2c;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        /* Result styling */
        .result-image {
            max-height: 300px;
            border-radius: 1rem;
            box-shadow: 0 0.25rem 0.5rem rgb(0 0 0 / 0.15);
            margin: 0 auto;
            display: block;
        }
        
        .recommendation-card {
            border-radius: 1rem;
            box-shadow: 0 0.25rem 0.75rem rgb(0 0 0 / 0.1);
            padding: 20px;
            background-color: #f8f9fa;
            margin-top: 20px;
        }
        
        /* Section titles */
        .section-title {
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            color: #212529;
        }
        
        .section-subtitle {
            text-align: center;
            color: #6c757d;
            margin-bottom: 30px;
        }
        
        /* Hide streamlit default elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* File uploader styling */
        .stFileUploader>div>div>div>div {
            color: #495057;
        }
        
        .uploaded-image {
            border-radius: 10px;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Add Bootstrap Icons
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Navbar with fixed scrolling behavior
st.markdown("""
    <nav class="navbar">
        <div class="navbar-container">
            <a class="navbar-brand" href="#hero">Mangalyze<span>.</span></a>
            <div class="nav-links">
                <a class="nav-link" href="#hero">Beranda</a>
                <a class="nav-link" href="#disease-cards">Jenis Penyakit</a>
                <a class="nav-link" href="#deteksi">Prediksi</a>
            </div>
        </div>
    </nav>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <section id="hero" class="hero">
        <h1>Deteksi Penyakit Daun Mangga</h1>
        <p>Gunakan AI untuk mengidentifikasi penyakit pada daun mangga dengan cepat dan akurat.</p>
        <a href="#deteksi" class="hero-btn">Mulai Deteksi</a>
    </section>
""", unsafe_allow_html=True)

# Diseases Section
st.markdown("""
    <section id="disease-cards">
        <h1 class="section-title">Jenis Penyakit Daun Mangga</h1>
        <p class="section-subtitle">Berikut beberapa penyakit yang bisa dideteksi sistem kami:</p>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .card-disease {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            min-height: 270px;
            display: flex;
            flex-direction: column;
            justify-content: center ;
            align-items: center !important;    /* center horizontal & vertical isi */
            text-align: center !important;     /* text center */
        }

        .card-disease i {
            font-size: 40px;
            margin-bottom: 6px;      /* rapatkan jarak bawah ikon */
            color: #0f6d3d;
        }

        .card-disease h3 {
            margin: 4px 0 6px 0;     /* rapat atas bawah */
            font-size: 18px;
        }

        .card-disease p {
            font-size: 14px;
            line-height: 1.3;
            margin: 0;               /* hilangkan margin default */
        }
    </style>
""", unsafe_allow_html=True)


# Baris pertama - 4 kolom
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-bug-fill"></i>
            <h3>Anthracnose</h3>
            <p>Penyakit jamur (<b>Colletotrichum gloeosporioides</b>) yang ditandai dengan munculnya bercak-bercak coklat tua hingga hitam.</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-virus"></i>
            <h3>Bacterial Canker</h3>
            <p>Penyakit berbasis bakteri yang ditandai dengan munculnya bercak nekrotik (mati) berair atau kering pada daun.</p>
        </div>
    """, unsafe_allow_html=True)
    
with col3:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-scissors"></i>
            <h3>Cutting Weevil</h3>
            <p>Serangan hama dari kelompok kumbang kecil yang menyerang tunas muda, tangkai bunga, dan cabang kecil.</p>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-droplet-half"></i>
            <h3>Die Back</h3>
            <p>Penyakit jamur yang menyebabkan kematian jaringan tanaman dimulai dari ujung (pucuk) ranting atau cabang.</p>
        </div>
    """, unsafe_allow_html=True)

# Baris kedua - 3 kolom
col5, col6, col7 = st.columns(3)

with col5:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-bug"></i>
            <h3>Gall Midge</h3>
            <p>Serangan hama yang menyebabkan terbentuknya "gall" (bengkokan atau benjolan abnormal) pada daun.</p>
        </div>
    """, unsafe_allow_html=True)
    
with col6:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-cloud-drizzle"></i>
            <h3>Powdery Mildew</h3>
            <p>Penyakit jamur (<b>Oidium mangiferae</b>) yang ditandai dengan munculnya lapisan putih seperti tepung pada permukaan daun.</p>
        </div>
    """, unsafe_allow_html=True)

with col7:
    st.markdown("""
        <div class="card-disease">
            <i class="bi bi-cloud-fog2"></i>
            <h3>Sooty Mould</h3>
            <p>Penyakit jamur yang menyebabkan lapisan berwarna hitam pekat seperti arang atau jelaga muncul di permukaan daun.</p>
        </div>
    """, unsafe_allow_html=True)

# Detection Section
st.markdown("""
    <section id="deteksi">
        <h2 class="section-title">Mulai Prediksi Penyakit</h2>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

# Style tinggi minimal untuk kotak agar sama tinggi
box_style = """
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #ddd;
    box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.08);
    min-height: 50px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
    width: 100%;
"""

with col_left:
    st.markdown(f"""
        <div style="{box_style}">
            <h3 style="text-align:center; font-weight:700; font-size:1.5rem;">Unggah Gambar Daun Mangga</h3>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown("</div>", unsafe_allow_html=True)  # tutup div sebelum tampilkan gambar
        st.image(image, caption="Pratinjau Gambar Daun", use_container_width=True)
    else:
        st.markdown("""
            <p style='text-align:center; color:#888; margin-top: 8px;'>Belum ada gambar yang diunggah.</p>
            </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown(f"""
        <div style="{box_style}">
            <h3 style="text-align:center; color:#178011; font-weight:700; font-size:1.5rem;">
                <i class="bi bi-clipboard-check" style="margin-right: 8px;"></i>Hasil Deteksi
            </h3>
    """, unsafe_allow_html=True)

    detect_btn = st.button("Deteksi Sekarang")

    if detect_btn:
        if image is None:
            st.warning("⚠️ Silakan unggah gambar terlebih dahulu.")
        else:
            with st.spinner("Menganalisis gambar..."):
                # Simpan sementara file agar bisa diproses
                filename = f"{int(time.time())}_{uploaded_file.name}"
                os.makedirs("temp", exist_ok=True)
                image_path = os.path.join("temp", filename)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Preprocess dan prediksi
                image_processed = preprocess(image_path)
                prediction = model.predict(image_processed)
                predicted_label = np.argmax(prediction)
                label_name = label_map[predicted_label]
                confidence = prediction[0][predicted_label] * 100
                recommendation = recommendation_map.get(label_name, "Tidak ada rekomendasi khusus.")

                # Hasil diagnosa dan rekomendasi
                st.markdown(f"""
                    <div style="display: flex; align-items: center; background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-top: 20px;">
                        <i class="bi bi-activity" style="font-size: 1.8rem; margin-right: 12px; color: #178011;"></i>
                        <div>
                            <h4 style="margin-bottom: 5px; font-size: 1.25rem;">Diagnosa</h4>
                            <p style="font-size: 1.1rem; font-weight: 600;">{label_name} ({confidence:.2f}%)</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                     <div style="
                        background-color: #fff;
                        padding: 15px;
                        border-radius: 10px;
                        margin-top: 15px;
                        border: 1px solid #ddd;
                        box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.05);
                    ">
                        <h4 style="color: #178011; margin-bottom: 10px; font-size: 1.25rem;">
                            <i class="bi bi-lightbulb-fill" style="margin-right: 8px;"></i>Rekomendasi Penanganan
                        </h4>
                        <p style="font-size: 1.05rem;">{recommendation}</p>
                    </div>
                """, unsafe_allow_html=True)

                os.remove(image_path)
    else:
        st.markdown("""
            <p style="text-align: center; color: #6c757d; font-style: italic; font-size: 1.05rem;">
                Unggah gambar dan klik "Deteksi Sekarang" untuk memulai.
            </p>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</section>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; margin-top: 50px; border-radius: 10px;">
        <p>&copy; 2025 Mangalyze. Dibuat oleh Tim Mangalyze.</p>
    </div>
    </div> <!-- Close main-container -->
""", unsafe_allow_html=True)

# Fixed JavaScript for smooth scrolling and active state
st.markdown("""
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Fungsi untuk handle smooth scroll
            function smoothScroll(target) {
                const element = document.querySelector(target);
                if (element) {
                    window.scrollTo({
                        top: element.offsetTop - 70,
                        behavior: 'smooth'
                    });
                }
            }
            
            // Handle klik nav link
            document.querySelectorAll('.nav-link').forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const target = this.getAttribute('href');
                    smoothScroll(target);
                });
            });
            
            // Handle scroll untuk active state
            window.addEventListener('scroll', function() {
                const scrollPos = window.scrollY + 100;
                document.querySelectorAll('.nav-link').forEach(link => {
                    const target = document.querySelector(link.getAttribute('href'));
                    if (target) {
                        if (target.offsetTop <= scrollPos && 
                            target.offsetTop + target.offsetHeight > scrollPos) {
                            link.classList.add('active');
                        } else {
                            link.classList.remove('active');
                        }
                    }
                });
            });
        });
    </script>
""", unsafe_allow_html=True)
