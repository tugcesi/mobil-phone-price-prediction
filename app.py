import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
import plotly.express as px

# PAGE CONFIGURATION
st.set_page_config(
    page_title="📱 Mobile Price Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3);
    }
    .main-header h1 {
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .result-box {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        padding: 25px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(155, 89, 182, 0.3);
    }
    .result-price {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    .result-label {
        font-size: 1.1em;
        opacity: 0.9;
    }
    .sidebar-title {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 15px;
        text-align: center;
        padding: 15px;
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        color: white;
        border-radius: 10px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1em;
        width: 100%;
        height: 50px;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(155, 89, 182, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        with open('mobile_price.pkl', 'rb') as f:
            bundle = pickle.load(f)
        model = bundle['model']
        return model
    except FileNotFoundError:
        try:
            model = joblib.load('mobile_price.joblib')
            return model
        except FileNotFoundError:
            st.error("❌ mobile_price.pkl veya mobile_price.joblib bulunamadı!")
            return None
    except Exception as e:
        st.error(f"❌ Model yükleme hatası: {str(e)}")
        return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_price_category_info(category):
    categories = {
        0: {'name': '💰 Low Cost',      'description': 'Bütçe Dostu',       'emoji': '💚', 'range': '$100 - $300', 'target': 'Giriş seviyesi kullanıcılar'},
        1: {'name': '💵 Medium Cost',   'description': 'Orta Fiyat',         'emoji': '🟡', 'range': '$300 - $600', 'target': 'Standart kullanıcılar'},
        2: {'name': '💳 High Cost',     'description': 'Yüksek Fiyat',       'emoji': '🔴', 'range': '$600 - $1200','target': 'Premium kullanıcılar'},
        3: {'name': '👑 Very High Cost','description': 'Çok Yüksek Fiyat',   'emoji': '⬛', 'range': '$1200+',      'target': 'Lüks segment'},
    }
    return categories.get(category, categories[0])


def predict_price_category(model, features_dict):
    try:
        # Notebook ile birebir aynı 9 feature
        feature_order = [
            'ram', 'performance_score', 'ram_x_cores', 'ram_to_memory',
            'battery_power', 'battery_per_weight', 'resolution',
            'px_width', 'px_height'
        ]

        feature_vector = [features_dict.get(f, 0) for f in feature_order]
        X = np.array(feature_vector).reshape(1, -1)

        prediction = model.predict(X)
        prediction = int(prediction.item() if isinstance(prediction, np.ndarray) else prediction)
        prediction = max(0, min(3, prediction))

        probas = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else None

        return prediction, probas

    except Exception as e:
        st.error(f"❌ Tahmin hatası: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("""
    <div class="main-header">
        <h1>📱 Mobile Price Classifier</h1>
        <p>🔮 Telefon özelliklerine göre fiyat kategorisini belirleyin</p>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.stop()

    st.success("✓ Model başarıyla yüklendi!")

    with st.sidebar:
        st.markdown('<div class="sidebar-title">📊 Telefon Özelliklerini Girin</div>', unsafe_allow_html=True)
        st.divider()

        features_dict = {}
        extra_features = {}

        st.markdown("**⚙️ Temel Donanım Özellikleri**")
        st.caption("🤖 Bu özellikler fiyat tahminine etki eder")

        col1, col2 = st.columns(2)
        with col1:
            ram = st.slider("🧠 RAM (MB)", 512, 16000, 1024, 256,
                            help="Düşük: 512-2048 | Orta: 2048-4096 | Yüksek: 4096+")
            features_dict['ram'] = ram
        with col2:
            cores = st.slider("🔧 CPU Cores", 2, 12, 4)

        col1, col2 = st.columns(2)
        with col1:
            performance_score = st.slider("⚡ Performans Skoru (0-100)", 0, 100, 30,
                                          help="Düşük: 0-40 | Orta: 40-70 | Yüksek: 70-100")
            features_dict['performance_score'] = performance_score
        with col2:
            memory_capacity = st.slider("💾 Depolama (GB)", 16, 512, 32, 16,
                                        help="Düşük: 16-64 | Orta: 64-256 | Yüksek: 256+")

        col1, col2 = st.columns(2)
        with col1:
            battery_power = st.slider("🔋 Pil (mAh)", 1000, 6000, 2500, 100,
                                      help="Düşük: 1000-3000 | Orta: 3000-5000 | Yüksek: 5000+")
            features_dict['battery_power'] = battery_power
        with col2:
            weight = st.slider("⚖️ Ağırlık (g)", 100, 250, 180, 5)

        col1, col2 = st.columns(2)
        with col1:
            screen_size = st.slider("📏 Ekran Boyutu (inç)", 3.5, 7.0, 5.5, 0.1)
        with col2:
            # pixel_density modele VERİLMİYOR — sadece resolution hesabı için kullanılıyor
            pixel_density = st.slider("📊 Pixel Density (ppi)", 70, 500, 200, 10,
                                      help="Düşük: 70-200 | Orta: 200-400 | Yüksek: 400+")

        col1, col2 = st.columns(2)
        with col1:
            px_width = st.slider("📐 Piksel Genişlik", 720, 2400, 1080, 90)
            features_dict['px_width'] = px_width
        with col2:
            px_height = st.slider("📐 Piksel Yükseklik", 1280, 3200, 2160, 160)
            features_dict['px_height'] = px_height

        # Hesaplanan features
        ram_x_cores = (ram / 1000) * cores
        features_dict['ram_x_cores'] = ram_x_cores

        ram_to_memory = ram / (memory_capacity * 1024)
        features_dict['ram_to_memory'] = ram_to_memory

        battery_per_weight = battery_power / weight
        features_dict['battery_per_weight'] = battery_per_weight

        # resolution = pixel_density * screen_size  (modele verilen feature)
        resolution = pixel_density * screen_size
        features_dict['resolution'] = resolution

        st.divider()
        st.markdown("**✨ İlave Özellikler**")
        st.caption("ℹ️ Bu özellikler fiyat tahminine etki etmez (bilgi amaçlı)")

        col1, col2, col3 = st.columns(3)
        with col1:
            extra_features['wifi']             = st.checkbox("📡 WiFi", value=True)
        with col2:
            extra_features['bluetooth']        = st.checkbox("🔵 Bluetooth", value=True)
        with col3:
            extra_features['nfc']              = st.checkbox("💳 NFC", value=False)

        col1, col2, col3 = st.columns(3)
        with col1:
            extra_features['fast_charging']    = st.checkbox("⚡ Hızlı Şarj", value=True)
        with col2:
            extra_features['fingerprint']      = st.checkbox("👆 Parmak İzi", value=True)
        with col3:
            extra_features['face_recognition'] = st.checkbox("😊 Yüz Tanıma", value=False)

        col1, col2, col3 = st.columns(3)
        with col1:
            extra_features['water_resistance'] = st.checkbox("💧 Su Geçirmez", value=False)
        with col2:
            extra_features['wireless_charging']= st.checkbox("🔌 Wireless Şarj", value=False)
        with col3:
            extra_features['stereo_speakers']  = st.checkbox("🔊 Stereo Hoparlör", value=False)

        col1, col2 = st.columns(2)
        with col1:
            extra_features['usb_type']  = st.selectbox("🔗 USB Tipi", ['Micro USB', 'USB-C', 'Lightning'])
        with col2:
            extra_features['sim_slots'] = st.slider("📞 SIM Slot", 1, 3, 2)

        st.divider()
        predict_button = st.button("🔮 Fiyat Kategorisini Belirle", use_container_width=True)

    # ============================================================
    # RESULTS
    # ============================================================
    if predict_button:
        with st.spinner('⏳ Fiyat kategorisi belirleniyor...'):
            predicted_category, probas = predict_price_category(model, features_dict)

            if predicted_category is not None:
                category_info = get_price_category_info(predicted_category)

                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">{category_info['emoji']} Fiyat Kategorisi</div>
                    <div class="result-price">{category_info['name']}</div>
                    <div class="result-label">{category_info['description']} — {category_info['range']}</div>
                </div>
                """, unsafe_allow_html=True)

                st.success("✓ Tahmin başarıyla tamamlandı!")
                st.divider()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🧠 RAM", f"{ram} MB")
                col2.metric("⚡ Performans", f"{performance_score}/100")
                col3.metric("💾 Depolama", f"{memory_capacity} GB")
                col4.metric("🔋 Pil", f"{battery_power} mAh")

                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("⚙️ Donanım Özellikleri")
                    st.table(pd.DataFrame({
                        'Özellik': ['RAM', 'Performans', 'Çekirdek', 'Depolama', 'Ağırlık'],
                        'Değer':   [f"{ram} MB", f"{performance_score}/100", f"{cores} cores",
                                    f"{memory_capacity} GB", f"{weight} g"]
                    }))

                with col2:
                    st.subheader("🖥️ Ekran Özellikleri")
                    st.table(pd.DataFrame({
                        'Özellik': ['Boyut', 'Pixel Density', 'Genişlik', 'Yükseklik', 'Çözünürlük'],
                        'Değer':   [f"{screen_size:.1f}\"", f"{pixel_density} ppi",
                                    f"{px_width} px", f"{px_height} px", f"{resolution:.0f}"]
                    }))

                st.divider()
                st.subheader("📊 Hesaplanan Model Parametreleri")
                col1, col2, col3 = st.columns(3)
                col1.metric("🧠✖️ RAM × Cores",  f"{ram_x_cores:.2f}")
                col2.metric("💾 RAM/Memory",      f"{ram_to_memory:.4f}")
                col3.metric("🔋⚖️ Pil/Ağırlık",  f"{battery_per_weight:.2f}")

                st.divider()

                if probas is not None:
                    st.subheader("📊 Kategori Olasılıkları")
                    prob_df = pd.DataFrame({
                        'Kategori':    ['💰 Low Cost', '💵 Medium Cost', '💳 High Cost', '👑 Very High Cost'],
                        'Olasılık (%)': [round(p * 100, 2) for p in probas]
                    })
                    fig = px.bar(prob_df, x='Kategori', y='Olasılık (%)',
                                 title='Fiyat Kategorisi Olasılıkları',
                                 color='Kategori',
                                 color_discrete_map={
                                     '💰 Low Cost': '#2ecc71',
                                     '💵 Medium Cost': '#f39c12',
                                     '💳 High Cost': '#e74c3c',
                                     '👑 Very High Cost': '#34495e'
                                 })
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()

                st.subheader(f"{category_info['emoji']} Hedef Kitle")
                st.info(f"""
                **Kategori:** {category_info['name']}  
                **Fiyat Aralığı:** {category_info['range']}  
                **Hedef:** {category_info['target']}
                """)

                st.divider()
                st.subheader("✨ İlave Özellikler Özeti")

                feature_map = {
                    'wifi': "📡 WiFi", 'bluetooth': "🔵 Bluetooth", 'nfc': "💳 NFC",
                    'fast_charging': "⚡ Hızlı Şarj", 'fingerprint': "👆 Parmak İzi Sensörü",
                    'face_recognition': "😊 Yüz Tanıma", 'water_resistance': "💧 Su Geçirmez (IP Rating)",
                    'wireless_charging': "🔌 Wireless Şarj", 'stereo_speakers': "🔊 Stereo Hoparlör"
                }
                active = [label for key, label in feature_map.items() if extra_features.get(key)]

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Mevcut Özellikler:**")
                    if active:
                        for f in active:
                            st.success(f"✅ {f}")
                    else:
                        st.info("ℹ️ Ek özellik seçilmedi")
                with col2:
                    st.write("**Bağlantı Bilgileri:**")
                    st.markdown(f"""
                    - **USB Tipi:** {extra_features['usb_type']}
                    - **SIM Slot:** {extra_features['sim_slots']} slot
                    """)

                st.divider()
                st.subheader("💡 Öneriler")
                recs = []
                if ram < 2048:   recs.append("🧠 RAM kapasitesi düşük - Temel görevler için uygun")
                elif ram < 4096: recs.append("🧠 RAM kapasitesi orta - Çoğu uygulama için yeterli")
                else:            recs.append("🧠 RAM kapasitesi yüksek - Gaming ve multitasking için ideal")

                if battery_power < 3500:   recs.append("🔋 Pil kapasitesi kısıtlı - Sık şarja ihtiyaç duyabilir")
                elif battery_power > 5000: recs.append("🔋 Pil kapasitesi bol - 2-3 gün yeterli olacak")
                else:                      recs.append("🔋 Pil kapasitesi iyi - Tüm gün yeterli")

                if pixel_density < 200:   recs.append("📊 Ekran piksel yoğunluğu düşük - Standart görünüm")
                elif pixel_density > 400: recs.append("��� Ekran piksel yoğunluğu yüksek - Çok keskin görünüm")

                if performance_score < 40:   recs.append("⚡ Performans temel görevler için uygun")
                elif performance_score < 70: recs.append("⚡ Performans iyi - Oyun ve uygulamalar rahat çalışır")
                else:                        recs.append("⚡ Performans üstün - Yoğun uygulamalar için ideal")

                if extra_features['water_resistance']: recs.append("💧 Su geçirmez - Zorlu koşullarda güvenli kullanım")
                if extra_features['wireless_charging']: recs.append("🔌 Kablosuz şarj - Konforlu şarj deneyimi")
                if extra_features['face_recognition'] and extra_features['fingerprint']:
                    recs.append("🔒 Çift biyometrik - Maksimum güvenlik")

                for r in recs:
                    st.info(r)

            else:
                st.error("❌ Tahmin başarısız")
                st.warning("⚠️ Lütfen tüm özellikleri kontrol et")


if __name__ == "__main__":
    main()
