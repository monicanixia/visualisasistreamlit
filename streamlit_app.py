import os
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from streamlit.components.v1 import html

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Analisis Sentimen Mobil Listrik",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
.main{
    background-color:#0E1117;
}
[data-testid="stSidebar"]{
    background-color:#111827;
}
h1,h2,h3,h4{
    color:white;
}
.metric-card{
    background-color:#1F2937;
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# PATH DATA
# ==========================================================
# Catatan: kode ini dijalankan sebagai aplikasi Streamlit biasa
# (streamlit run app.py), BUKAN di Google Colab.
# Jadi tidak pakai drive.mount() — itu hanya berfungsi di Colab
# dan akan error kalau dijalankan di luar Colab.
#
# Folder data harus ditaruh sejajar dengan file app.py ini,
# atau atur BASE_PATH sesuai lokasi file kamu.

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Kalau mau pakai path manual (misal di server / lokal), ganti baris di atas dengan:
# BASE_PATH = "/path/ke/folder/data"

REQUIRED_FILES = {
    "yt_sentimen": "dataset_sentimen_final_Youtube.xlsx",
    "tt_sentimen": "dataset_sentimen_final Tiktok Baru.xlsx",
    "yt_aspek":    "dataset_final_aspek_Youtube.xlsx",
    "tt_aspek":    "dataset_final_aspek Tiktok.xlsx",
    "lda_yt":      "top_words_lda Youtube.xlsx",
    "lda_tt":      "top_words_lda Tiktok.xlsx",
    "xgb_yt":      "ringkasan_xgboost_Youtube.xlsx",
    "xgb_tt":      "ringkasan_xgboost_Tiktok.xlsx",
    "lstm_yt":     "ringkasan_lstm_Youtube.xlsx",
    "lstm_tt":     "ringkasan_lstm_Tiktok.xlsx",
}

PYLDAVIS_FILES = {
    "Youtube": "pyldavis_k10_Youtube (2).html",
    "TikTok":  "pyldavis_k10 Tiktok.html",
}


def check_missing_files():
    """Cek file mana yang tidak ditemukan, biar errornya jelas dari awal."""
    missing = []
    for key, fname in REQUIRED_FILES.items():
        path = os.path.join(BASE_PATH, fname)
        if not os.path.exists(path):
            missing.append(fname)
    return missing


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    data = {}
    for key, fname in REQUIRED_FILES.items():
        path = os.path.join(BASE_PATH, fname)
        data[key] = pd.read_excel(path)
    return data


missing_files = check_missing_files()

if missing_files:
    st.error("❌ File berikut tidak ditemukan di folder data:")
    for f in missing_files:
        st.write(f"- `{f}`")
    st.info(
        f"Pastikan semua file Excel ditaruh di folder: `{BASE_PATH}`\n\n"
        "Atau ubah variabel `BASE_PATH` di kode sesuai lokasi file kamu."
    )
    st.stop()

data = load_data()

yt_sentimen = data["yt_sentimen"]
tt_sentimen = data["tt_sentimen"]
yt_aspek    = data["yt_aspek"]
tt_aspek    = data["tt_aspek"]
lda_yt      = data["lda_yt"]
lda_tt      = data["lda_tt"]
xgb_yt      = data["xgb_yt"]
xgb_tt      = data["xgb_tt"]
lstm_yt     = data["lstm_yt"]
lstm_tt     = data["lstm_tt"]

# ==========================================================
# DATA GABUNGAN
# ==========================================================

dataset = pd.concat([yt_sentimen, tt_sentimen], ignore_index=True)


def value_counts_df(series, col_name):
    """
    Helper agar value_counts() -> DataFrame 2 kolom selalu konsisten,
    tidak peduli versi pandas (menghindari bug nama kolom 'index' vs nama asli).
    """
    vc = series.value_counts().rename_axis(col_name).reset_index(name="Jumlah")
    return vc


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🚗 Analisis Sentimen")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Dashboard",
        "Distribusi Sentimen",
        "Perbandingan Dataset",
        "Analisis Aspek",
        "WordCloud",
        "Topik LDA",
        "PyLDAvis",
        "Evaluasi XGBoost",
        "Evaluasi LSTM",
        "Perbandingan Model",
        "Data Komentar"
    ]
)

# ==========================================================
# DASHBOARD
# ==========================================================

if menu == "Dashboard":

    st.title("🚗 Dashboard Analisis Sentimen Mobil Listrik")

    if "sentiment_label_final" not in dataset.columns:
        st.error("Kolom 'sentiment_label_final' tidak ditemukan di dataset.")
        st.stop()

    total_data = len(dataset)
    total_yt = len(yt_sentimen)
    total_tt = len(tt_sentimen)

    label_col = dataset["sentiment_label_final"].astype(str)

    positif = label_col.str.contains("Positif", case=False, na=False).sum()
    netral  = label_col.str.contains("Netral", case=False, na=False).sum()
    negatif = label_col.str.contains("Negatif", case=False, na=False).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", f"{total_data:,}")
    c2.metric("Youtube", f"{total_yt:,}")
    c3.metric("TikTok", f"{total_tt:,}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Positif", f"{positif:,}")
    c5.metric("Netral", f"{netral:,}")
    c6.metric("Negatif", f"{negatif:,}")

    st.divider()

    st.subheader("Distribusi Sentimen Keseluruhan")

    sentimen = value_counts_df(dataset["sentiment_label_final"], "Sentimen")

    fig = px.bar(
        sentimen,
        x="Sentimen",
        y="Jumlah",
        text="Jumlah",
        color="Sentimen",
        title="Distribusi Sentimen Seluruh Dataset"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# DISTRIBUSI SENTIMEN
# ==========================================================

elif menu == "Distribusi Sentimen":

    st.title("📊 Distribusi Sentimen")

    sumber = st.radio("Pilih Dataset", ["Gabungan", "Youtube", "TikTok"], horizontal=True)

    if sumber == "Youtube":
        df_pilih = yt_sentimen
    elif sumber == "TikTok":
        df_pilih = tt_sentimen
    else:
        df_pilih = dataset

    if "sentiment_label_final" not in df_pilih.columns:
        st.error("Kolom 'sentiment_label_final' tidak ditemukan.")
        st.stop()

    sentimen = value_counts_df(df_pilih["sentiment_label_final"], "Sentimen")

    fig = px.bar(
        sentimen,
        x="Sentimen",
        y="Jumlah",
        text="Jumlah",
        color="Sentimen",
        title=f"Distribusi Sentimen Dataset {sumber}"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(sentimen, use_container_width=True)

# ==========================================================
# PERBANDINGAN DATASET
# ==========================================================

elif menu == "Perbandingan Dataset":

    st.title("⚖️ Perbandingan Youtube vs TikTok")

    yt = value_counts_df(yt_sentimen["sentiment_label_final"], "Sentimen")
    yt["Dataset"] = "Youtube"

    tt = value_counts_df(tt_sentimen["sentiment_label_final"], "Sentimen")
    tt["Dataset"] = "TikTok"

    gabung = pd.concat([yt, tt], ignore_index=True)

    fig = px.bar(
        gabung,
        x="Sentimen",
        y="Jumlah",
        color="Dataset",
        barmode="group",
        text="Jumlah",
        title="Perbandingan Sentimen Youtube dan TikTok"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(gabung, use_container_width=True)

# ==========================================================
# ANALISIS ASPEK
# ==========================================================

elif menu == "Analisis Aspek":

    st.title("🎯 Analisis Aspek")

    sumber = st.radio("Pilih Dataset", ["Gabungan", "Youtube", "TikTok"], horizontal=True)

    if sumber == "Youtube":
        df_pilih = yt_sentimen
    elif sumber == "TikTok":
        df_pilih = tt_sentimen
    else:
        df_pilih = dataset

    if "label_aspect" not in df_pilih.columns:
        st.error("Kolom 'label_aspect' tidak ditemukan di dataset ini.")
        st.stop()

    aspek = value_counts_df(df_pilih["label_aspect"], "Aspek")

    fig = px.bar(
        aspek,
        x="Aspek",
        y="Jumlah",
        text="Jumlah",
        color="Jumlah",
        title=f"Distribusi Aspek Dataset {sumber}"
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(aspek, use_container_width=True)

# ==========================================================
# WORDCLOUD
# ==========================================================

elif menu == "WordCloud":

    st.title("☁️ WordCloud")

    sumber = st.radio("Dataset", ["Gabungan", "Youtube", "TikTok"], horizontal=True)

    if sumber == "Youtube":
        df_pilih = yt_sentimen
    elif sumber == "TikTok":
        df_pilih = tt_sentimen
    else:
        df_pilih = dataset

    pilihan = st.selectbox("Pilih Sentimen", ["Positif", "Netral", "Negatif"])

    if "sentiment_label_final" not in df_pilih.columns or "text" not in df_pilih.columns:
        st.error("Kolom 'sentiment_label_final' atau 'text' tidak ditemukan.")
        st.stop()

    temp = df_pilih[
        df_pilih["sentiment_label_final"].astype(str).str.lower() == pilihan.lower()
    ]

    teks = " ".join(temp["text"].dropna().astype(str))

    if len(teks.strip()) == 0:
        st.warning("Tidak ada data untuk ditampilkan.")
    else:
        wc = WordCloud(
            width=1500,
            height=800,
            background_color="black",
            colormap="viridis",
            max_words=200
        ).generate(teks)

        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
        st.caption(f"WordCloud Sentimen {pilihan} - Dataset {sumber}")

# ==========================================================
# TOPIK LDA
# ==========================================================

elif menu == "Topik LDA":

    st.title("🧠 Topic Modeling LDA")

    sumber = st.radio("Pilih Dataset", ["Youtube", "TikTok"], horizontal=True)

    if sumber == "Youtube":
        st.subheader("Topik LDA Youtube")
        st.dataframe(lda_yt, use_container_width=True)
    else:
        st.subheader("Topik LDA TikTok")
        st.dataframe(lda_tt, use_container_width=True)

# ==========================================================
# PYLDAVIS
# ==========================================================

elif menu == "PyLDAvis":

    st.title("📌 Visualisasi PyLDAvis")

    sumber = st.radio("Pilih Dataset", ["Youtube", "TikTok"], horizontal=True)

    file_html = os.path.join(BASE_PATH, PYLDAVIS_FILES[sumber])

    if not os.path.exists(file_html):
        st.error(f"File HTML PyLDAvis tidak ditemukan: `{PYLDAVIS_FILES[sumber]}`")
        st.info(f"Pastikan file ada di folder: `{BASE_PATH}`")
    else:
        with open(file_html, "r", encoding="utf-8") as f:
            source_code = f.read()
        html(source_code, height=900, scrolling=True)

# ==========================================================
# EVALUASI XGBOOST
# ==========================================================

elif menu == "Evaluasi XGBoost":

    st.title("🚀 Evaluasi XGBoost")

    tab1, tab2 = st.tabs(["Youtube", "TikTok"])

    with tab1:
        st.subheader("Hasil Evaluasi Youtube")
        st.dataframe(xgb_yt, use_container_width=True)

    with tab2:
        st.subheader("Hasil Evaluasi TikTok")
        st.dataframe(xgb_tt, use_container_width=True)

# ==========================================================
# EVALUASI LSTM
# ==========================================================

elif menu == "Evaluasi LSTM":

    st.title("🤖 Evaluasi LSTM")

    tab1, tab2 = st.tabs(["Youtube", "TikTok"])

    with tab1:
        st.subheader("Hasil Evaluasi Youtube")
        st.dataframe(lstm_yt, use_container_width=True)

    with tab2:
        st.subheader("Hasil Evaluasi TikTok")
        st.dataframe(lstm_tt, use_container_width=True)

# ==========================================================
# PERBANDINGAN MODEL
# ==========================================================

elif menu == "Perbandingan Model":

    st.title("⚖️ Perbandingan Model")

    sumber = st.radio("Pilih Dataset", ["Youtube", "TikTok"], horizontal=True)

    if sumber == "Youtube":
        df_xgb, df_lstm = xgb_yt, lstm_yt
    else:
        df_xgb, df_lstm = xgb_tt, lstm_tt

    st.subheader(f"Ringkasan XGBoost - {sumber}")
    st.dataframe(df_xgb, use_container_width=True)

    st.subheader(f"Ringkasan LSTM - {sumber}")
    st.dataframe(df_lstm, use_container_width=True)

# ==========================================================
# DATA KOMENTAR
# ==========================================================

elif menu == "Data Komentar":

    st.title("📝 Data Komentar")

    sumber = st.radio("Pilih Dataset", ["Gabungan", "Youtube", "TikTok"], horizontal=True)

    if sumber == "Youtube":
        df_pilih = yt_sentimen
    elif sumber == "TikTok":
        df_pilih = tt_sentimen
    else:
        df_pilih = dataset

    st.dataframe(df_pilih, use_container_width=True)

    csv = df_pilih.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Dataset",
        data=csv,
        file_name=f"{sumber}.csv",
        mime="text/csv"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown(
    """
    <center>
    <h5>
    Dashboard Analisis Sentimen Mobil Listrik Indonesia
    <br>
    Menggunakan LDA, XGBoost dan LSTM
    </h5>
    </center>
    """,
    unsafe_allow_html=True
)
