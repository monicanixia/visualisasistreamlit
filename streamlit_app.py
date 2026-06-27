"""
PART 001
streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from streamlit.components.v1 import html

@st.cache_data
def load_data():

    yt = pd.read_excel("dataset_sentimen_final_Youtube Baru.xlsx")
    tt = pd.read_excel("dataset_sentimen_final Tiktok Baru.xlsx")

    aspek_yt = pd.read_excel("dataset_final_aspek_Youtube.xlsx")
    aspek_tt = pd.read_excel("dataset_final_aspek Tiktok.xlsx")

    lda_yt = pd.read_excel("top_words_lda Youtube.xlsx")
    lda_tt = pd.read_excel("top_words_lda Tiktok.xlsx")

    xgb_yt = pd.read_excel("ringkasan_xgboost_Youtube.xlsx")
    xgb_tt = pd.read_excel("ringkasan_xgboost_Tiktok.xlsx")

    lstm_yt = pd.read_excel("ringkasan_lstm_Youtube.xlsx")
    lstm_tt = pd.read_excel("ringkasan_lstm_Tiktok.xlsx")
    word_yt = pd.read_excel("dataset_wordcloud_Youtube.xlsx")
    word_tt = pd.read_excel("dataset_wordcloud_Tiktok.xlsx")

    return (
        yt,
        tt,
        aspek_yt,
        aspek_tt,
        lda_yt,
        lda_tt,
        xgb_yt,
        xgb_tt,
        lstm_yt,
        lstm_tt,
        word_yt,
        word_tt
    )


# =========================
# TIDAK ADA INDENTASI LAGI
# =========================

(
    yt,
    tt,
    aspek_yt,
    aspek_tt,
    lda_yt,
    lda_tt,
    xgb_yt,
    xgb_tt,
    lstm_yt,
    lstm_tt,
    word_yt,
    word_tt
) = load_data()


dataset = pd.concat([yt, tt], ignore_index=True)


def hitung_statistik(data):

    total = len(data)

    positif = len(data[data["sentiment_label_final"] == "Positif"])

    netral = len(data[data["sentiment_label_final"] == "Netral"])

    negatif = len(data[data["sentiment_label_final"] == "Negatif"])

    return total, positif, netral, negatif
 


st.set_page_config(
    page_title="Analisis Sentimen Mobil Listrik Indonesia",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main{
    background-color:#f5f7fb;
}
.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 2px 10px rgba(0,0,0,.08);
}
h1,h2,h3{
    color:#1565C0;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🚗 Analisis Sentimen Mobil Listrik")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Dashboard",
        "📁 Dataset",
        "📊 Distribusi Label",
        "☁️ Word Cloud",
        "🔤 Top Words",
        "📈 Evaluasi Model"
    ]
)

if menu=="🏠 Dashboard":
    st.title("Dashboard Analisis Sentimen Mobil Listrik Indonesia")

    total, positif, netral, negatif = hitung_statistik(dataset)
    
    c1,c2,c3 = st.columns(3)

    c1.metric("📊 Total", total)

    c2.metric("▶️ Youtube", len(yt))

    c3.metric("🎵 TikTok", len(tt))

    st.markdown("---")

    kiri,kanan=st.columns([2,1])

    with kiri:
        st.subheader("Ringkasan")
        st.write("""
Dashboard ini nantinya akan menampilkan:

- Total komentar
- Total Youtube
- Total TikTok
- Distribusi Sentimen
- Grafik Interaktif
- Statistik Dataset
- Insight otomatis
""")

    with kanan:
        st.subheader("Menu")
        st.write("""
✅ Dashboard

✅ Dataset

✅ Distribusi Label

✅ WordCloud

✅ Top Words

✅ Evaluasi Model
""")

elif menu=="📁 Dataset":

    st.title("📁 Dataset Hasil Preprocessing")

    pilihan = st.radio(
        "Pilih Dataset",
        ["📺 YouTube", "🎵 TikTok"],
        horizontal=True
    )

    if pilihan == "📺 YouTube":
        data = yt
        nama = "YouTube"

    else:
        data = tt
        nama = "TikTok"

    st.markdown(f"### Dataset {nama}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Jumlah Data", len(data))

    with col2:
        st.metric("Jumlah Kolom", len(data.columns))

    st.markdown("---")

    st.subheader("Preview Dataset")

    st.dataframe(
        data,
        use_container_width=True,
        height=500
    )

    st.download_button(
        label="⬇ Download Dataset",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name=f"dataset_preprocessing_{nama}.csv",
        mime="text/csv"
    )

elif menu=="📊 Distribusi Label":

    st.title("📊 Distribusi Label Sentimen")

    pilihan = st.radio(
        "Pilih Dataset",
        ["📺 YouTube", "🎵 TikTok"],
        horizontal=True
    )

    if pilihan == "📺 YouTube":
        data = yt
        nama = "YouTube"
    else:
        data = tt
        nama = "TikTok"

    st.subheader(f"Distribusi Label Dataset {nama}")

    # Menghitung jumlah label
    distribusi = (
        data["sentiment_label_final"]
        .value_counts()
        .reindex(["Positif", "Negatif", "Netral"], fill_value=0)
        .reset_index()
    )

    distribusi.columns = ["Sentimen", "Jumlah"]

    # Warna batang
    warna = {
        "Positif": "#2ecc71",
        "Negatif": "#e74c3c",
        "Netral": "#95a5a6"
    }

    fig = px.bar(
        distribusi,
        x="Sentimen",
        y="Jumlah",
        color="Sentimen",
        text="Jumlah",
        color_discrete_map=warna
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        title=f"Distribusi Sentimen Dataset {nama}",
        xaxis_title="Label Sentimen",
        yaxis_title="Jumlah Data",
        showlegend=False,
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "😊 Positif",
        int(distribusi.loc[
            distribusi["Sentimen"]=="Positif",
            "Jumlah"
        ].values[0])
    )

    c2.metric(
        "😠 Negatif",
        int(distribusi.loc[
            distribusi["Sentimen"]=="Negatif",
            "Jumlah"
        ].values[0])
    )

    c3.metric(
        "😐 Netral",
        int(distribusi.loc[
            distribusi["Sentimen"]=="Netral",
            "Jumlah"
        ].values[0])
    )

    st.markdown("### Tabel Distribusi Label")

    st.dataframe(
        distribusi,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# HALAMAN WORD CLOUD
# PART 1
# Import Library + Load Dataset
# ============================================================

import os
from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud

from collections import Counter

# ============================================================
# KONFIGURASI FOLDER
# ============================================================

BASE_DIR = Path(__file__).parent

# ============================================================
# LOKASI DATASET
# ============================================================

# Jika file berada di folder utama repository

YT_FILE = BASE_DIR / "dataset_wordcloud_Youtube.xlsx"
TT_FILE = BASE_DIR / "dataset_wordcloud_Tiktok.xlsx"

# ============================================================
# JIKA DATASET BERADA DI FOLDER "dataset"
# ============================================================

if not YT_FILE.exists():

    YT_FILE = BASE_DIR / "dataset" / "dataset_wordcloud_Youtube.xlsx"

if not TT_FILE.exists():

    TT_FILE = BASE_DIR / "dataset" / "dataset_wordcloud_Tiktok.xlsx"

# ============================================================
# VALIDASI FILE
# ============================================================

if not YT_FILE.exists():

    st.error(
        f"""
Dataset YouTube tidak ditemukan.

Lokasi yang dicek :

{YT_FILE}
"""
    )

    st.stop()

if not TT_FILE.exists():

    st.error(
        f"""
Dataset TikTok tidak ditemukan.

Lokasi yang dicek :

{TT_FILE}
"""
    )

    st.stop()

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data(show_spinner=False)

def load_wordcloud_dataset(path):

    df = pd.read_excel(path)

    return df

word_yt = load_wordcloud_dataset(YT_FILE)

word_tt = load_wordcloud_dataset(TT_FILE)

# ============================================================
# MEMBERSIHKAN NAMA KOLOM
# ============================================================

word_yt.columns = (
    word_yt.columns
    .str.strip()
    .str.lower()
)

word_tt.columns = (
    word_tt.columns
    .str.strip()
    .str.lower()
)

# ============================================================
# VALIDASI KOLOM
# ============================================================

kolom_wajib = [

    "text_wordcloud",

    "sentimen"

]

for kolom in kolom_wajib:

    if kolom not in word_yt.columns:

        st.error(
            f"""
Kolom '{kolom}' tidak ditemukan pada dataset YouTube.

Kolom yang tersedia :

{list(word_yt.columns)}
"""
        )

        st.stop()

for kolom in kolom_wajib:

    if kolom not in word_tt.columns:

        st.error(
            f"""
Kolom '{kolom}' tidak ditemukan pada dataset TikTok.

Kolom yang tersedia :

{list(word_tt.columns)}
"""
        )

        st.stop()

# ============================================================
# NORMALISASI SENTIMEN
# ============================================================

word_yt["sentimen"] = (

    word_yt["sentimen"]

    .fillna("")

    .astype(str)

    .str.strip()

    .str.title()

)

word_tt["sentimen"] = (

    word_tt["sentimen"]

    .fillna("")

    .astype(str)

    .str.strip()

    .str.title()

)

# ============================================================
# MEMBERSIHKAN KOLOM WORD CLOUD
# ============================================================

word_yt["text_wordcloud"] = (

    word_yt["text_wordcloud"]

    .fillna("")

    .astype(str)

)

word_tt["text_wordcloud"] = (

    word_tt["text_wordcloud"]

    .fillna("")

    .astype(str)

)

# ============================================================
# MENGHAPUS DUPLIKAT
# ============================================================

word_yt = word_yt.drop_duplicates().reset_index(drop=True)

word_tt = word_tt.drop_duplicates().reset_index(drop=True)

# ============================================================
# INFORMASI DATASET
# ============================================================

jumlah_youtube = len(word_yt)

jumlah_tiktok = len(word_tt)

total_dataset = jumlah_youtube + jumlah_tiktok

# ============================================================
# WARNA WORD CLOUD
# ============================================================

WORDCLOUD_WIDTH = 1800

WORDCLOUD_HEIGHT = 900

BACKGROUND_COLOR = "white"

COLORMAP = "viridis"

MAX_WORD = 200

# ============================================================
# STOPWORDS TAMBAHAN
# ============================================================

CUSTOM_STOPWORDS = {

    "mobil",
    "listrik",
    "kendaraan",
    "ev",
    "indonesia",

    "yg",
    "ya",
    "iya",
    "aja",
    "nih",
    "sih",
    "bang",
    "bro",
    "deh",
    "dong",
    "kan",
    "kok",

    "aku",
    "saya",
    "kami",
    "kita",
    "orang",

    "buat",
    "jadi",
    "karena",
    "kalau",
    "kalo",

    "lebih",
    "masih",
    "cukup",
    "banyak",

    "itu",
    "ini",

    "the",
    "and",

    "nya",
    "pun",
    "nihh",
    "dongg",
    "lah",

    "udah",
    "sudah",
    "biar",
    "dari",
    "dengan",
    "dalam",
    "ke",
    "di",
    "yang",
    "untuk",
    "pada",
    "oleh",
    "atau",
    "agar",
    "sebagai",
    "sebuah",
    "karna"

}

# ============================================================
# FUNGSI MEMBERSIHKAN TEXT
# ============================================================

def clean_wordcloud_text(text):

    text = str(text)

    text = text.replace("[", " ")

    text = text.replace("]", " ")

    text = text.replace("'", " ")

    text = text.replace('"', " ")

    text = text.replace(",", " ")

    text = " ".join(text.split())

    return text

word_yt["text_wordcloud"] = word_yt["text_wordcloud"].apply(clean_wordcloud_text)

word_tt["text_wordcloud"] = word_tt["text_wordcloud"].apply(clean_wordcloud_text)

# ============================================================
# AKHIR PART 1
# ============================================================

# ============================================================
# HALAMAN WORD CLOUD
# PART 2
# ============================================================

elif menu == "☁️ Word Cloud":

    st.title("☁️ Word Cloud Sentimen")

    st.markdown("""
Visualisasi **Word Cloud** digunakan untuk menampilkan kata-kata yang paling sering muncul
berdasarkan **dataset** dan **label sentimen**.

Silakan pilih dataset dan sentimen yang ingin dianalisis.
""")

    st.markdown("---")

    # ============================================================
    # INFORMASI DATASET
    # ============================================================

    info1, info2, info3 = st.columns(3)

    info1.metric(
        "Dataset YouTube",
        f"{jumlah_youtube:,}"
    )

    info2.metric(
        "Dataset TikTok",
        f"{jumlah_tiktok:,}"
    )

    info3.metric(
        "Total Dataset",
        f"{total_dataset:,}"
    )

    st.markdown("---")

    # ============================================================
    # PILIH DATASET DAN SENTIMEN
    # ============================================================

    col1, col2 = st.columns(2)

    with col1:

        dataset_pilih = st.selectbox(
            "📂 Pilih Dataset",
            (
                "YouTube",
                "TikTok"
            ),
            index=0
        )

    with col2:

        label_pilih = st.selectbox(
            "😊 Pilih Sentimen",
            (
                "Positif",
                "Netral",
                "Negatif"
            ),
            index=0
        )

    st.markdown("---")

    # ============================================================
    # MEMILIH DATASET
    # ============================================================

    if dataset_pilih == "YouTube":

        data = word_yt.copy()

    else:

        data = word_tt.copy()

    # ============================================================
    # FILTER SENTIMEN
    # ============================================================

    data = data[
        data["sentimen"] == label_pilih
    ].copy()

    # ============================================================
    # INFORMASI HASIL FILTER
    # ============================================================

    st.subheader(f"📊 Dataset {dataset_pilih}")

    jumlah_filter = len(data)

    total_kata = (
        data["text_wordcloud"]
        .fillna("")
        .astype(str)
        .str.split()
        .explode()
        .shape[0]
    )

    unik_kata = (
        data["text_wordcloud"]
        .fillna("")
        .astype(str)
        .str.split()
        .explode()
        .nunique()
    )

    met1, met2, met3 = st.columns(3)

    met1.metric(
        "Jumlah Komentar",
        f"{jumlah_filter:,}"
    )

    met2.metric(
        "Total Kata",
        f"{total_kata:,}"
    )

    met3.metric(
        "Kata Unik",
        f"{unik_kata:,}"
    )

    st.markdown("---")

    # ============================================================
    # JIKA DATA KOSONG
    # ============================================================

    if data.empty:

        st.warning(
            f"""
Tidak ada data dengan label sentimen **{label_pilih}**
pada dataset **{dataset_pilih}**.
"""
        )

        st.stop()

    # ============================================================
    # MEMBENTUK TEXT WORD CLOUD
    # ============================================================

    text = " ".join(

        data["text_wordcloud"]

        .fillna("")

        .astype(str)

    )

    text = clean_wordcloud_text(text)

    if len(text.strip()) == 0:

        st.warning("Seluruh isi text kosong.")

        st.stop()

    st.markdown("---")

    # ============================================================
    # LANJUT KE PART 3
    # ============================================================
    # ============================================================
    # MEMBUAT WORD CLOUD
    # ============================================================

    wc = WordCloud(

        width=WORDCLOUD_WIDTH,

        height=WORDCLOUD_HEIGHT,

        background_color=BACKGROUND_COLOR,

        stopwords=CUSTOM_STOPWORDS,

        max_words=MAX_WORD,

        collocations=False,

        contour_width=2,

        contour_color="steelblue",

        colormap=COLORMAP,

        random_state=42

    ).generate(text)

    # ============================================================
    # TAMPILKAN WORD CLOUD
    # ============================================================

    st.subheader(
        f"☁️ Word Cloud {dataset_pilih} ({label_pilih})"
    )

    fig, ax = plt.subplots(figsize=(18, 9))

    ax.imshow(
        wc,
        interpolation="bilinear"
    )

    ax.axis("off")

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.markdown("---")

    # ============================================================
    # HITUNG FREKUENSI KATA ASLI
    # ============================================================

    semua_kata = []

    for kalimat in data["text_wordcloud"]:

        kata = str(kalimat).split()

        kata = [

            k.lower()

            for k in kata

            if len(k) > 1

            and k.lower() not in CUSTOM_STOPWORDS

        ]

        semua_kata.extend(kata)

    counter = Counter(semua_kata)

    top_word = (

        pd.DataFrame(

            counter.items(),

            columns=[

                "Kata",

                "Frekuensi"

            ]

        )

        .sort_values(

            by="Frekuensi",

            ascending=False

        )

        .reset_index(drop=True)

        .head(20)

    )

    # ============================================================
    # JIKA TOP WORD KOSONG
    # ============================================================

    if top_word.empty:

        st.warning(
            "Tidak ada kata yang dapat ditampilkan."
        )

        st.stop()

    # ============================================================
    # STATISTIK SETELAH WORD CLOUD
    # ============================================================

    st.subheader("📊 Statistik Word Cloud")

    m1, m2, m3 = st.columns(3)

    m1.metric(

        "Jumlah Kata",

        f"{len(semua_kata):,}"

    )

    m2.metric(

        "Kata Unik",

        f"{len(counter):,}"

    )

    m3.metric(

        "Top Word",

        top_word.iloc[0]["Kata"]

    )

    st.markdown("---")

    # ============================================================
    # GRAFIK TOP 20 KATA
    # ============================================================

    st.subheader("📈 20 Kata Paling Sering Muncul")

    fig_bar = px.bar(

        top_word,

        x="Frekuensi",

        y="Kata",

        orientation="h",

        color="Frekuensi",

        text="Frekuensi",

        color_continuous_scale="Viridis"

    )

    fig_bar.update_layout(

        height=700,

        title=f"Top 20 Kata ({dataset_pilih} - {label_pilih})",

        yaxis=dict(

            categoryorder="total ascending"

        ),

        coloraxis_showscale=False

    )

    fig_bar.update_traces(

        textposition="outside"

    )

    st.plotly_chart(

        fig_bar,

        use_container_width=True

    )

    st.markdown("---")

    # ============================================================
    # LANJUT PART 4
    # ============================================================
    # ============================================================
    # TABEL TOP 20 KATA
    # ============================================================

    st.subheader("📋 Tabel Frekuensi Kata")

    with st.expander("Lihat Tabel Top 20 Kata", expanded=True):

        tampil_top = top_word.copy()

        tampil_top.index = tampil_top.index + 1

        tampil_top.index.name = "No"

        st.dataframe(

            tampil_top,

            use_container_width=True

        )

    st.markdown("---")

    # ============================================================
    # PENCARIAN KATA
    # ============================================================

    st.subheader("🔍 Cari Kata")

    keyword = st.text_input(

        "Masukkan kata yang ingin dicari",

        placeholder="contoh : baterai"

    )

    if keyword != "":

        hasil = top_word[
            top_word["Kata"].str.contains(
                keyword,
                case=False,
                na=False
            )
        ]

        if len(hasil) == 0:

            st.warning("Kata tidak ditemukan.")

        else:

            st.success(f"Ditemukan {len(hasil)} kata.")

            st.dataframe(

                hasil,

                use_container_width=True,

                hide_index=True

            )

    st.markdown("---")

    # ============================================================
    # DOWNLOAD CSV
    # ============================================================

    csv = top_word.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="⬇️ Download CSV",

        data=csv,

        file_name=f"Top20_{dataset_pilih}_{label_pilih}.csv",

        mime="text/csv"

    )

    # ============================================================
    # DOWNLOAD EXCEL
    # ============================================================

    from io import BytesIO

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        top_word.to_excel(

            writer,

            index=False,

            sheet_name="Top Word"

        )

    st.download_button(

        label="📥 Download Excel",

        data=output.getvalue(),

        file_name=f"Top20_{dataset_pilih}_{label_pilih}.xlsx",

        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    )

    st.markdown("---")

    # ============================================================
    # RINGKASAN
    # ============================================================

    st.subheader("📝 Ringkasan")

    st.info(f"""
Dataset : **{dataset_pilih}**

Label Sentimen : **{label_pilih}**

Jumlah Komentar : **{jumlah_filter:,}**

Jumlah Kata : **{len(semua_kata):,}**

Jumlah Kata Unik : **{len(counter):,}**

Kata yang paling sering muncul adalah **{top_word.iloc[0]['Kata']}**
dengan frekuensi **{top_word.iloc[0]['Frekuensi']}** kali.
""")

    st.markdown("---")

    # ============================================================
    # SELESAI PART 4
    # LANJUT KE PART 5
    # ============================================================
    # ============================================================
    # DISTRIBUSI SENTIMEN
    # ============================================================

    st.subheader("📊 Distribusi Sentimen")

    if dataset_pilih == "YouTube":

        distribusi = (
            word_yt["sentimen"]
            .value_counts()
            .rename_axis("Sentimen")
            .reset_index(name="Jumlah")
        )

    else:

        distribusi = (
            word_tt["sentimen"]
            .value_counts()
            .rename_axis("Sentimen")
            .reset_index(name="Jumlah")
        )

    fig_sentimen = px.pie(

        distribusi,

        names="Sentimen",

        values="Jumlah",

        hole=0.45,

        color="Sentimen",

        color_discrete_map={

            "Positif": "#2ecc71",

            "Netral": "#f1c40f",

            "Negatif": "#e74c3c"

        }

    )

    fig_sentimen.update_layout(

        height=500,

        legend_title="Label Sentimen"

    )

    st.plotly_chart(

        fig_sentimen,

        use_container_width=True

    )

    st.markdown("---")

    # ============================================================
    # 10 KATA TERATAS
    # ============================================================

    st.subheader("🏆 10 Kata Terpopuler")

    top10 = top_word.head(10).copy()

    top10.index = top10.index + 1

    st.table(top10)

    st.markdown("---")

    # ============================================================
    # DATASET HASIL FILTER
    # ============================================================

    with st.expander("📄 Lihat Dataset Hasil Filter"):

        tampil = data.copy()

        st.dataframe(

            tampil,

            use_container_width=True,

            hide_index=True

        )

    st.markdown("---")

    # ============================================================
    # DOWNLOAD DATASET HASIL FILTER
    # ============================================================

    csv_data = data.to_csv(

        index=False

    ).encode("utf-8")

    st.download_button(

        label="⬇️ Download Dataset Hasil Filter",

        data=csv_data,

        file_name=f"Dataset_{dataset_pilih}_{label_pilih}.csv",

        mime="text/csv"

    )

    st.markdown("---")

    # ============================================================
    # INFORMASI
    # ============================================================

    with st.expander("ℹ️ Informasi Visualisasi"):

        st.markdown("""

### Penjelasan

Visualisasi Word Cloud digunakan untuk menampilkan kata-kata yang
paling sering muncul pada komentar berdasarkan label sentimen.

Semakin besar ukuran suatu kata pada Word Cloud,
semakin sering kata tersebut muncul pada komentar.

Tahapan yang dilakukan:

- Dataset dipilih (YouTube atau TikTok)
- Data difilter berdasarkan label sentimen
- Dilakukan penghapusan stopwords
- Word Cloud dibentuk menggunakan library **WordCloud**
- Frekuensi kata dihitung menggunakan **Counter**
- Ditampilkan grafik Top 20 kata
- Hasil dapat diunduh dalam format CSV maupun Excel

""")

    st.success("✅ Visualisasi Word Cloud berhasil dibuat.")

    # ============================================================
    # END WORD CLOUD
    # ============================================================

elif menu=="🔤 Top Words":
    st.title("Top Words")
    st.warning("Fitur akan dilengkapi pada Part 005.")

elif menu=="📈 Evaluasi Model":
    st.title("Evaluasi Model")
    st.warning("Fitur akan dilengkapi pada Part 006.")
