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

from collections import Counter

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
# Ganti seluruh blok:
# elif menu=="☁️ Word Cloud":
# ============================================================

elif menu == "☁️ Word Cloud":

    st.title("☁️ Word Cloud Sentimen")

    st.markdown("""
Visualisasi Word Cloud digunakan untuk melihat kata yang paling sering
muncul berdasarkan dataset dan label sentimen.
""")

    st.divider()

    # =====================================================
    # PILIH DATASET
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:
        dataset_pilih = st.selectbox(
            "📂 Dataset",
            ("YouTube", "TikTok")
        )

    with col2:
        label_pilih = st.selectbox(
            "😊 Label Sentimen",
            ("Positif", "Netral", "Negatif")
        )

    # =====================================================
    # AMBIL DATASET
    # =====================================================

    if dataset_pilih == "YouTube":
        df = word_yt.copy()
    else:
        df = word_tt.copy()

    # =====================================================
    # NORMALISASI NAMA KOLOM
    # =====================================================

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # =====================================================
    # VALIDASI KOLOM
    # =====================================================

    kolom_wajib = [
        "text_wordcloud",
        "sentimen"
    ]

    kolom_hilang = [
        c for c in kolom_wajib
        if c not in df.columns
    ]

    if kolom_hilang:
        st.error(
            "Kolom tidak ditemukan: "
            + ", ".join(kolom_hilang)
        )
        st.stop()

    # =====================================================
    # FILTER SENTIMEN
    # =====================================================

    df["sentimen"] = (
        df["sentimen"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )

    df = df[
        df["sentimen"] == label_pilih
    ].copy()

    if df.empty:
        st.warning("Tidak ada data untuk sentimen ini.")
        st.stop()

    # =====================================================
    # MEMBERSIHKAN TEXT
    # =====================================================

    df["text_wordcloud"] = (
        df["text_wordcloud"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.replace(",", " ", regex=False)
    )

    # =====================================================
    # STOPWORDS
    # =====================================================

    stop_words = {
        "mobil","listrik","kendaraan","indonesia","ev",
        "yang","yg","nya","aja","nih","dong","kan",
        "sih","deh","bro","bang","aku","saya",
        "kami","kita","orang","jadi","buat",
        "karena","kalau","kalo","lebih","masih",
        "cukup","banyak","ini","itu","the","and"
    }
# =====================================================
# KAMUS KATA PER SENTIMEN
# =====================================================

    # =====================================================
    # LANJUT KE PART 2
    # =====================================================
# ============================================================
# PART 2
# MEMBUAT FREKUENSI KATA DOMINAN
# Tempelkan tepat setelah PART 1
# ============================================================

    # =====================================================
    # TOKENISASI
    # =====================================================

    semua_kata = []

    for kalimat in df["text_wordcloud"]:

        token = str(kalimat).split()

        token = [
            t.strip()
            for t in token
            if len(t.strip()) > 2
            and t.strip() not in stop_words
        ]

    semua_kata.extend(token)

    # =====================================================
    # HITUNG FREKUENSI
    # =====================================================

# =====================================================
# FREKUENSI BERDASARKAN SENTIMEN
# =====================================================

    from collections import Counter

    def hitung_counter(df_sentimen):

        kata = []

        for kalimat in df_sentimen["text_wordcloud"]:

            token = str(kalimat).split()

            token = [

                t.strip()

                for t in token

                if len(t.strip()) > 2

                and t.strip() not in stop_words

            ]

            kata.extend(token)

        return Counter(kata)

# Dataset lengkap

    dataset_semua = word_yt.copy() if dataset_pilih == "YouTube" else word_tt.copy()

    dataset_semua["sentimen"] = (
        dataset_semua["sentimen"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )

    counter_pos = hitung_counter(
        dataset_semua[
            dataset_semua["sentimen"] == "Positif"
        ]
    )

    counter_net = hitung_counter(
        dataset_semua[
            dataset_semua["sentimen"] == "Netral"
        ]
    )

    counter_neg = hitung_counter(
        dataset_semua[
            dataset_semua["sentimen"] == "Negatif"
        ]
    )

    if label_pilih == "Positif":

        utama = counter_pos
        lain1 = counter_net
        lain2 = counter_neg

    elif label_pilih == "Netral":

        utama = counter_net
        lain1 = counter_pos
        lain2 = counter_neg

    else:

        utama = counter_neg
        lain1 = counter_pos
        lain2 = counter_net

    kata_dominan = {}

    for kata, freq in utama.items():

        if freq > lain1.get(kata,0) and freq > lain2.get(kata,0):

            kata_dominan[kata] = freq

    counter = Counter(kata_dominan)

top_word = (
    pd.DataFrame(
        counter.items(),
        columns=[
            "Kata",
            "Frekuensi"
        ]
    )
    .sort_values(
        "Frekuensi",
        ascending=False
    )
    .reset_index(drop=True)
)

if top_word.empty:

    st.warning("Tidak ada kata yang dapat divisualisasikan.")

    st.stop()
    # =====================================================
    # MEMBANGUN TEKS WORD CLOUD
    # =====================================================

    text_wordcloud = " ".join(

        [

            (kata + " ") * int(freq)

            for kata, freq in kata_dominan.items()

        ]

    )

    # =====================================================
    # MEMBUAT WORD CLOUD
    # =====================================================

    wc = WordCloud(

        width=1800,

        height=900,

        background_color="white",

        stopwords=stop_words,

        max_words=200,

        collocations=False,

        contour_width=1,

        contour_color="steelblue",

        colormap="viridis",

        random_state=42

    ).generate(text_wordcloud)

    # =====================================================
    # TAMPILKAN WORD CLOUD
    # =====================================================

    st.subheader(
        f"☁️ Word Cloud {dataset_pilih} - {label_pilih}"
    )

    fig, ax = plt.subplots(figsize=(18, 9))

    ax.imshow(
        wc,
        interpolation="bilinear"
    )

    ax.axis("off")

    st.pyplot(
        fig,
        use_container_width=True
    )

    # =====================================================
    # STATISTIK
    # =====================================================

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Jumlah Komentar",
        len(df)
    )

    c2.metric(
        "Jumlah Kata",
        len(semua_kata)
    )

    c3.metric(
        "Kata Unik",
        len(counter)
    )

    # =====================================================
    # LANJUT KE PART 3
    # =====================================================
# ============================================================
# PART 3
# GRAFIK, TABEL, DAN DOWNLOAD
# Tempelkan tepat setelah PART 2
# ============================================================

    st.divider()

    # =====================================================
    # TOP 20 KATA
    # =====================================================

    top10 = top_word.head(10).copy()

    st.subheader("📈 Top 10 Kata")

    fig_bar = px.bar(
        top10,
        x="Frekuensi",
        y="Kata",
        orientation="h",
        color="Frekuensi",
        text="Frekuensi",
        color_continuous_scale="viridis"
    )

    fig_bar.update_layout(
        title=f"Top 10 Kata ({dataset_pilih} - {label_pilih})",
        height=650,
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False
    )

    fig_bar.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # TABEL TOP 20
    # =====================================================

    st.subheader("📋 Tabel Frekuensi Kata")

    tampil = top10.copy()

    tampil.index = tampil.index + 1
    tampil.index.name = "No"

    st.dataframe(
        tampil,
        use_container_width=True,
        height=500
    )

    st.divider()

    # =====================================================
    # PENCARIAN KATA
    # =====================================================

    keyword = st.text_input(
        "🔍 Cari Kata",
        placeholder="Contoh: baterai"
    )

    if keyword:

        hasil = top_word[
            top_word["Kata"].str.contains(
                keyword,
                case=False,
                na=False
            )
        ]

        if hasil.empty:

            st.warning("Kata tidak ditemukan.")

        else:

            st.success(
                f"Ditemukan {len(hasil)} kata."
            )

            st.dataframe(
                hasil,
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    # =====================================================
    # DOWNLOAD TOP WORD
    # =====================================================

    csv = top10.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Top 10 Kata",
        data=csv,
        file_name=f"Top20_{dataset_pilih}_{label_pilih}.csv",
        mime="text/csv"
    )

    # =====================================================
    # LANJUT KE PART 4
    # =====================================================
# ============================================================
# PART 4
# DISTRIBUSI SENTIMEN, DATASET FILTER, DAN INSIGHT
# Tempelkan tepat setelah PART 3
# ============================================================

    st.divider()

    # =====================================================
    # DISTRIBUSI SENTIMEN
    # =====================================================

    st.subheader("📊 Distribusi Sentimen")

    sumber = word_yt if dataset_pilih == "YouTube" else word_tt

    distribusi = (
        sumber["sentimen"]
        .fillna("")
        .astype(str)
        .str.title()
        .value_counts()
        .reindex(["Positif", "Netral", "Negatif"], fill_value=0)
        .rename_axis("Sentimen")
        .reset_index(name="Jumlah")
    )

    fig_pie = px.pie(
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

    fig_pie.update_layout(
        height=500,
        legend_title="Sentimen"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # DATASET HASIL FILTER
    # =====================================================

    with st.expander("📄 Lihat Dataset Hasil Filter", expanded=False):

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=450
        )

    csv_filter = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Dataset Hasil Filter",
        data=csv_filter,
        file_name=f"Dataset_{dataset_pilih}_{label_pilih}.csv",
        mime="text/csv"
    )

    st.divider()

    # =====================================================
    # INSIGHT SINGKAT
    # =====================================================

    st.subheader("📝 Ringkasan")

    kata_utama = top10.iloc[0]["Kata"]
    frekuensi_utama = int(top10.iloc[0]["Frekuensi"])

    st.info(f"""
Dataset : **{dataset_pilih}**

Sentimen : **{label_pilih}**

Jumlah Komentar : **{len(df):,}**

Jumlah Kata : **{len(semua_kata):,}**

Jumlah Kata Unik : **{len(counter):,}**

Kata yang paling sering muncul adalah **{kata_utama}**
sebanyak **{frekuensi_utama}** kali.
""")

    st.success("✅ Visualisasi Word Cloud berhasil dibuat.")

    # =====================================================
    # LANJUT KE PART 5
    # =====================================================
# ============================================================
# PART 5
# PENUTUP HALAMAN WORD CLOUD
# Tempelkan tepat setelah PART 4
# ============================================================

    st.divider()

    # =====================================================
    # DOWNLOAD GAMBAR WORD CLOUD
    # =====================================================

    import io

    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

    buffer.seek(0)

    st.download_button(
        label="🖼️ Download Word Cloud (PNG)",
        data=buffer,
        file_name=f"WordCloud_{dataset_pilih}_{label_pilih}.png",
        mime="image/png"
    )

    st.divider()

    # =====================================================
    # INSIGHT OTOMATIS
    # =====================================================

    st.subheader("💡 Insight Word Cloud")

    lima_kata = ", ".join(top10["Kata"].head(5).tolist())

    st.success(f"""
Berdasarkan visualisasi Word Cloud pada dataset **{dataset_pilih}**
dengan label sentimen **{label_pilih}**, kata yang paling dominan adalah
**{kata_utama}** dengan frekuensi **{frekuensi_utama}** kemunculan.

Lima kata yang paling sering muncul adalah:
**{lima_kata}**.
""")

    st.divider()

    # =====================================================
    # KETERANGAN
    # =====================================================

    with st.expander("ℹ️ Keterangan Visualisasi"):

        st.markdown("""
- Word Cloud dibuat dari kolom **text_wordcloud**.
- Data telah difilter berdasarkan dataset dan label sentimen.
- Stopwords dihapus sebelum proses visualisasi.
- Frekuensi kata dihitung menggunakan **Counter**.
- Grafik Top 20 berasal dari hasil frekuensi yang sama dengan Word Cloud.
- Hasil dapat diunduh dalam format CSV maupun PNG.
""")

    st.divider()

    # =====================================================
    # FOOTER
    # =====================================================

    st.caption(
        "Dashboard Analisis Sentimen Mobil Listrik Indonesia | Word Cloud"
    )

# ============================================================
# END WORD CLOUD
# ============================================================

elif menu=="🔤 Top Words":
    st.title("Top Words")
    st.warning("Fitur akan dilengkapi pada Part 005.")

elif menu=="📈 Evaluasi Model":
    st.title("Evaluasi Model")
    st.warning("Fitur akan dilengkapi pada Part 006.")
