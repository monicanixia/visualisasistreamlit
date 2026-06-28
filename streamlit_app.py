"""
PART 001
streamlit_app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
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

elif menu=="☁️ Word Cloud":

    st.title("☁️ Word Cloud Sentimen")

    st.markdown("""
Visualisasi Word Cloud digunakan untuk melihat kata-kata yang paling
sering muncul berdasarkan dataset dan label sentimen.
""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        dataset_pilih = st.selectbox(
            "📂 Pilih Dataset",
            ["YouTube", "TikTok"]
        )

    with col2:
        label_pilih = st.selectbox(
            "😊 Pilih Sentimen",
            ["Positif", "Netral", "Negatif"]
        )

    st.markdown("---")

    if dataset_pilih == "YouTube":
        data = word_yt.copy()
    else:
        data = word_tt.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    if "sentimen" not in data.columns:
        st.error("Kolom 'sentimen' tidak ditemukan.")
        st.stop()

    if "text_wordcloud" not in data.columns:
        st.error("Kolom 'text_wordcloud' tidak ditemukan.")
        st.stop()

    data["sentimen"] = (
        data["sentimen"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )

    data = data[
        data["sentimen"] == label_pilih
    ].copy()

    if data.empty:
        st.warning(f"Tidak ada data sentimen {label_pilih}.")
        st.stop()

    text = " ".join(
        data["text_wordcloud"]
        .fillna("")
        .astype(str)
    )

    text = (
        text.replace("["," ")
            .replace("]"," ")
            .replace("'"," ")
            .replace('"'," ")
            .replace(","," ")
    )

    text = " ".join(text.split())

    stop_words = {
        "mobil","listrik","kendaraan","ev","indonesia",
        "yg","nya","aja","nih","sih","bang","bro",
        "deh","dong","kan","aku","saya","kami",
        "kita","orang","buat","jadi","karena",
        "kalau","kalo","lebih","masih","cukup",
        "banyak","itu","ini","the","and"
    }

    # Part 2 akan dimulai dari pembuatan WordCloud.

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
    ).generate(text)

    st.subheader(f"☁️ Word Cloud {dataset_pilih} - {label_pilih}")

    fig, ax = plt.subplots(figsize=(18,9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # =====================================================
    # HITUNG FREKUENSI KATA
    # =====================================================

    from collections import Counter

    semua_kata = []

    for kalimat in data["text_wordcloud"]:

        tokens = (
            str(kalimat)
            .lower()
            .replace("["," ")
            .replace("]"," ")
            .replace("'"," ")
            .replace(","," ")
            .split()
        )

        semua_kata.extend(
            [
                t for t in tokens
                if len(t) > 2
                and t not in stop_words
            ]
        )

    counter = Counter(semua_kata)

    top_word = (
        pd.DataFrame(
            counter.items(),
            columns=["Kata","Frekuensi"]
        )
        .sort_values(
            "Frekuensi",
            ascending=False
        )
        .reset_index(drop=True)
    )

    if top_word.empty:
        st.warning("Tidak ada kata yang dapat ditampilkan.")
        st.stop()

    c1,c2,c3 = st.columns(3)

    c1.metric("Jumlah Komentar", len(data))
    c2.metric("Jumlah Kata", len(semua_kata))
    c3.metric("Jumlah Kata Unik", len(counter))

    st.markdown("---")

    top20 = top_word.head(20)

    st.subheader("📈 Top 20 Kata")

    fig2 = px.bar(
        top20,
        x="Frekuensi",
        y="Kata",
        orientation="h",
        color="Frekuensi",
        text="Frekuensi",
        color_continuous_scale="viridis"
    )

    fig2.update_layout(
        title=f"Top 20 Kata ({dataset_pilih} - {label_pilih})",
        height=700,
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="total ascending")
    )

    fig2.update_traces(textposition="outside")

    st.plotly_chart(fig2, use_container_width=True)

    # Part 3 dimulai dari tabel, download, dan insight.

    # =====================================================
    # TABEL TOP 20 KATA
    # =====================================================

    st.subheader("📋 Tabel Top 20 Kata")

    tampil_top = top20.copy()
    tampil_top.index = tampil_top.index + 1
    tampil_top.index.name = "No"

    st.dataframe(
        tampil_top,
        use_container_width=True,
        hide_index=False,
        height=450
    )

    st.markdown("---")

    # =====================================================
    # PENCARIAN KATA
    # =====================================================

    st.subheader("🔍 Cari Kata")

    keyword = st.text_input(
        "Masukkan kata yang ingin dicari",
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
            st.success(f"Ditemukan {len(hasil)} kata.")
            st.dataframe(
                hasil,
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")

    # =====================================================
    # DOWNLOAD TOP WORD
    # =====================================================

    csv = top20.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Top 20 Kata",
        data=csv,
        file_name=f"Top20_{dataset_pilih}_{label_pilih}.csv",
        mime="text/csv"
    )

    st.markdown("---")

    

# ============================================================
# TOPWORDS_FULL_PART1.py
# Tempelkan menggantikan:
# elif menu=="🔤 Top Words":
# ============================================================

elif menu=="🔤 Top Words":

    st.title("🔤 Top Words")

    st.markdown("""
Halaman ini menampilkan kata yang paling sering muncul berdasarkan
dataset dan label sentimen.
""")

    st.divider()

    # =====================================================
    # PILIH DATASET DAN SENTIMEN
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:
        dataset_pilih = st.selectbox(
            "📂 Pilih Dataset",
            ["YouTube", "TikTok"],
            key="tw_dataset"
        )

    with col2:
        label_pilih = st.selectbox(
            "😊 Pilih Sentimen",
            ["Positif", "Netral", "Negatif"],
            key="tw_sentimen"
        )

    st.divider()

    # =====================================================
    # AMBIL DATASET
    # =====================================================

    if dataset_pilih == "YouTube":
        data = word_yt.copy()
    else:
        data = word_tt.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
    )

    if "text_wordcloud" not in data.columns:
        st.error("Kolom text_wordcloud tidak ditemukan.")
        st.stop()

    if "sentimen" not in data.columns:
        st.error("Kolom sentimen tidak ditemukan.")
        st.stop()

    data["sentimen"] = (
        data["sentimen"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.title()
    )

    data = data[
        data["sentimen"] == label_pilih
    ].copy()

    if data.empty:
        st.warning("Tidak ada data yang sesuai.")
        st.stop()

    # =====================================================
    # STOPWORDS
    # =====================================================

    stop_words = {
        "mobil","listrik","kendaraan","indonesia","ev",
        "yg","nya","aja","nih","kan","dong","sih",
        "aku","saya","kami","kita","orang",
        "buat","jadi","karena","kalau","kalo",
        "lebih","masih","cukup","banyak",
        "itu","ini","the","and"
    }

    # =====================================================
    # LANJUT KE TopWords_Full_Part2.py
    # =====================================================
# ============================================================
# TOPWORDS_FULL_PART2.py
# Tempelkan tepat setelah TopWords_Full_Part1.py
# ============================================================

    from collections import Counter

    # =====================================================
    # TOKENISASI
    # =====================================================

    semua_kata = []

    for kalimat in data["text_wordcloud"]:

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

    counter = Counter(semua_kata)

    top_word = (
        pd.DataFrame(
            counter.items(),
            columns=["Kata", "Frekuensi"]
        )
        .sort_values(
            by="Frekuensi",
            ascending=False
        )
        .reset_index(drop=True)
    )

    if top_word.empty:
        st.warning("Tidak ada kata yang dapat ditampilkan.")
        st.stop()

    # =====================================================
    # PILIH JUMLAH KATA
    # =====================================================
    jumlah = 20

    top_tampil = top_word.head(jumlah).copy()

    st.metric("Jumlah Kata Unik", len(counter))

    st.divider()

    # =====================================================
    # TABEL TOP WORDS
    # =====================================================

    st.subheader(f"📋 Top {jumlah} Kata")

    tampil = top_tampil.copy()
    tampil.index = tampil.index + 1
    tampil.index.name = "No"

    st.dataframe(
        tampil,
        use_container_width=True,
        height=500
    )

    # =====================================================
    # PENCARIAN KATA
    # =====================================================

    cari = st.text_input(
        "🔍 Cari Kata",
        key="tw_cari"
    )

    if cari:

        hasil = top_word[
            top_word["Kata"].str.contains(
                cari,
                case=False,
                na=False
            )
        ]

        st.dataframe(
            hasil,
            use_container_width=True,
            hide_index=True
        )

    # =====================================================
    # LANJUT KE TopWords_Full_Part3.py
    # =====================================================
# ============================================================
# TOPWORDS_FULL_PART3.py
# Tempelkan tepat setelah TopWords_Full_Part2.py
# ============================================================

    st.divider()

    # =====================================================
    # GRAFIK HORIZONTAL
    # =====================================================

    st.subheader(f"📊 Grafik Top {jumlah} Kata")

    fig_bar = px.bar(
        top_tampil,
        x="Frekuensi",
        y="Kata",
        orientation="h",
        text="Frekuensi",
        color="Frekuensi",
        color_continuous_scale="viridis"
    )

    fig_bar.update_layout(
        title=f"Top {jumlah} Kata ({dataset_pilih} - {label_pilih})",
        height=650,
        yaxis=dict(categoryorder="total ascending"),
        coloraxis_showscale=False
    )

    fig_bar.update_traces(textposition="outside")

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # GRAFIK VERTIKAL
    # =====================================================

    st.subheader("📈 Grafik Vertikal")

    fig_col = px.bar(
        top_tampil,
        x="Kata",
        y="Frekuensi",
        text="Frekuensi",
        color="Frekuensi",
        color_continuous_scale="viridis"
    )

    fig_col.update_layout(
        height=550,
        xaxis_title="Kata",
        yaxis_title="Frekuensi",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig_col,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # METRIK
    # =====================================================

    c1, c2, c3 = st.columns(3)

    c1.metric("Jumlah Komentar", len(data))
    c2.metric("Jumlah Kata", len(semua_kata))
    c3.metric("Kata Unik", len(counter))

    st.info(
        f"Kata paling sering muncul adalah **{top_tampil.iloc[0]['Kata']}** "
        f"dengan frekuensi **{int(top_tampil.iloc[0]['Frekuensi'])}**."
    )

    # =====================================================
    # LANJUT KE TopWords_Full_Part4.py
    # =====================================================
# ============================================================
# TOPWORDS_FULL_PART4.py
# Tempelkan tepat setelah TopWords_Full_Part3.py
# ============================================================

    st.divider()

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    csv = top_tampil.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Top Words (CSV)",
        data=csv,
        file_name=f"TopWords_{dataset_pilih}_{label_pilih}.csv",
        mime="text/csv"
    )

    # =====================================================
    # DOWNLOAD EXCEL
    # =====================================================

    import io

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        top_tampil.to_excel(
            writer,
            index=False,
            sheet_name="Top Words"
        )

    excel_buffer.seek(0)

    st.download_button(
        label="📄 Download Top Words (Excel)",
        data=excel_buffer,
        file_name=f"TopWords_{dataset_pilih}_{label_pilih}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()


# ==========================================================
# EvaluasiModel_Full_Part1.py
# Tempelkan menggantikan:
# elif menu=="📈 Evaluasi Model":
# ==========================================================

elif menu=="📈 Evaluasi Model":

    st.title("📈 Evaluasi Model")

    st.markdown("""
Halaman ini menampilkan hasil evaluasi model klasifikasi
sentimen menggunakan **XGBoost** dan **LSTM**.
""")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        dataset_pilih = st.selectbox(
            "📂 Dataset",
            ["YouTube","TikTok"],
            key="eval_dataset"
        )

    with col2:
        model_pilih = st.selectbox(
            "🤖 Model",
            ["XGBoost","LSTM"],
            key="eval_model"
        )

    with col3:
        split_pilih = st.selectbox(
            "📊 Data Split",
            ["80:20","70:30","60:40"],
            key="eval_split"
        )

    st.divider()

    if model_pilih=="XGBoost":

        if dataset_pilih=="YouTube":
            df = pd.read_excel("ringkasan_xgboost_Youtube.xlsx")
        else:
            df = pd.read_excel("ringkasan_xgboost_Tiktok.xlsx")

    else:

        if dataset_pilih=="YouTube":
            df = pd.read_excel("ringkasan_lstm_Youtube.xlsx")
        else:
            df = pd.read_excel("ringkasan_lstm_Tiktok.xlsx")

    df.columns = df.columns.str.strip()

    if "Split" in df.columns:
        hasil = df[df["Split"].astype(str)==split_pilih].copy()
    else:
        hasil = df.copy()

    if hasil.empty:
        st.warning("Data evaluasi tidak ditemukan.")
        st.stop()

    st.success(
        f"Dataset : {dataset_pilih} | Model : {model_pilih} | Split : {split_pilih}"
    )

    # LANJUT KE EvaluasiModel_Full_Part2.py
# ==========================================================
# EvaluasiModel_Full_Part2.py
# Tempelkan setelah EvaluasiModel_Full_Part1.py
# ==========================================================

    st.subheader("📋 Hasil Evaluasi Model")

    st.dataframe(
        hasil,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =====================================================
    # AMBIL METRIK
    # =====================================================

    def ambil_nilai(kolom):
        if kolom in hasil.columns:
            return float(hasil.iloc[0][kolom])
        return 0.0

    accuracy = ambil_nilai("Accuracy")
    precision = ambil_nilai("Precision")
    recall = ambil_nilai("Recall")
    f1score = ambil_nilai("F1-Score")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🎯 Accuracy",
        f"{accuracy:.2%}" if accuracy <= 1 else f"{accuracy:.2f}%"
    )

    c2.metric(
        "📌 Precision",
        f"{precision:.2%}" if precision <= 1 else f"{precision:.2f}%"
    )

    c3.metric(
        "📍 Recall",
        f"{recall:.2%}" if recall <= 1 else f"{recall:.2f}%"
    )

    c4.metric(
        "🏆 F1-Score",
        f"{f1score:.2%}" if f1score <= 1 else f"{f1score:.2f}%"
    )

    st.divider()

    # LANJUT KE EvaluasiModel_Full_Part3.py
# ==========================================================
# EvaluasiModel_Full_Part3.py
# Tempelkan setelah EvaluasiModel_Full_Part2.py
# ==========================================================

    st.divider()

    st.subheader("🖼️ Confusion Matrix")

    split_file = split_pilih.replace(":", "_")

# Samakan dengan nama file di GitHub
    if dataset_pilih == "YouTube":
        dataset_file = "Youtube"
    else:
        dataset_file = "Tiktok"

    if model_pilih == "XGBoost":
        prefix = "cm_xgboost"
    else:
        prefix = "cm_lstm"

    nama_file = f"{prefix}_{split_file}_{dataset_file}.png"

# Debug (boleh dihapus nanti)
    st.write("Nama file :", nama_file)

if os.path.exists(nama_file):

    st.image(
        nama_file,
        caption=f"Confusion Matrix {model_pilih} | {dataset_pilih} | {split_pilih}",
        use_container_width=True
    )

    with open(nama_file, "rb") as img:

        st.download_button(
            "⬇️ Download Confusion Matrix",
            data=img,
            file_name=nama_file,
            mime="image/png"
        )

else:

    st.error(f"File tidak ditemukan : {nama_file}")

    st.divider()

    st.subheader("📈 Interpretasi Singkat")

    st.info(
        f'''
Model **{model_pilih}** pada dataset **{dataset_pilih}**
dengan pembagian data **{split_pilih}**
ditampilkan melalui confusion matrix di atas.
Diagonal utama menunjukkan jumlah prediksi yang benar,
sedangkan sel lainnya menunjukkan kesalahan klasifikasi.
'''
    )

    # LANJUT KE EvaluasiModel_Full_Part4.py
# ==========================================================
# EvaluasiModel_Full_Part4.py
# Tempelkan setelah EvaluasiModel_Full_Part3.py
# ==========================================================

    st.divider()

    st.divider()

    st.subheader("🎯 Ringkasan Hasil")

    terbaik = metric_df.loc[
        metric_df["Nilai"].idxmax(),
        "Metrik"
    ]

    st.success(f'''
Dataset : **{dataset_pilih}**

Model : **{model_pilih}**

Split : **{split_pilih}**

Metrik dengan nilai tertinggi adalah **{terbaik}**.
''')

    with st.expander("ℹ️ Keterangan Metrik"):

        st.markdown("""
- **Accuracy** : Persentase prediksi yang benar.
- **Precision** : Ketepatan prediksi pada setiap kelas.
- **Recall** : Kemampuan model menemukan seluruh data pada suatu kelas.
- **F1-Score** : Rata-rata harmonis antara Precision dan Recall.
""")

    # =====================================================
    # LANJUT KE EvaluasiModel_Full_Part5.py
    # =====================================================
# ==========================================================
# EvaluasiModel_Full_Part5.py
# Tempelkan setelah EvaluasiModel_Full_Part4.py
# ==========================================================

    st.divider()

    st.subheader("💾 Download Hasil Evaluasi")

    csv = hasil.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Hasil (CSV)",
        data=csv,
        file_name=f"Evaluasi_{model_pilih}_{dataset_pilih}_{split_pilih}.csv",
        mime="text/csv"
    )

    import io

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        hasil.to_excel(
            writer,
            index=False,
            sheet_name="Evaluasi Model"
        )

    excel_buffer.seek(0)

    st.download_button(
        label="📄 Download Hasil (Excel)",
        data=excel_buffer,
        file_name=f"Evaluasi_{model_pilih}_{dataset_pilih}_{split_pilih}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()

    st.subheader("📝 Kesimpulan")

    st.success(
        f"""
Model **{model_pilih}** pada dataset **{dataset_pilih}**
dengan pembagian data **{split_pilih}** menghasilkan:

- Accuracy : **{accuracy:.4f}**
- Precision : **{precision:.4f}**
- Recall : **{recall:.4f}**
- F1-Score : **{f1score:.4f}**

Hasil ini dapat digunakan sebagai dasar untuk
membandingkan performa model pada berbagai skenario
pengujian.
"""
    )

    st.divider()

    st.caption(
        "Dashboard Analisis Sentimen Mobil Listrik Indonesia | Evaluasi Model"
    )

# ==========================================================
# END EVALUASI MODEL
# ==========================================================

