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
"""
WordCloud_Full_Part2.py

Lanjutan setelah WordCloud_Full_Part1.py
"""

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
"""
WordCloud_Full_Part3.py

Lanjutan setelah WordCloud_Full_Part2.py
"""

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

    # =====================================================
    # DISTRIBUSI SENTIMEN
    # =====================================================

    st.subheader("📊 Distribusi Sentimen")

    distribusi = (
        (word_yt if dataset_pilih == "YouTube" else word_tt)["sentimen"]
        .value_counts()
        .rename_axis("Sentimen")
        .reset_index(name="Jumlah")
    )

    fig3 = px.pie(
        distribusi,
        names="Sentimen",
        values="Jumlah",
        hole=0.45,
        color="Sentimen",
        color_discrete_map={
            "Positif":"#2ecc71",
            "Netral":"#f1c40f",
            "Negatif":"#e74c3c"
        }
    )

    fig3.update_layout(height=500)

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # DATASET HASIL FILTER
    # =====================================================

    with st.expander("📄 Dataset Hasil Filter"):

        st.dataframe(
            data,
            use_container_width=True,
            hide_index=True,
            height=450
        )

    csv_filter = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Dataset Filter",
        data=csv_filter,
        file_name=f"Dataset_{dataset_pilih}_{label_pilih}.csv",
        mime="text/csv"
    )

    st.markdown("---")

    st.info(f"""
**Dataset:** {dataset_pilih}

**Sentimen:** {label_pilih}

**Jumlah Komentar:** {len(data):,}

**Jumlah Kata:** {len(semua_kata):,}

**Jumlah Kata Unik:** {len(counter):,}

**Kata Terbanyak:** {top20.iloc[0]["Kata"]}

**Frekuensi:** {top20.iloc[0]["Frekuensi"]}
""")

    st.success("✅ Visualisasi Word Cloud berhasil dibuat.")

    # Part 4:
    # Optimasi agar hanya menampilkan kata yang dominan
    # pada sentimen yang dipilih (recommended).
"""

WordCloud_Full_Part4.py



Optimasi Word Cloud agar lebih representatif terhadap sentimen.

Letakkan setelah proses filtering data dan SEBELUM membuat WordCloud.

Bagian ini menggantikan proses pembentukan `semua_kata`.

"""



    # =====================================================

    # TOKENISASI & FREKUENSI PER SENTIMEN

    # =====================================================



    from collections import Counter



    def ekstrak_kata(df):



        kata = []



        for kalimat in df["text_wordcloud"].fillna("").astype(str):



            token = (

                kalimat.lower()

                .replace("["," ")

                .replace("]"," ")

                .replace("'"," ")

                .replace(","," ")

                .split()

            )



            token = [

                t for t in token

                if len(t) > 2

                and t not in stop_words

            ]



            kata.extend(token)



        return Counter(kata)



    yt_counter = ekstrak_kata(word_yt)

    tt_counter = ekstrak_kata(word_tt)



    gabung = word_yt if dataset_pilih == "YouTube" else word_tt



    pos_counter = ekstrak_kata(

        gabung[gabung["sentimen"]=="Positif"]

    )



    net_counter = ekstrak_kata(

        gabung[gabung["sentimen"]=="Netral"]

    )



    neg_counter = ekstrak_kata(

        gabung[gabung["sentimen"]=="Negatif"]

    )



    if label_pilih == "Positif":

        target = pos_counter

        lawan1 = net_counter

        lawan2 = neg_counter

    elif label_pilih == "Netral":

        target = net_counter

        lawan1 = pos_counter

        lawan2 = neg_counter

    else:

        target = neg_counter

        lawan1 = pos_counter

        lawan2 = net_counter



    kata_dominan = {}



    for kata, freq in target.items():



        if freq >= max(lawan1.get(kata,0), lawan2.get(kata,0)):

            kata_dominan[kata] = freq



    if len(kata_dominan) == 0:

        kata_dominan = dict(target)



    text = " ".join(

        [

            (kata + " ") * freq

            for kata, freq in kata_dominan.items()

        ]

    )



    semua_kata = []



    for kata, freq in kata_dominan.items():

        semua_kata.extend([kata]*freq)



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



    st.info(

        "ℹ️ Word Cloud menggunakan kata yang dominan pada sentimen yang dipilih sehingga kata yang lebih identik dengan sentimen lain akan dikurangi."

    )

"""
WordCloud_Full_Part5.py

Bagian penutup halaman Word Cloud.
Gunakan setelah seluruh visualisasi selesai.
"""

    # =====================================================
    # INSIGHT OTOMATIS
    # =====================================================

    st.markdown("---")
    st.subheader("📝 Insight Otomatis")

    if not top_word.empty:

        lima_besar = ", ".join(top_word.head(5)["Kata"].tolist())

        st.success(
            f"""
Pada dataset **{dataset_pilih}** dengan sentimen **{label_pilih}**,
kata yang paling dominan adalah **{top_word.iloc[0]['Kata']}**
sebanyak **{int(top_word.iloc[0]['Frekuensi'])}** kemunculan.

Lima kata yang paling sering muncul:
**{lima_besar}**
"""
        )

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
        file_name=f"wordcloud_{dataset_pilih}_{label_pilih}.png",
        mime="image/png"
    )

    st.markdown("---")

    # =====================================================
    # KETERANGAN
    # =====================================================

    with st.expander("ℹ️ Keterangan"):

        st.markdown("""
Word Cloud dibuat berdasarkan komentar yang telah difilter sesuai
dataset dan label sentimen.

Versi optimasi ini memprioritaskan kata yang lebih dominan pada
sentimen yang dipilih sehingga visualisasi menjadi lebih representatif.

Visualisasi ini digunakan sebagai analisis deskriptif terhadap
kecenderungan kata pada masing-masing sentimen.
""")

    # =====================================================
    # FOOTER
    # =====================================================

    st.caption(
        "Analisis Sentimen Mobil Listrik Indonesia • Streamlit Dashboard"
    )

# ============================================================
# WORDCLOUD_FULL_PART6.py
# OPSIONAL - PENYEMPURNAAN HALAMAN WORD CLOUD
# Tempelkan setelah WordCloud_Full_Part5.py
# ============================================================

    st.divider()

    # =====================================================
    # TOP 30 KATA
    # =====================================================

    st.subheader("📌 Top 30 Kata")

    top30 = top_word.head(30).copy()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.dataframe(
            top30,
            use_container_width=True,
            hide_index=True,
            height=500
        )

    with col2:
        st.metric("Top Kata", top30.iloc[0]["Kata"])
        st.metric("Frekuensi", int(top30.iloc[0]["Frekuensi"]))
        st.metric("Jumlah Top Kata", len(top30))

    st.divider()

    # =====================================================
    # DOWNLOAD TOP 30
    # =====================================================

    csv30 = top30.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Top 30 Kata",
        data=csv30,
        file_name=f"Top30_{dataset_pilih}_{label_pilih}.csv",
        mime="text/csv"
    )

    st.divider()

    # =====================================================
    # INFORMASI DATASET
    # =====================================================

    with st.expander("ℹ️ Informasi Dataset"):

        st.write(f"Dataset : **{dataset_pilih}**")
        st.write(f"Sentimen : **{label_pilih}**")
        st.write(f"Jumlah komentar : **{len(df):,}**")
        st.write(f"Jumlah kata : **{len(semua_kata):,}**")
        st.write(f"Jumlah kata unik : **{len(counter):,}**")

    st.success("✅ Halaman Word Cloud selesai dimuat.")


elif menu=="🔤 Top Words":
    st.title("Top Words")
    st.warning("Fitur akan dilengkapi pada Part 005.")

elif menu=="📈 Evaluasi Model":
    st.title("Evaluasi Model")
    st.warning("Fitur akan dilengkapi pada Part 006.")
