import streamlit as st
from PIL import Image

image = Image.open("img/logo2.png")
logo = Image.open("img/logo.png")
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    col2.image(image, use_column_width=True)
    col2.markdown("""
        <h1 class="title" style='text-align: center;'>SMKN 1 Cibinong</h1>
        <div class="des-title" style='text-align: center;'>
        SMKN 1 Cibinong merupakah salah satu sekolah unggulan di daerah Kabupaten Bogor. Dan kami merupakan 
        murid yang berasal dari SMKN 1 Cibinong. Tim kami bernama 'SMKN 1 Cibinong NEVTIK Team', yang
        beranggotakan Aris S.P., M.Ghufron,& M.Ariffanka.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    col2.image(logo, use_column_width=True)
    col2.markdown("""
        <h1 class="title" style='text-align: center;'>NEVTIK</h1>
        <div class="des-title" style='text-align: center;'>NEVTIK adalah salah satu organisasi yang ada di SMKN 1 Cibinong
        yang bertujuan untuk mewadahi siswa-siswi SMKN 1 Cibinong yang ingin meningkatkan skill di bidang IT.
        Kami bertiga merupakan bagian dari organisasi tersebut.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    col2.markdown("""
                  """)
    
    col2.markdown("""
        <div class="des-title" style='text-align: center;'>Pada kesempatan kali kami membuat web untuk menerapkan
        analisis kami pada notebook dan juga untuk menerapkan model yang telah di buat pada notebook projek data rill kami.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    col2.markdown("""
        <div class="des-title" style='text-align: center;'>Projek data rill kami adalah membuat prediksi chances keberhasilan
        orang-orang yang ingin mendaftar ke SMKN 1 Cibinong. Nantinya orang-orang akan menginputkan nilainya pada page demo
        untuk megecek chances dirinya agar bisa berksekolah di SMKN 1 Cibinong. Kami menggunakan data PPDB 2020-2021 sebagai
        bahan analisis kami.</div>
        <br><br>
    """, unsafe_allow_html=True)
    \
        col2.markdown("""
        <div class="des-title" style='text-align: center;'>Web ini terdiri dari beberapa page yakni, landing page untuk 
        penjelasan awal sekaligus perkenalan mengenai projek kami, dashboard page berisi analisis data(visualisasi data), & 
        demo page untuk mengakses model kami,our team page untuk memberi tahu siapa saja yang terkait dalam projek ini, & contact us untuk 
        menhubungi pihak kami jika ada sesuatu.</div>
        <br><br>
    """, unsafe_allow_html=True)
