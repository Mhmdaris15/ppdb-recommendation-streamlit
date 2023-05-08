import streamlit as st
from PIL import Image
from io import BytesIO
import base64

# img1= Image.open("img/aris.jpg")
# img2= Image.open("img/ghufron.jpg")
# img3= Image.open("img/Arip.jpg")

with st.container():

    col1, col2, col3 = st.columns([1, 2, 1])
    
    # Fungsi untuk menampilkan gambar
    # def display_image(image_path, size):
    #     img = Image.open(image_path)
    #     img = img.resize(size)
    #     st.image(img)

    # Fungsi untuk menampilkan nama dan deskripsi anggota tim
    def display_team_member(name, role, description, image_path, size):
        with st.container():
            st.markdown(f"<h2 style='text-align: left'>{name}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: left'>{role}</h3>", unsafe_allow_html=True)
            st.write(description)
    #Fungsi untuk menampilkan gambar lingkaran
    def display_circle_image(image_path, size):
        img = Image.open(image_path)
        img = img.resize(size)
        st.markdown(f'<img src="data:image/png;base64,{image_to_base64(img)}" style="border-radius: 50%;ali-items:center;"/>', unsafe_allow_html=True)

    # Fungsi untuk mengubah gambar menjadi format base64
    def image_to_base64(image):
        with BytesIO() as buffer:
            image.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()
    # Judul halaman
    st.markdown("<h1 style='text-align: left'>Our Team</h1>", unsafe_allow_html=True)

    # Daftar anggota tim
    display_circle_image("img/aris.jpg", (200, 200))
    display_team_member(
        name="Muhammad Aris SeptaNugroho",
        role="Leader Team",
        description="Orang yang bertanggung jawab  atas semua projek tim kami terutama Kaggle.",
        image_path="img/logo.png",
        size=(200, 200)
    )

    st.markdown(""" """)

    display_circle_image("img/ghufron.jpg", (200, 200))
    display_team_member(
        name="Muhammad Ghufron",
        role="Member Team",
        description="Orang yang bertanggung jawab atas notebook data rill.",
        image_path="img/logo2.png",
        size=(200, 200)
    )
    
    st.markdown(""" """)

    display_circle_image("img/Arip.jpg", (200, 200))
    display_team_member(
        name="Muhammad Ariffanka",
        role="Member Team",
        description="Orang yang bertanggung jawab atas pembuatan web dokumentasi.",
        image_path="img/logo2.png",
        size=(200, 200)
    )
 
