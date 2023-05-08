import streamlit as st

def main():
    # Judul Halaman
    st.title("Contact Us")

    # Gambar Header
    header_image = "img/logo.png"
    st.image(header_image, width=90)

    # Konten Halaman
    st.write("""
    We'd love to hear from you! You can follow us on Instagram or reach out to us using the contact information below:
    """)

    # Tombol Instagram
    st.markdown(
        """
        <div style="text-align:center">
        <a href='https://www.instagram.com/nevtikacademy/' target="_blank">
           Instagram
        </a>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
