import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from prettytable import PrettyTable
import plotly.figure_factory as ff
from IPython.display import display
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from scipy.stats.mstats import winsorize
from streamlit import session_state as state

ppdb21= pd.read_csv("ppdb_2021.csv")
ppdb20= pd.read_csv("ppdb_2020.csv")
ppdb22= pd.read_csv("ppdb_2022.csv")


# with st.sidebar:
#     # st.markdown("# Home page 🎈")
#     # st.write("Hello, I'm a Streamlit app!")
#     # knowledge = st.selectbox("Do you have a good knowledge in Cybersecurity?", ["Yes", "No"])
#     # if knowledge == "Yes":
#     #     st.write("You're guaranted can surfing on Internet safely!")
#     # else:
#     #     st.markdown("""
#     #     Please, read the following article:\n
#     #     [SNYK Cybersecurity](https://learn.snyk.io/lessons/)
#     #     """)

    
with st.container():
    # data.info()
    logo, title_text, logo1 = st.columns([1,3,1])
    
    logo.image("img/logo.png",width=70)
    logo1.image("img/logo1.png",width=70)
    
    title_text.markdown("""
    <div class="header-logo" style="display: flex; justify-content:space-around; align-items:center;">
            <img src="img/logo.png" alt="">
            <h3><span>PPDB</span>-<span style="color: lightskyblue;">SMKN 1 Cibinong</span></h3>
            <img src="img/logo1.png" alt="">
    </div>
    """,unsafe_allow_html=True)
    
    st.markdown("""
        <h1 class="title" style='text-align: center;'>Projek Kelompok NEVTIK SMKN 1 Cibinong!</h1>
        <div class="des-title" style='text-align: center;'>Projek kelompok NEVTIK ialah pengolahan data PPDB 2022 SMKN 1 Cibinong</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    
    
    
    
    #hapus Unnamed
    ppdb22 = ppdb22.loc[:, ~ppdb22.columns.str.contains('^Unnamed')]
    ppdb22 = ppdb22.rename(columns={'Jarak ': 'Jarak', 'Pilihan ': 'Pilihan'})
    
    ppdb20 = ppdb20.rename(columns=lambda x: x.lower().replace(' ', '_'))
    ppdb21 = ppdb21.rename(columns=lambda x: x.lower().replace(' ', '_'))
    ppdb22 = ppdb22.rename(columns=lambda x: x.lower().replace(' ', '_'))
    
    #samain kolom
    print(set(ppdb22 == ppdb21.columns))
    print(set(ppdb22 == ppdb20.columns))
    
    #format penamaan buat rename
    rename_cols = lambda col_name: "_".join(re.split("/| ", col_name.lower())) if len(re.split("/| ", col_name.lower())) > 0 else col_name.lower()

    
    #merubah data dll
    for year in range(20, 23):
        exec(f"ppdb{year}.insert(3, 'tahun_diterima', {year})") # insert year column to each dataset
        exec(f"ppdb{year}.rename(rename_cols, axis='columns', inplace=True)")
        exec(f"ppdb{year}.loc[:, 'agama1':'skor'] = ppdb{year}.loc[:, 'agama1':'skor'].astype(float)") # convert data type
        exec(f"ppdb{year}['tanggal_lahir'] = pd.to_datetime(ppdb{year}['tanggal_lahir'])") # convert to date time type
    
    ppdb = pd.concat([ppdb20, ppdb21, ppdb22], ignore_index=True)
    
    #remove outliers ppdb2020
    Q1 = ppdb20['skor'].quantile(0.25)
    Q3 = ppdb20['skor'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    ppdb20 = ppdb20[(ppdb20['skor'] > lower_limit) & (ppdb20['skor'] < upper_limit)]
    
    #remove outliers ppdb2021
    Q1 = ppdb21['skor'].quantile(0.25)
    Q3 = ppdb21['skor'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    ppdb21 = ppdb21[(ppdb21['skor'] > lower_limit) & (ppdb21['skor'] < upper_limit)]
    
    #remove outliers ppdb2023
    Q1 = ppdb22['skor'].quantile(0.25)
    Q3 = ppdb22['skor'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    ppdb22 = ppdb22[(ppdb22['skor'] > lower_limit) & (ppdb22['skor'] < upper_limit)]
    
    comp_acro = []
    for comp in ppdb.pilihan:
        clean_comp = comp.replace('DAN','').replace(',','') # remove any conjunction
        acro = ""
    
    fragments = clean_comp.split()
    for frag in fragments:
        acro += frag[0] # take first letter for each word
        if len(fragments) <= 1:
            acro *= 2
            
    comp_acro.append(acro)
    
    # ppdb['pilihan'] = comp_acro
    
    st.markdown("""
        <div class="des-title" style='text-align: center;'>Data di bawah ini merupakan data siswa-siswi yang mengikuti PPDB<br>
        dan data siswa-siswi di bawah ini merupakan data hasil gabungan PPDB 2020-2022 yang sudah diolah sedemikian rupa.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    ppdb
    
    # hitung jumlah yang daftar
    n_regs = ppdb.tahun_diterima.value_counts().reset_index()
    n_regs.columns = ['year', 'registrants']

    # pakai chart yang berbentuk lingkaran dengan plotly
    fig = px.pie(n_regs, values='registrants', names='year', color='year',
             color_discrete_sequence=['greenyellow', 'slateblue', 'red'])

    # atur title dan legend
    fig.update_layout(
    title={
        'text': "Jummlah pendaftar tahun 2020, 2021 dan 2022",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 16}
    },
    legend={
        'y':0.5,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'middle'
    }
)

    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'>Setelah melihat data di atas kini kami akan menampilkan jumlah<br>
        pendaftar dari masing-masing tahun PPDB diadakan.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    #menampilkannya dengan plotly streamlit
    st.plotly_chart(fig)
    
    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'>Dari data di atas terdapat sebuah ketimpangan(karena banyaknya<br>
        data pendaftar pada thaun 2022)pada 2022, namun data pendaftar tahun 2020 dan 2021 tidak berbeda jauh.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    # menghitung jumlah siswa per jenis kelamin dan tahun
    ppdb_grouped = ppdb.groupby(['tahun_diterima', 'jenis_kelamin']).size().reset_index(name='jumlah_siswa')

    # membuat visualisasi histogram
    fig = px.histogram(ppdb_grouped, x='jenis_kelamin', y='jumlah_siswa', color='tahun_diterima', barmode='group',
                    color_discrete_sequence=['#0077c2', '#b2301d', '#f6a800'])
    fig.update_layout(
        title={
            'text': "Jumlah Siswa PPDB Berdasarkan Jenis Kelamin per Tahunnya",
            'x':0.5,
            'y':0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Jenis Kelamin",
        yaxis_title="Jumlah Siswa",
        font=dict(size=12)
    )

    st.plotly_chart(fig)
    
    # menghitung jumlah siswa per provinsi dan tahun diterima
    ppdb_grouped = ppdb.groupby(['provinsi', 'tahun_diterima'])['nama_pendaftar'].count().reset_index()

    # mengatur kombinasi warna
    color_map = {'2020': '#0077c2', '2021': '#b2301d', '2022': '#f6a800'}

    fig = px.histogram(ppdb_grouped[ppdb_grouped['tahun_diterima'] != 20.5][ppdb_grouped['tahun_diterima'] != 21.5], 
                    x='provinsi', 
                    y='nama_pendaftar', 
                    color='tahun_diterima', 
                    barmode='group',
                    color_discrete_map=color_map)

    fig.update_layout(
        title={
            'text': "Jumlah Siswa PPDB Berdasarkan Provinsi Asal per Tahunnya",
            'x':0.5,
            'y':0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Provinsi Asal",
        yaxis_title="Jumlah Siswa",
        font=dict(size=12)
    )
    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'>Pada data ppdb para pendaftar berasal dari beberapa provinsi, berikut data jumlah pendaftar<br>
        berdasarkan provinsi.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(fig)

    # Filter data untuk tahun 2020, 2021, dan 2022
    ppdb_filtered = ppdb[ppdb['tahun_diterima'].isin([20, 21, 22])]

    # Menghitung jumlah siswa per kecamatan
    ppdb_grouped = ppdb_filtered.groupby(['kecamatan', 'tahun_diterima'])['nama_pendaftar'].count().reset_index()
    ppdb_grouped.rename(columns={'nama_pendaftar':'jumlah_siswa'}, inplace=True)

    # Mengatur kombinasi warna
    color_map = {'2020': '#0077c2', '2021': '#b2301d', '2022': '#f6a800'}

    # Membuat histogram
    fig = px.histogram(ppdb_grouped, x='kecamatan', y='jumlah_siswa', color='tahun_diterima', barmode='group',
                    color_discrete_map=color_map)

    # Menambahkan judul dan label pada plot
    fig.update_layout(
        title={
            'text': "Jumlah Siswa PPDB Berdasarkan Kecamatan Asal per Tahunnya",
            'x':0.5,
            'y':0.95,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Kecamatan Asal",
        yaxis_title="Jumlah Siswa",
        font=dict(size=12)
    )

    # Menampilkan plot
    st.plotly_chart(fig)

    
    percent_saintek = {}
    percent_soshum = {}

# Group the data by pilihan (the chosen major)
    grouped = ppdb.groupby('pilihan')

# Loop through each group and calculate the percentage of SAINTEK and SOSHUM participants
    for pilihan, group in grouped.groups.items():
        kelompok = ppdb.loc[group]
        
        # Take the columns to calculate the number of saintek or soshum
        kolom_saintek = ['ipa1', 'ipa2', 'ipa3', 'ipa4', 'ipa5', 'matematika1', 'matematika2', 'matematika3', 'matematika4', 'matematika5']
        kolom_soshum = ['ips1', 'ips2', 'ips3', 'ips4', 'ips5', 'ppkn1', 'ppkn2', 'ppkn3', 'ppkn4', 'ppkn5', 'bing1', 'bing2', 'bing3', 'bing4', 'bing5', 'bindo1', 'bindo2', 'bindo3', 'bindo4', 'bindo5']
        
        # Count the number of participants who chose the SAINTEK or SOSHUM subjects in this group
        count_saintek = kelompok[kolom_saintek].sum().sum()
        count_soshum = kelompok[kolom_soshum].sum().sum()
        
        # Calculate the percentage of the number of participants who chose the SAINTEK or SOSHUM studies subject in this group
        total_peserta = len(kelompok)
        percent_saintek[pilihan] = round((count_saintek / total_peserta) * 100)
        percent_soshum[pilihan] = round((count_soshum / total_peserta) * 100)

    # Create a dataframe from the calculation results
    df_percent = pd.DataFrame({'Saintek (%)': percent_saintek, 'Soshum (%)': percent_soshum})

    # Highlight the highest value in each column
    highlighted = df_percent.style.highlight_max(axis=0)

    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'> Berikut presentase setiap jurusan yang cemderung fokus pada<br>
        SAINTEK dan SOSHUM.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    # Display the highlighted dataframe using Streamlit
    st.write(highlighted)
    
    #tools visualisasi, presresntasi deman antara soshum dan saintek
    ppdb['total_skor'] = ppdb.sum(axis=1)
    ppdb['kategori'] = 'SAINTEK'
    total_jurusan = ppdb.groupby('pilihan')['skor'].count()
    ppdb.loc[(ppdb[['matematika1', 'matematika2', 'matematika3', 'matematika4', 'matematika5', 'ipa1', 'ipa2', 'ipa3', 'ipa4', 'ipa5']].sum(axis=1) < ppdb[['bindo1', 'bindo2', 'bindo3', 'bindo4', 'bindo5', 'ips1', 'ips2', 'ips3', 'ips4', 'ips5']].sum(axis=1)) & (ppdb[['ppkn1', 'ppkn2', 'ppkn3', 'ppkn4', 'ppkn5', 'bing1', 'bing2', 'bing3', 'bing4', 'bing5']].sum(axis=1) > 0), 'kategori'] = 'SOSHUM'
    total_saintek = ppdb.loc[ppdb['kategori'] == 'SAINTEK', 'total_skor'].count()
    total_soshum = ppdb.loc[ppdb['kategori'] == 'SOSHUM', 'total_skor'].count()
    persentase_saintek = ppdb.loc[ppdb['kategori'] == 'SAINTEK'].groupby('pilihan')['total_skor'].count() / total_jurusan * 100
    persentase_soshum = ppdb.loc[ppdb['kategori'] == 'SOSHUM'].groupby('pilihan')['total_skor'].count() / total_jurusan * 100

    #kode menampilkan chart untuk pendafatar yang memilih mapel saintek atau soshum
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=total_jurusan.index,
        y=persentase_saintek,
        name='SAINTEK'
    ))
    fig.add_trace(go.Bar(
        x=total_jurusan.index,
        y=persentase_soshum,
        name='SOSHUM'
    ))
    fig.update_layout(
        title='Perbandingan Siswa yang menyukai Mapel SAINTEK dan SOSHUM per Jurusan',
        xaxis_title='Jurusan',
        yaxis_title='Persentase Siswa (%)',
        barmode='stack'
    )
    st.plotly_chart(fig)

    
    #kode untuk manampilkan chart jumlah distribusi skor di SMKN 1 Cibinong
    fig = px.histogram(ppdb, x='skor', color='tahun_diterima', nbins=12, opacity=0.6,
                   marginal='rug', barmode='overlay', color_discrete_sequence=['red', 'blue', 'green'],
                   labels={'skor': 'Score', 'count': 'Count'}, title='Histogram of Score Distribution SMKN 1 Cibinong')

    fig.update_layout(
        xaxis=dict(title='Score'),
        yaxis=dict(title='Count'),
        title=dict(text='Histogram of Student Score Distribution SMKN 1 Cibinong', font=dict(size=16), x=0.5),
        hovermode='x',
        legend=dict(title='Year Accepted', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    st.plotly_chart(fig)

    #Box plot digunakan untuk melihat distribusi skor dalam Data PPDB 2020 setiap mata pelajaran.
    mata_pelajaran = ['ppkn1', 'ppkn2', 'ppkn3', 'ppkn4', 'ppkn5', 
                    'bindo1', 'bindo2', 'bindo3', 'bindo4', 'bindo5', 
                    'matematika1', 'matematika2', 'matematika3', 'matematika4', 'matematika5', 
                    'ipa1', 'ipa2', 'ipa3', 'ipa4', 'ipa5', 
                    'ips1', 'ips2', 'ips3', 'ips4', 'ips5', 
                    'bing1', 'bing2', 'bing3', 'bing4', 'bing5']

    fig20 = go.Figure()
    fig20.update_layout(title='Distribution of scores Box Plot in PPDB 2020 Data', title_x=0.5)

    for pelajaran in mata_pelajaran:
        fig20.add_trace(go.Box(y=ppdb20[pelajaran], name=pelajaran))
        
    fig20.update_layout(xaxis_title='Subject', yaxis_title='Score')
    
    #Box plot digunakan untuk melihat distribusi skor dalam Data PPDB 2021 setiap mata pelajaran.
    mata_pelajaran = ['ppkn1', 'ppkn2', 'ppkn3', 'ppkn4', 'ppkn5', 
                    'bindo1', 'bindo2', 'bindo3', 'bindo4', 'bindo5', 
                    'matematika1', 'matematika2', 'matematika3', 'matematika4', 'matematika5', 
                    'ipa1', 'ipa2', 'ipa3', 'ipa4', 'ipa5', 
                    'ips1', 'ips2', 'ips3', 'ips4', 'ips5', 
                    'bing1', 'bing2', 'bing3', 'bing4', 'bing5']

    fig21 = go.Figure()
    fig21.update_layout(title='Distribution of scores Box Plot in PPDB 2021 Data', title_x=0.5)

    for pelajaran in mata_pelajaran:
        fig21.add_trace(go.Box(y=ppdb21[pelajaran], name=pelajaran))
        
    fig21.update_layout(xaxis_title='Subject', yaxis_title='Score')
    
    #Box plot digunakan untuk melihat distribusi skor dalam Data PPDB 2022 setiap mata pelajaran.
    mata_pelajaran = ['ppkn1', 'ppkn2', 'ppkn3', 'ppkn4', 'ppkn5', 
                    'bindo1', 'bindo2', 'bindo3', 'bindo4', 'bindo5', 
                    'matematika1', 'matematika2', 'matematika3', 'matematika4', 'matematika5', 
                    'ipa1', 'ipa2', 'ipa3', 'ipa4', 'ipa5', 
                    'ips1', 'ips2', 'ips3', 'ips4', 'ips5', 
                    'bing1', 'bing2', 'bing3', 'bing4', 'bing5']

    fig22 = go.Figure()
    fig22.update_layout(title='Distribution of scores Box Plot in PPDB 2022 Data', title_x=0.5)

    for pelajaran in mata_pelajaran:
        fig22.add_trace(go.Box(y=ppdb22[pelajaran], name=pelajaran))
        
    fig22.update_layout(xaxis_title='Subject', yaxis_title='Score')

    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'> Berikut jumlah distribusi skor di SMKN 1 Cibinong.</div>
        <br><br>
    """, unsafe_allow_html=True)
    #nampilin chart
    st.plotly_chart(fig20)
    st.plotly_chart(fig21)
    st.plotly_chart(fig22)
    
    #data yang sama dengan di atas namun dengan menghapus outliers korelasinya pada ppdb 2020
    # a list containing column names for each subject
    mata_pelajaran = ['ppkn1', 'ppkn2', 'ppkn3', 'ppkn4', 'ppkn5', 
                    'bindo1', 'bindo2', 'bindo3', 'bindo4', 'bindo5', 
                    'matematika1', 'matematika2', 'matematika3', 'matematika4', 'matematika5', 
                    'ipa1', 'ipa2', 'ipa3', 'ipa4', 'ipa5', 
                    'ips1', 'ips2', 'ips3', 'ips4', 'ips5', 
                    'bing1', 'bing2', 'bing3', 'bing4', 'bing5']

    # create figure and axes objects with subplots
    fig20 = go.Figure()
    fig20.update_layout(title='Distribution of scores Box Plot in PPDB 2020 Data', title_x=0.5)

    # add box plots for each subject
    for pelajaran in mata_pelajaran:
        # winsorize the data to remove outliers
        winsorized_data = winsorize(ppdb20[pelajaran], limits=[0.05, 0.05])
        
        # add box plot with winsorized data
        fig20.add_trace(go.Box(y=winsorized_data, name=pelajaran))

    # update x-axis and y-axis labels
    fig20.update_layout(xaxis_title='Subject', yaxis_title='Score')
    
    #data yang sama dengan di atas namun dengan menghapus outliers korelasinya pada ppdb 2020
    # a list containing column names for each subject
    mata_pelajaran = ['ppkn1', 'ppkn2', 'ppkn3', 'ppkn4', 'ppkn5', 
                    'bindo1', 'bindo2', 'bindo3', 'bindo4', 'bindo5', 
                    'matematika1', 'matematika2', 'matematika3', 'matematika4', 'matematika5', 
                    'ipa1', 'ipa2', 'ipa3', 'ipa4', 'ipa5', 
                    'ips1', 'ips2', 'ips3', 'ips4', 'ips5', 
                    'bing1', 'bing2', 'bing3', 'bing4', 'bing5']

    # create figure and axes objects with subplots
    fig21 = go.Figure()
    fig21.update_layout(title='Distribution of scores Box Plot in PPDB 2021 Data', title_x=0.5)

    # add box plots for each subject
    for pelajaran in mata_pelajaran:
        # winsorize the data to remove outliers
        winsorized_data = winsorize(ppdb21[pelajaran], limits=[0.05, 0.05])
        
        # add box plot with winsorized data
        fig21.add_trace(go.Box(y=winsorized_data, name=pelajaran))

    # update x-axis and y-axis labels
    fig21.update_layout(xaxis_title='Subject', yaxis_title='Score')
    
    #data yang sama dengan di atas namun dengan menghapus outliers korelasinya pada ppdb 2020
    # a list containing column names for each subject
    mata_pelajaran = ['ppkn1', 'ppkn2', 'ppkn3', 'ppkn4', 'ppkn5', 
                    'bindo1', 'bindo2', 'bindo3', 'bindo4', 'bindo5', 
                    'matematika1', 'matematika2', 'matematika3', 'matematika4', 'matematika5', 
                    'ipa1', 'ipa2', 'ipa3', 'ipa4', 'ipa5', 
                    'ips1', 'ips2', 'ips3', 'ips4', 'ips5', 
                    'bing1', 'bing2', 'bing3', 'bing4', 'bing5']

    # create figure and axes objects with subplots
    fig22 = go.Figure()
    fig22.update_layout(title='Distribution of scores Box Plot in PPDB 2022 Data', title_x=0.5)

    # add box plots for each subject
    for pelajaran in mata_pelajaran:
        # winsorize the data to remove outliers
        winsorized_data = winsorize(ppdb22[pelajaran], limits=[0.05, 0.05])
        
        # add box plot with winsorized data
        fig22.add_trace(go.Box(y=winsorized_data, name=pelajaran))

    # update x-axis and y-axis labels
    fig22.update_layout(xaxis_title='Subject', yaxis_title='Score')

    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'> Berikut jumlah distribusi dengan outliers korelasi antara skore<br>
        dan mapel di SMKN 1 Cibinong.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(fig20)
    st.plotly_chart(fig21)
    st.plotly_chart(fig22)

    #Heatmap digunakan untuk melihat korelasi antara skor dalam PPDB 2020 pada setiap mata pelajaran.
    data = go.Heatmap(
                z=ppdb20[mata_pelajaran].corr().values,
                x=mata_pelajaran,
                y=mata_pelajaran,
                colorscale='Viridis'
            )

    layout = go.Layout(
                title=dict(text="Korelasi antara Skor pada Setiap Mata Pelajaran", x=0.5, font=dict(size=15, color='white', family='Arial')),
                xaxis=dict(title=dict(text="Mata Pelajaran", font=dict(size=18))),
                yaxis=dict(title=dict(text="Mata Pelajaran", font=dict(size=18))),
            )

    fig20 = go.Figure(data=data, layout=layout)

    #Heatmap digunakan untuk melihat korelasi antara skor dalam PPDB 2021 pada setiap mata pelajaran.
    data = go.Heatmap(
                z=ppdb21[mata_pelajaran].corr().values,
                x=mata_pelajaran,
                y=mata_pelajaran,
                colorscale='Viridis'
            )

    layout = go.Layout(
                title=dict(text="Korelasi antara Skor pada Setiap Mata Pelajaran", x=0.5, font=dict(size=15, color='white', family='Arial')),
                xaxis=dict(title=dict(text="Mata Pelajaran", font=dict(size=18))),
                yaxis=dict(title=dict(text="Mata Pelajaran", font=dict(size=18))),
            )

    fig21 = go.Figure(data=data, layout=layout)
    
    #Heatmap digunakan untuk melihat korelasi antara skor dalam PPDB 2022 pada setiap mata pelajaran.
    data = go.Heatmap(
                z=ppdb22[mata_pelajaran].corr().values,
                x=mata_pelajaran,
                y=mata_pelajaran,
                colorscale='Viridis'
            )

    layout = go.Layout(
                title=dict(text="Korelasi antara Skor pada Setiap Mata Pelajaran", x=0.5, font=dict(size=15, color='white', family='Arial')),
                xaxis=dict(title=dict(text="Mata Pelajaran", font=dict(size=18))),
                yaxis=dict(title=dict(text="Mata Pelajaran", font=dict(size=18))),
            )

    fig22 = go.Figure(data=data, layout=layout)
    
    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'> Berikut korelasi di setiap mata pelajaran.</div>
        <br><br>
    """, unsafe_allow_html=True)
    
    #menampilkan chart dari ppdb 2020-2022
    st.plotly_chart(fig20)
    st.plotly_chart(fig21)
    st.plotly_chart(fig22)
    
    #menampilkan chart 10 besar murid sekolah(SMP) yang mendaftar di SMKN 1 Cibinong
    school_count = ppdb.groupby('tahun_diterima').asal_sekolah.value_counts()
    school_count_2020 = school_count[20][:10]
    school_count_2021 = school_count[21][:10]
    school_count_2022 = school_count[22][:10]

    # Define colors
    colors_2020 = sns.color_palette('husl', len(school_count_2020)).as_hex()
    colors_2021 = sns.color_palette('bright', len(school_count_2021)).as_hex()
    colors_2022 = sns.color_palette('husl', len(school_count_2022)).as_hex() 

    # Create the subplots
    fig = make_subplots(rows=3, cols=1, subplot_titles=("2020", "2021", "2022"), vertical_spacing=0.2)

    # Create the bar plots
    fig.add_trace(go.Bar(x=school_count_2020.index, y=school_count_2020, marker_color=colors_2020),
                row=1, col=1)
    fig.add_trace(go.Bar(x=school_count_2021.index, y=school_count_2021, marker_color=colors_2021),
                row=2, col=1)
    fig.add_trace(go.Bar(x=school_count_2022.index, y=school_count_2022, marker_color=colors_2022),
                row=3, col=1)

    # Set the title and axis labels
    fig.update_layout(
        title="Top 10 Junior High School by Number of Registrants",
        title_font_size=20,
        xaxis=dict(title="Junior High School", title_font_size=20),
        yaxis=dict(title="Number of Registrants", title_font_size=20, title_standoff=20),
        margin=dict(t=150) # Menambahkan margin 100 pada bagian atas
    )


    # Set the font size of the subplot titles and axis labels
    fig.update_annotations(font_size=20)

    # Set the size of the figure
    fig.update_layout(height=1200, width=1000)

    #display chart
    st.plotly_chart(fig)
    
    mvp = {"school":[],"score":[]}
    for i in range(ppdb.shape[0]):
        if ppdb.iloc[i,:]['skor'] > ppdb['skor'].mean():
            mvp["school"].append(ppdb.iloc[i,:]['asal_sekolah'])
            mvp["score"].append(ppdb.iloc[i,:]['skor'])
            
    above_avg = pd.DataFrame(mvp)
    top_5_school = above_avg.school.value_counts()[:10]

    fig = px.bar(x=top_5_school.index, y=top_5_school, color=top_5_school.index, color_discrete_sequence=px.colors.sequential.Magma)
    fig.update_layout(title="Top 10 Above Average Registrant's Junior High School Score in 2020-2022",
                    xaxis_title="Junior High School",
                    yaxis_title="Quantity")
    
    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'> Berikut top 10 sekolah dengan nilai pendaftar tertinggi.</div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(fig)

    #kompetensi skill paling laku
    comp_by_score = pd.DataFrame(ppdb.groupby(['pilihan','tahun_diterima']).agg(['mean','count']).skor)
    comp_avg_2020 = comp_by_score.xs(20, level="tahun_diterima")
    comp_avg_2021 = comp_by_score.xs(21, level="tahun_diterima")
    comp_avg_2022 = comp_by_score.xs(22, level="tahun_diterima")

    # display results
    st.write("2020 Average Scores by Choice:")
    st.write(comp_avg_2020)

    st.write("2021 Average Scores by Choice:")
    st.write(comp_avg_2021)

    st.write("2022 Average Scores by Choice:")
    st.write(comp_avg_2022)
    
    # Group data by skill competencies and year of admission, and calculate mean and count
    comp_by_score = ppdb.groupby(['pilihan','tahun_diterima']).agg(['mean','count'])['skor'].reset_index()

    # Separate data by year of admission
    comp_avg_2020 = comp_by_score[comp_by_score['tahun_diterima'] == 20]
    comp_avg_2021 = comp_by_score[comp_by_score['tahun_diterima'] == 21]
    comp_avg_2022 = comp_by_score[comp_by_score['tahun_diterima'] == 22]

    # Create fig and subplots
    fig = make_subplots(rows=3, cols=1, subplot_titles=("Number of Registrants by Skill Competencies in 2020",
                                                        "Number of Registrants by Skill Competencies in 2021",
                                                        "Number of Registrants by Skill Competencies in 2022"))

    # Add trace for each subplot
    fig.add_trace(
        go.Bar(x=comp_avg_2020['pilihan'], y=comp_avg_2020['count'], name="2020"),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=comp_avg_2021['pilihan'], y=comp_avg_2021['count'], name="2021"),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(x=comp_avg_2022['pilihan'], y=comp_avg_2022['count'], name="2022"),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        height=1500, 
        width=1000, 
        margin=dict(t=150) # memberi jarak 100 pixel pada bagian atas judul
    )


    fig.update_xaxes(title_text="Skill Competencies", row=3, col=1)
    fig.update_yaxes(title_text="Number of Registrants", row=2, col=1)

    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'> Jumlah pendaftar di setiap jurusan.</div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(fig)

        # Calculate mean score by skill competencies and year
    comp_by_score = pd.DataFrame(ppdb.groupby(['pilihan', 'tahun_diterima']).agg(['mean'])['skor'])
    comp_by_score.columns = ['mean_score']
    comp_by_score.reset_index(inplace=True)

    # Filter by year
    comp_avg_2020 = comp_by_score[comp_by_score['tahun_diterima'] == 20]
    comp_avg_2021 = comp_by_score[comp_by_score['tahun_diterima'] == 21]
    comp_avg_2022 = comp_by_score[comp_by_score['tahun_diterima'] == 22]

    # Create subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Average Score by Skill Competencies in 2020",
                                                        "Average Score by Skill Competencies in 2021",
                                                        "Average Score by Skill Competencies in 2022",
                                                        ""))

    # Add traces to subplots
    fig.add_trace(go.Bar(x=comp_avg_2020['pilihan'], y=comp_avg_2020['mean_score'], name='2020'), row=1, col=1)
    fig.add_trace(go.Bar(x=comp_avg_2021['pilihan'], y=comp_avg_2021['mean_score'], name='2021'), row=1, col=2)
    fig.add_trace(go.Bar(x=comp_avg_2022['pilihan'], y=comp_avg_2022['mean_score'], name='2022'), row=2, col=1)

    # Update layout
    fig.update_layout(height=1200, width=1000, title_text="Average Score by Skill Competencies in 2020-2022")
    fig.update_xaxes(title_text="Skill Competencies", row=1, col=1)
    fig.update_xaxes(title_text="Skill Competencies", row=1, col=2)
    fig.update_xaxes(title_text="Skill Competencies", row=2, col=1)
    fig.update_xaxes(title_text="Skill Competencies", row=2, col=2)
    fig.update_yaxes(title_text="Average Score", row=1, col=1, range=[2800, 3050])
    fig.update_yaxes(title_text="Average Score", row=1, col=2, range=[2800, 3050])
    fig.update_yaxes(title_text="Average Score", row=2, col=1, range=[2800, 3050])
    fig.update_yaxes(title_text="Average Score", row=2, col=2, range=[2800, 3050])

    #teks keterangan
    st.markdown("""
        <div class="des-title" style='text-align: center;'>Kompetensi skill rata-rata dengan skor rata-rata tertinggi.</div>
    """, unsafe_allow_html=True)
    
    # Show plot
    st.plotly_chart(fig)
    

