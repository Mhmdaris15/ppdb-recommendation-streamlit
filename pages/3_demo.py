import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import optuna
import re
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

st.set_page_config(layout="wide")

# model_xgb = xgb.Booster()
# model_xgb.load_model("./models/model.json")

# with open("./models/model.json", "r") as f:
#     model_xgb = xgb.Booster()
#     model_xgb.load_model(f)
    
with st.container():
    ppdb21= pd.read_csv("./ppdb_2021.csv")
    ppdb20= pd.read_csv("./ppdb_2020.csv")
    ppdb22= pd.read_csv("./ppdb_2022.csv")
     #hapus Unnamed
    ppdb22 = ppdb22.loc[:, ~ppdb22.columns.str.contains('^Unnamed')]
    ppdb22 = ppdb22.rename(columns={'Jarak ': 'Jarak', 'Pilihan ': 'Pilihan'})
    
    ppdb20 = ppdb20.rename(columns=lambda x: x.lower().replace(' ', '_'))
    ppdb21 = ppdb21.rename(columns=lambda x: x.lower().replace(' ', '_'))
    ppdb22 = ppdb22.rename(columns=lambda x: x.lower().replace(' ', '_'))
    
    #samain kolom
    print(set(ppdb22 == ppdb21.columns))
    print(set(ppdb22 == ppdb20.columns))
    
    def clean_pilihan_2021(pilihan):
        pilihan = pilihan.split(' - ')[1]
        pilihan = pilihan.split(' ')
        try: 
            pilihan.remove('DAN')
        except: 
            pass
        pilihan = ''.join([sen[0] if sen[0] != 'M' else 'MM' for sen in pilihan])
        pilihan = 'TFLM' if pilihan == "TFLMM" else pilihan
        return pilihan

    def clean_pilihan_2020(pilihan):
        pilihan = pilihan.split(' - ')[1]
        pilihan = pilihan.split(' ')
        try: 
            pilihan.remove('DAN')
        except: 
            pass
        pilihan = ''.join([sen[0] if sen[0] != 'M' else 'MM' for sen in pilihan])
        pilihan = 'TFLM' if pilihan == "TFLMM" else pilihan
        return pilihan

    def clean_pilihan_2022(pilihan):
        pilihan = pilihan.strip()
        pilihan = pilihan.split(' ')
        try:
            pilihan.remove('DAN')
        except:
            pass
        pilihan = ''.join([sen[0] for sen in pilihan])
        pilihan = 'TOI' if pilihan == 'TE' else pilihan
        return pilihan

    #format penamaan buat rename
    rename_cols = lambda col_name: "_".join(re.split("/| ", col_name.lower())) if len(re.split("/| ", col_name.lower())) > 0 else col_name.lower()

    
    #merubah data dll
    for year in range(20, 23):
        exec(f"ppdb{year}.insert(3, 'tahun_diterima', {year})") # insert year column to each dataset
        exec(f"ppdb{year}.rename(rename_cols, axis='columns', inplace=True)")
        exec(f"ppdb{year}.loc[:, 'agama1':'skor'] = ppdb{year}.loc[:, 'agama1':'skor'].astype(float)") # convert data type
        exec(f"ppdb{year}['tanggal_lahir'] = pd.to_datetime(ppdb{year}['tanggal_lahir'])") # convert to date time type
        exec(f"ppdb{year}['pilihan'] = ppdb{year}.pilihan.apply(clean_pilihan_20{year})")


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
#     # Define the rank generator function
#     def rank_generator(rank_arr):
#         quantile = np.percentile(rank_arr, np.arange(25, 100, 25)) # distinguish by quantile
#         for rank in rank_arr:
#             if rank <= quantile[0]:
#                 category = "High Chance"
#             elif rank <= quantile[1]:
#                 category = "Medium Chance"
#             elif rank <= quantile[2]:
#                 category = "Low Chance"
#             else:
#                 category = "Very Low Chance"
#             yield category

#     # Define the objective function for hyperparameter tuning
#     def objective(trial, X_train, X_test, y_train, y_test):
#         model = build_model(trial)
#         model.fit(X_train, y_train)
#         return model.score(X_test, y_test)

#     # Define the function to predict chances and display results
#     def predict_chances(score, models):
#         chances = PrettyTable()
#         chances.field_names = ['Skill Comp.', 'Chance']

#         for model_name, model in models.items():
#             chance_num = model.predict(np.array(score).reshape(1, -1))[0]
#             chance_str = list(rank_generator(model.classes_))[chance_num]
#             chances.add_row([model_name, chance_str])
        
#         st.table(chances)

#     # Load the data and create the ranked competition list
#     # ppdb = pd.read_csv("ppdb.csv")
#     ranked_comp = []

#     for comp_name in ppdb.pilihan.unique():
#         df = ppdb[ppdb.pilihan == comp_name][['nama_pendaftar', 'pilihan', 'skor']]
#         df.sort_values('skor', ascending=False, inplace=True)
#         df.reset_index(drop=True, inplace=True)
#         df.insert(0, 'rank', df.index.values + 1)
#         df.insert(1, 'rank_category', list(rank_generator(df['rank'])))
#         ranked_comp.append(df)

#     # Train XGBoost models for each competition
#     models = {}
#     accuracies = {}

#     for i in ranked_comp:
#         comp_name = i.pilihan.unique()[0].lower()
#         x = i[['skor']]
#         y = i['rank_category']
#         t_size = 0.2 if comp_name != "tflm" else 0.4

#         le = LabelEncoder()
#         y = le.fit_transform(y)  # transform string labels to numerical labels

#         X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=13, stratify=y)

#         study = optuna.create_study(direction='maximize')
#         study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=100)

#         best_params = study.best_params
#         st.write(f'Best hyperparameters for {comp_name}: {best_params}')

#         model = XGBClassifier(**best_params)
#         model.fit(X_train, y_train)

#         accuracy = model.score(X_test, y_test)
#         accuracies[comp_name] = accuracy # save model accuracy to the dictionary

#         models[comp_name] = model

#         st.write("Accuracy for {}: {:.2f}".format(comp_name, accuracy))

#     # Create the Streamlit app
#     st.title("Predict Chances for Skill Competitions")
#     score = st

    def rank_generator(rank_arr):
        quantile = np.percentile(rank_arr, np.arange(25,100,25)) # distinguish by quantile
        for rank in rank_arr:
            if rank <= quantile[0]:
                category = "High Chance"
            elif rank <= quantile[1]:
                category = "Medium Chance"
            elif rank <= quantile[2]:
                category = "Low Chance"
            else:
                category = "Very Low Chance"
            yield category

    # Load data
    # ppdb = pd.read_csv('data_ppdb.csv')

    # Create ranked companies
    ranked_comp = []
    for comp_name in ppdb.pilihan.unique():
        locals()[f"{comp_name.lower()}_rank"] = pd.DataFrame(ppdb[ppdb.pilihan == comp_name][['nama_pendaftar','pilihan','skor']])
        locals()[f"{comp_name.lower()}_rank"].sort_values('skor', ascending=False,inplace=True)
        locals()[f"{comp_name.lower()}_rank"].reset_index(drop=True,inplace=True)
        locals()[f"{comp_name.lower()}_rank"].insert(0,'rank',locals()[f"{comp_name.lower()}_rank"].index.values + 1)
        locals()[f"{comp_name.lower()}_rank"].insert(1, 'rank_category', list(rank_generator(locals()[f"{comp_name.lower()}_rank"]['rank'])))
        ranked_comp.append(locals()[f"{comp_name.lower()}_rank"])

    # Create streamlit app
    st.title('PPDB Rank Classifier')

    # Display table for each ranked company
    for i, comp in enumerate(ranked_comp):
        st.write(f"## {ppdb.pilihan.unique()[i]}")
        st.write(comp)
        # Define the layout of the app
    # st.set_page_config(page_title='XGBoost Hyperparameter Tuning', page_icon=':chart_with_upwards_trend:')

    # Define helper functions
    def build_model(trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.9, 0.1),
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.9, 0.1),
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y_train))
        }
        
        model = XGBClassifier(**params)
        
        return model


    def objective(trial):
        model = build_model(trial)
        model.fit(X_train, y_train)
        return model.score(X_test, y_test)

    # Define the main section of the app
        # Create dictionaries to store accuracies and models
    accuracies = {}
    models = {}

    # Create PrettyTable object to display results
    model_acc = PrettyTable()
    model_acc.field_names = ["Competition", "Accuracy"]

    best_params_models = {
        'rpl': {'learning_rate': 0.00016186094336899463, 'max_depth': 4, 'n_estimators': 900, 'min_child_weight': 10, 'subsample': 0.8, 'colsample_bytree': 0.7},
        'toi': {'learning_rate': 0.011210608133115814, 'max_depth': 10, 'n_estimators': 850, 'min_child_weight': 2, 'subsample': 0.7, 'colsample_bytree': 0.6},
        'mm': {'learning_rate': 0.0014625676483288616, 'max_depth': 9, 'n_estimators': 150, 'min_child_weight': 2, 'subsample': 0.7, 'colsample_bytree': 0.8},
        'tkj': {'learning_rate': 0.00010704587052700323, 'max_depth': 10, 'n_estimators': 350, 'min_child_weight': 5, 'subsample': 0.6, 'colsample_bytree': 0.8},
        'sija': {'learning_rate': 0.037862663956716905, 'max_depth': 4, 'n_estimators': 1000, 'min_child_weight': 2, 'subsample': 0.7, 'colsample_bytree': 0.9},
        'tflm': {'learning_rate': 0.02865093595842963, 'max_depth': 2, 'n_estimators': 650, 'min_child_weight': 2, 'subsample': 0.9, 'colsample_bytree': 0.8},
        'dpib': {'learning_rate': 0.03717597979731156, 'max_depth': 9, 'n_estimators': 50, 'min_child_weight': 2, 'subsample': 0.6, 'colsample_bytree': 0.7},
        'tp': {'learning_rate': 0.02600101163772328, 'max_depth': 2, 'n_estimators': 200, 'min_child_weight': 2, 'subsample': 0.6, 'colsample_bytree': 0.9},
        'tkro': {'learning_rate': 0.002009698797092612, 'max_depth': 8, 'n_estimators': 800, 'min_child_weight': 3, 'subsample': 0.9, 'colsample_bytree': 0.9},
        'bkp': {'learning_rate': 0.0594738621683631, 'max_depth': 8, 'n_estimators': 450, 'min_child_weight': 2, 'subsample': 0.9, 'colsample_bytree': 0.9}
    }

    # Loop through each competition
    for i in ranked_comp:
        comp_name = i.pilihan.unique()[0].lower()
        x = i[['skor']]
        y = i['rank_category']
        t_size = 0.4 if i.pilihan.unique()[0] != "TFLM" else 0.4

        le = LabelEncoder()
        y = le.fit_transform(y)  # transform string labels to numerical labels

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=t_size, random_state=13, stratify=y)

        # study = optuna.create_study(direction='maximize')
        # study.optimize(objective, n_trials=100)

        best_params = best_params_models[f'{comp_name}']
        st.write(f'Best hyperparameters for {comp_name}: {best_params}')

        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        accuracies[comp_name] = accuracy # save model accuracy to the dictionary

        model_acc.add_row([comp_name, accuracy])

        models[comp_name] = model

    # Display results
    st.header("XGBoost Hyperparameter Tuning Results")
    # st.write(model_acc)

# import pickle
# import streamlit as st
# import numpy as np
# import pandas as pd
# from prettytable import PrettyTable


# with open("model.pkl", "rb") as f:
#     model=pickle.load(f)
    
# with open("ppdb.pkl", "rb") as a:
#     ppdb=pickle.load(a)
    
# with st.container():
#     # test.predict(123)
    
#     with open('model.pkl', 'rb') as file:
#         model = pickle.load(file)
    
#     # st.write(type(model))
#     booster = model.get_booster()
#     st.write(booster)
#         # Create rank classifier generator
#     def rank_generator(rank_arr):
#         quantile = np.percentile(rank_arr, np.arange(25,100,25)) # distinguish by quantile
#         for rank in rank_arr:
#             if rank <= quantile[0]:
#                 category = "High Chance"
#             elif rank <= quantile[1]:
#                 category = "Medium Chance"
#             elif rank <= quantile[2]:
#                 category = "Low Chance"
#             else:
#                 category = "Very Low Chance"
#             yield category
            
#     # Create ranked companies
#     ranked_comp = []
#     for comp_name in ppdb.pilihan.unique():
#         locals()[f"{comp_name.lower()}_rank"] = pd.DataFrame(ppdb[ppdb.pilihan == comp_name][['nama_pendaftar','pilihan','skor']])
#         locals()[f"{comp_name.lower()}_rank"].sort_values('skor', ascending=False,inplace=True)
#         locals()[f"{comp_name.lower()}_rank"].reset_index(drop=True,inplace=True)
#         locals()[f"{comp_name.lower()}_rank"].insert(0,'rank',locals()[f"{comp_name.lower()}_rank"].index.values + 1)
#         locals()[f"{comp_name.lower()}_rank"].insert(1, 'rank_category', list(rank_generator(locals()[f"{comp_name.lower()}_rank"]['rank'])))
#         ranked_comp.append(locals()[f"{comp_name.lower()}_rank"])


#     for i in ranked_comp:
#         comp_name = i.pilihan.unique()[0].lower()
#         x = i[['skor']]
#         y = i['rank_category']
#         t_size = 0.4 if i.pilihan.unique()[0] != "TFLM" else 0.4
    
        
#     # Get hyperparameter values
#     learning_rate = model.get_params()['learning_rate']
#     n_estimators = model.get_params()['n_estimators']

#     # Use hyperparameter values in Streamlit app
#     st.write("Learning rate: ", learning_rate)
#     st.write("Number of estimators: ", n_estimators)
    
    # accuracies = {}
    # models= {}
    # booster = model.get_booster()
    # feature_importance = booster.get_score(importance_type='gain')
    # st.write(booster)
    # for comp_name, comp_model in models.items():
    #     y_pred = comp_model.predict(X_test)
    #     accuracies[comp_name] = accuracy_score(y_test, y_pred)

    # Display accuracies in Streamlit
    for comp_name, accuracy in accuracies.items():
        st.write("Accuracy for {}: {:.2f}".format(comp_name, accuracy))
            
    # Define a function to make predictions
    def predict_chances(score):
        chances = PrettyTable()
        chances.field_names = ['Skill Comp.', 'Chance']

        for model_name, model in models.items():
            chance_num = model.predict(np.array(score).reshape(1, -1))[0] # predict the chance of getting the competition
            chance_str = list(rank_generator(model.classes_))[chance_num] # convert numeric label to string category
            chances.add_row([model_name, chance_str]) # add the result to the table
        st.write(chances)

    # Define an input form for users to enter score data
    st.write("Enter your competition score below:")
    
    # Combine the features into a single numpy array
    score = st.number_input("Skill Competition 1")

    # Make a prediction and display the result
    prediction = predict_chances(score)
    st.write("Based on your competition scores, you have a chance of: ", prediction)


    