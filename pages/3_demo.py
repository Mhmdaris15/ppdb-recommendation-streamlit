import pickle
import streamlit as st
import numpy as np
from prettytable import PrettyTable


with open("model.pkl", "rb") as f:
    model=pickle.load(f)
    
with open("predict_chances.pkl", "rb") as a:
    test=pickle.load(a)
    
with st.container():

    # display accuracy for each competition
    # st.write('Model Accuracies')
    # for model['comp_name'], model['accuracy'] in accuracies.items():
    #     st.write("Accuracy for {}: {:.2f}".format(model['comp_name'], model['accuracy']))
    # test= dir(model)
    # st.write(test)
    # st.write('Enter values for prediction')
    # feature1 = st.number_input('Feature 1')

    # make a prediction based on the input values
    # prediction = model.predict(np.array([feature1]).reshape(1, -1))[0]
    # test.predict(123)
    
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
        
    model.comp_name
        
    # Get hyperparameter values
    learning_rate = model.get_params()['learning_rate']
    n_estimators = model.get_params()['n_estimators']

    # Use hyperparameter values in Streamlit app
    st.write("Learning rate: ", learning_rate)
    st.write("Number of estimators: ", n_estimators)
    
    accuracies = {}
    models= {}
    for comp_name, comp_model in models.items():
        y_pred = comp_model.predict(X_test)
        accuracies[comp_name] = accuracy_score(y_test, y_pred)

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

    # # Assign model variables to new variables
    # model_variable_1 = model.variable_1
    # model_variable_2 = model.variable_2

    # # Use model variables in Streamlit app
    # st.write("Model variable 1: ", model_variable_1)
    # st.write("Model variable 2: ", model_variable_2)

    # display the predicted class
    # st.write('The predicted class is', prediction)
    # st.write(model)