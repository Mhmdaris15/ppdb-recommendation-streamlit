import pickle
import streamlit as st
import numpy as np

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
    test.predict(123)

    # display the predicted class
    # st.write('The predicted class is', prediction)
    # st.write(model)