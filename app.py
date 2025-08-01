import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the model
try:
    model = joblib.load('machine_failure_model.pkl')
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Define the function to make predictions
def predict_failure(air_temp, process_temp, rot_speed, torque, tool_wear, type_l, type_m):
    # Create a scaler and scale the input data
    input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear, type_l, type_m]])
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction[0]

def main():
    st.title("Machine Failure Prediction")

    # Input features
    air_temp = st.number_input("Enter the air temperature [K]", min_value=0.0)
    process_temp = st.number_input("Enter the process temperature [K]", min_value=0.0)
    rot_speed = st.number_input("Enter the rotational speed [rpm]", min_value=0.0)
    torque = st.number_input("Enter the torque [Nm]", min_value=0.0)
    tool_wear = st.number_input("Enter the tool wear [min]", min_value=0.0)
    type_l = st.selectbox("Select Type L", [0, 1])
    type_m = st.selectbox("Select Type M", [0, 1])

    # Predict button
    if st.button("Predict"):
        if 'model' not in globals():
            st.error("Model not loaded correctly. Please check the model file.")
        else:
            result = predict_failure(air_temp, process_temp, rot_speed, torque, tool_wear, type_l, type_m)
            if result == 0:
                st.success("The machine is predicted to be in good condition.")
            else:
                st.error("The machine is at risk of failure.")

if __name__ == '__main__':
    main()