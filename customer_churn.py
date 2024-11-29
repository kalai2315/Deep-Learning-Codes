import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import HeNormal
from keras.optimizers import Adam
import pickle

# Title of the Streamlit app
st.title("Customer Churn Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Feature and target selection
    try:
        x = df[['MonthlyCharges', 'tenure', 'TotalCharges']]
        y = df['Churn_Yes']
        
        # Handle missing values
        data = pd.concat([x, y], axis=1)
        data = data.dropna()
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Build the model
        model = Sequential()
        model.add(Dense(12, input_dim=3, activation='relu', kernel_initializer=HeNormal()))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Train the model
        st.write("Training the model...")
        history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=0)

        # Save the model using pickle
        with open('churn_model.pkl', 'wb') as file:
            pickle.dump(model, file)
        st.success("Model trained and saved successfully!")

        # Evaluate the model
        st.write("Model Accuracy:")
        _, accuracy = model.evaluate(x_test, y_test, verbose=0)
        st.write(f"Accuracy: {accuracy * 100:.2f}%")

        # Confusion Matrix
        y_pred = (model.predict(x_test) > 0.5).astype(int)
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        # Allow the user to make predictions
        st.subheader("Make a New Prediction")
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=0.01)
        tenure = st.number_input("Tenure (Months)", min_value=0, step=1)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, step=0.01)

        if st.button("Predict Churn"):
            # Scale the input data
            new_data = np.array([[MonthlyCharges, tenure, TotalCharges]])
            new_data_scaled = scaler.transform(new_data)

            # Load the trained model
            with open('churn_model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)

            # Make the prediction
            prediction = loaded_model.predict(new_data_scaled)
            prediction_label = "Churn" if prediction > 0.5 else "No Churn"
            st.write(f"Prediction: {prediction_label} (Probability: {prediction[0][0]:.2f})")

    except KeyError as e:
        st.error(f"KeyError: {e}. Ensure the dataset contains the correct columns.")
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload a dataset to proceed.")
