import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import joblib

data = pd.read_csv('recorded_video/r_output_gait_parameters_per_frame.csv')

def r_train_model():
    X = data[['height','shoulder_width','hip_width','left_knee_angle','right_knee_angle','left_elbow_angle','right_elbow_angle','height_width_ratio','shoulder_to_hip_ratio']]
    y = data['person_name']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, callbacks=[early_stopping])

    model.save('r_gait_model.h5')
    print("Model saved as 'r_gait_model.h5'")
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Label encoder saved as 'label_encoder.pkl'")
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
