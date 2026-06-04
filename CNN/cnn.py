import streamlit as st
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="CNN – MNIST Digit Classifier", layout="wide")

# -----------------------------
# Freeze randomness
# -----------------------------
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# -----------------------------
# Load & Cache Data
# -----------------------------
@st.cache_data
def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test  = X_test.reshape(-1, 28, 28, 1)  / 255.0
    return X_train, y_train, X_test, y_test

# -----------------------------
# Build, Train & Cache Model
# -----------------------------
@st.cache_resource
def build_and_train_model(X_train, y_train, epochs=10, train_size=None):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),          # prevents overfitting
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    if train_size is not None:
        X_train_use = X_train[:train_size]
        y_train_use = y_train[:train_size]
    else:
        X_train_use = X_train
        y_train_use = y_train

    history = model.fit(
        X_train_use, y_train_use,
        epochs=epochs,
        batch_size=64,
        validation_split=0.1,
        verbose=0
    )

    return model, history

# -----------------------------
# Load data & train model
# -----------------------------
X_train, y_train, X_test, y_test = load_data()

# ---- Sidebar: Mode selection (simple fast demo vs full) ----
mode = st.sidebar.selectbox("Mode", ["Simple (fast demo)", "Full (train)"])

if mode == "Simple (fast demo)":
    demo_epochs = 1
    demo_train_size = 2000
    spinner_text = "Training small demo CNN (fast, cached)"
else:
    demo_epochs = 10
    demo_train_size = None
    spinner_text = "Training CNN model... (only once, cached after)"

with st.spinner(spinner_text):
    model, history = build_and_train_model(X_train, y_train, epochs=demo_epochs, train_size=demo_train_size)

# Predictions
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
acc = accuracy_score(y_test, y_pred)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Vanilla CNN — MNIST Digit Classifier")
st.markdown("Built with **TensorFlow / Keras** | Dataset: MNIST (60,000 training images)")

if mode == "Full (train)":
    st.divider()

    # ---- Row 1: Accuracy metric ----
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Test Accuracy", f"{acc * 100:.2f}%")
    col2.metric("📦 Training Samples", "54,000")
    col3.metric("🔍 Test Samples", "10,000")

    st.divider()

    # ---- Row 2: Training History ----
    st.subheader("📈 Training History")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(history.history['accuracy'],    label='Train Accuracy', color='steelblue')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy',   color='darkorange')
        ax1.set_title("Model Accuracy over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(history.history['loss'],    label='Train Loss', color='steelblue')
        ax2.plot(history.history['val_loss'], label='Val Loss',  color='darkorange')
        ax2.set_title("Model Loss over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        st.pyplot(fig2)

    st.divider()

    # ---- Row 3: Confusion Matrix ----
    st.subheader("🔢 Confusion Matrix")

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=range(10), yticklabels=range(10), ax=ax3
    )
    ax3.set_title("Confusion Matrix — Test Set")
    ax3.set_xlabel("Predicted Label")
    ax3.set_ylabel("True Label")
    st.pyplot(fig3)

    st.divider()

    # ---- Row 4: Live Prediction ----
    st.subheader("🎯 Live Prediction")

    index = st.slider("Select a Test Image", 0, len(X_test) - 1, 0)

    col1, col2 = st.columns([1, 2])

    with col1:
        image = X_test[index].reshape(28, 28)
        true_label = y_test[index]

        fig4, ax4 = plt.subplots()
        ax4.imshow(image, cmap='gray')
        ax4.set_title(f"True Label: {true_label}", fontsize=14)
        ax4.axis('off')
        st.pyplot(fig4)

    with col2:
        if st.button("🔍 Predict Digit"):
            prediction = model.predict(X_test[index].reshape(1, 28, 28, 1), verbose=0)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.success(f"**Predicted Digit: {predicted_label}**")
            st.info(f"Confidence: {confidence:.2f}%")

            # Show all class probabilities
            fig5, ax5 = plt.subplots(figsize=(8, 3))
            ax5.bar(range(10), prediction[0], color='steelblue')
            ax5.set_xticks(range(10))
            ax5.set_xlabel("Digit Class")
            ax5.set_ylabel("Probability")
            ax5.set_title("Prediction Probabilities for Each Class")
            st.pyplot(fig5)

else:
    # Simple fast UI: show just accuracy and live predict
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("✅ Test Accuracy", f"{acc * 100:.2f}%")

    st.divider()
    st.subheader("🎯 Quick Prediction (Simple Mode)")
    index = st.slider("Select a Test Image", 0, len(X_test) - 1, 0)

    col1, col2 = st.columns([1, 2])
    with col1:
        image = X_test[index].reshape(28, 28)
        true_label = y_test[index]

        fig4, ax4 = plt.subplots()
        ax4.imshow(image, cmap='gray')
        ax4.set_title(f"True Label: {true_label}", fontsize=14)
        ax4.axis('off')
        st.pyplot(fig4)

    with col2:
        if st.button("🔍 Predict Digit"):
            prediction = model.predict(X_test[index].reshape(1, 28, 28, 1), verbose=0)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.success(f"**Predicted Digit: {predicted_label}**")
            st.info(f"Confidence: {confidence:.2f}%")