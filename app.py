import streamlit as st
import joblib
import time
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("spamsentry.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Sample Metrics (Replace with actual computed values)
metrics = {
    "Accuracy": 95,
    "Precision": 94,
    "Recall": 93,
    "F1-Score": 93.5
}

# Custom Page Configuration
st.set_page_config(page_title="SpamSentry", page_icon="📩", layout="centered")

# Custom CSS for Styling
st.markdown("""
    <style>
        body { background-color: #872341; }
        .stTextArea textarea { border-radius: 8px; font-size: 16px; padding: 10px; }
        .stButton button { background-color: #007bff; color: white; border-radius: 8px; font-size: 18px; padding: 10px 20px; }
        .stMarkdown { font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# Layout: Image & Title
col1, col2 = st.columns([1, 3])
with col1:
    st.image("spamsentry_logo.jpg", width=300)
with col2:
    st.title("Spam-Sentry: Spam Email Detector")
    st.write("🚀 Enter a message to check if it's **Spam** or **Ham**.")

# User Input
user_input = st.text_area("✍️ Type your message below:", "", height=150,)

# Expander for additional info
with st.expander("ℹ️ How does this work?"):
    st.write("This model analyzes your message using **Natural Language Processing (NLP)** and classifies it as **Spam** or **Ham**.")
    st.write("Spam-Sentry uses Naive Bayes Classifier for classification.")
    st.write(f"This model has an accuracy of {metrics['Accuracy']}%.")

# Button to Predict
if st.button("🚀 Analyze Message"):
    if user_input.strip():
        with st.spinner("Analyzing message..."):
            time.sleep(2)
            input_vectorized = vectorizer.transform([user_input])
            prediction = model.predict(input_vectorized)
            result = "🚨 **Spam Detected!**" if prediction[0] == 1 else "✅ **Ham - Safe Message**"

        # Display Prediction Result
        st.subheader("📝 Result:")
        st.markdown(result, unsafe_allow_html=True)

        # Add Confetti Animation if Message is Ham
        if prediction[0] == 0:
            st.balloons()
    else:
        st.warning("⚠️ Please enter a message before clicking the button!")

# Display Model Metrics as a Bar Chart
st.subheader("📊 Model Performance Metrics")
fig, ax = plt.subplots()
ax.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green', 'red'])
ax.set_ylabel("Percentage")
ax.set_ylim(0, 100)
ax.set_title("Evaluation Metrics")
st.pyplot(fig)

# Display Sample Validation Graph
st.subheader("📈 Model Validation Performance")
x = np.arange(1, 6)  # Example epochs
train_accuracy = [85, 88, 90, 92, 95]
val_accuracy = [82, 86, 89, 91, 94]
fig, ax = plt.subplots()
ax.plot(x, train_accuracy, label="Train Accuracy", marker='o')
ax.plot(x, val_accuracy, label="Validation Accuracy", marker='s')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Training vs Validation Accuracy")
ax.legend()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Made By Team SpamSentry")
