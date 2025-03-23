import streamlit as st
import joblib
import time

# Load the trained model and vectorizer
model = joblib.load("spamsentry.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Custom Page Configuration
st.set_page_config(page_title="SpamSentry", page_icon="📩", layout="centered")

# Custom CSS for Styling
st.markdown("""
    <style>
        body { background-color: #872341; }
        .stTextArea textarea { background-color: ; border-radius: 8px; font-size: 16px; padding: 10px; }
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

# Button to Predict
if st.button("🚀 Analyze Message"):
    if user_input.strip():  # Check if input is not empty
        with st.spinner("Analyzing message..."):
            time.sleep(2)  # Simulate processing time
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

# Footer
st.markdown("---")
st.markdown("Made By Team SpamSentry")
