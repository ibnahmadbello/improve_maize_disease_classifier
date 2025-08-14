# ======================
# 3. STREAMLIT DEPLOYMENT APP
# ======================
# Save this as app.py
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import pickle

# Load assets
@st.cache_resource
def load_assets():
    model = joblib.load('maize_rf_model.pkl')
    with open('feature_extractor.pkl', 'rb') as f:
        feature_extractor = pickle.load(f)
    with open('class_names.pkl', 'rb') as f:
        classes = pickle.load(f)
    return model, feature_extractor, classes

def extract_features(img, model):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

# App UI
st.title("🌽 Maize Disease Classifier")
st.markdown("Upload an image of a maize leaf for disease diagnosis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    model, feature_extractor, classes = load_assets()
    img = Image.open(uploaded_file).convert('RGB')
    
    st.image(img, caption="Uploaded Image", width=300)
    
    with st.spinner('Analyzing...'):
        features = extract_features(img, feature_extractor)
        proba = model.predict_proba([features])[0]
    
    st.subheader("Diagnosis Results")
    pred_class = classes[np.argmax(proba)]
    confidence = max(proba) * 100
    
    if pred_class != 'Healthy':
        st.error(f"⚠️ Detected: {pred_class} ({confidence:.1f}% confidence)")
    else:
        st.success(f"✅ Healthy ({confidence:.1f}% confidence)")
    
    # Show probabilities
    st.subheader("Disease Probabilities")
    prob_data = {classes[i]: float(proba[i]) for i in range(len(classes))}
    st.bar_chart(prob_data)


