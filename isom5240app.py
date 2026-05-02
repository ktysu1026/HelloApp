import streamlit as st
from transformers import pipeline
from PIL import Image

# Set up the app title and layout
st.title("👥 Gender Classification using ViT")
st.write("Upload an image to predict the person's gender.")

# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_classifier():
    return pipeline("image-classification", model="prithivMLmods/MetaCLIP-2-Gender-Identifier")

gender_classifier = load_classifier()

# File uploader for user images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Classifying gender..."):
        # Classify gender
        gender_predictions = gender_classifier(image)
        
        # Sort predictions by score (highest first)
        gender_predictions = sorted(gender_predictions, key=lambda x: x['score'], reverse=True)
        
        # Display results with appropriate emoji
        top_prediction = gender_predictions[0]
        gender_label = top_prediction['label']
        
        # Add gender-specific emoji
        gender_emoji = "👩" if gender_label.lower() == "female" else "👨"
        
        st.success(f"{gender_emoji} **Predicted Gender: {gender_label}**")
        st.write(f"Confidence Score: {top_prediction['score']:.2%}")
        
        # Optional: Show all probabilities in a chart
        with st.expander("See detailed probabilities"):
            labels = [p['label'] for p in gender_predictions]
            scores = [p['score'] for p in gender_predictions]
            st.bar_chart(data=dict(zip(labels, scores)))
            
        # Add disclaimer
        st.caption("⚠️ Note: This model predicts based on visual patterns only and should not be used to infer someone's gender identity. Results are for demonstration purposes only.")

