import streamlit as st
from PIL import Image
import easyocr
from gtts import gTTS
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from collections import Counter
import numpy as np
import os

# Set up Google API Key
with open("C:/Users/ATHUL/Desktop/gemini-apikey.txt") as f:
    GOOGLE_API_KEY = f.read().strip()

# Initialize LangChain with Google Generative AI model
chat_model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")

# Initialize EasyOCR
ocr_reader = easyocr.Reader(['en'], gpu=False)

# Initialize YOLO
yolo_model = YOLO("C:/Users/ragesh/OneDrive/Desktop/Gen AI APP/Langachain Project/yolov3.pt")

# Initialize BLIP
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Streamlit Page Configuration
st.set_page_config(page_title="Vision Mate", layout="wide")

# Sidebar
st.sidebar.title("Vision Mate")
st.sidebar.markdown(
    """
    <div style="font-size: 16px; line-height: 1.6; color: #444;">
    <strong>Vision Mate</strong> is an AI-powered assistant designed for visually impaired individuals to:
    <ul>
        <li><strong>Describe images</strong> and provide concise summaries.</li>
        <li><strong>Identify potential hazards</strong> in the surroundings.</li>
        <li><strong>Extract text</strong> and summarize details.</li>
    </ul>
    <h4>How to Use:</h4>
    <ol>
        <li>Upload an image using the uploader below.</li>
        <li>Wait for the system to process the image.</li>
        <li>View a summary, detected objects, and hazards.</li>
        <li>Listen to an audio summary of the results.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True
)

# Title
st.markdown("<h1 style='color: #00D4FF; text-align: center;'>Vision Mate: Your AI Vision Assistant</h1>", unsafe_allow_html=True)

# File Uploader
uploaded_image = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])

# Utility Functions
def extract_text(image):
    """Extract text using EasyOCR."""
    results = ocr_reader.readtext(np.array(image), detail=0)
    return " ".join(results) if results else "No text detected."

def detect_objects(image):
    """Detect objects using YOLO and group them with counts."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image_np = np.array(image)
    results = yolo_model(image_np)
    detected_labels = [yolo_model.names[int(box.cls)] for box in results[0].boxes]
    object_counts = Counter(detected_labels)
    formatted_objects = [
        f"{count} {label}{'s' if count > 1 else ''}" for label, count in object_counts.items()
    ]
    return ", ".join(formatted_objects) if formatted_objects else "No objects detected."

def describe_scene(image):
    """Generate a concise scene description using the BLIP model."""
    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

def generate_prediction(objects, text, scene_description):
    """Generate a concise AI Assistance response."""
    base_prompt = f'''
    Provide a concise summary of the uploaded image, including:
    - Key elements (e.g., objects, interactions, or people).
    - Any potential hazards or risks based on the scene.
    - Relevant extracted text if applicable.

    Scene Description: {scene_description}
    Detected Objects: {objects}
    Extracted Text: {text}
    '''
    prompt = ChatPromptTemplate.from_messages([("system", base_prompt), ("user", base_prompt)])
    output_parser = StrOutputParser()
    chain = prompt | chat_model | output_parser
    try:
        result = chain.invoke({})
        return result
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        return "Sorry, something went wrong. Please try again."

def text_to_audio(text):
    """Convert text to speech."""
    audio_file = "output.mp3"
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(audio_file)
    return audio_file

# Main Application
if uploaded_image:
    col1, col2 = st.columns([1, 2])

    # Display Image in Left Column
    with col1:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Display Results in Right Column
    with col2:
        with st.spinner("Processing image..."):
            text = extract_text(image)
            objects = detect_objects(image)
            scene_description = describe_scene(image)
            prediction = generate_prediction(objects, text, scene_description)

        st.markdown("<h2 style='color: #00D4FF;'>AI Assistance</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 16px; line-height: 1.6; color: #444;'>{prediction}</div>", unsafe_allow_html=True)

        st.markdown("<h2 style='color: #00D4FF;'>Detected Objects</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 16px; line-height: 1.6; color: #444;'>{objects}</div>", unsafe_allow_html=True)

        st.markdown("<h2 style='color: #00D4FF;'>Extracted Text</h2>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 16px; line-height: 1.6; color: #444;'>{text}</div>", unsafe_allow_html=True)

        st.markdown("<h2 style='color: #00D4FF;'>Audio Summary</h2>", unsafe_allow_html=True)
        with st.spinner("Generating audio..."):
            combined_text = f"{prediction}\nDetected Objects: {objects}\nExtracted Text: {text}"
            audio_file = text_to_audio(combined_text)
            with open(audio_file, "rb") as file:
                st.audio(file.read(), format="audio/mp3")
            os.remove(audio_file)
else:
    st.info("Upload an image to see results.")
