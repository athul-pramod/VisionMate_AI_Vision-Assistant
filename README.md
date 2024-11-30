# VisionMate: AI Vision Assistant

**VisionMate** is an AI-powered application designed to assist visually impaired individuals by analyzing images and providing actionable insights in textual and audio formats. This project leverages state-of-the-art AI models for scene understanding, object detection, text extraction, and audio summarization.

## Features
- **Image Analysis**: Upload an image and receive a detailed analysis.
- **Scene Description**: Generate concise and meaningful scene descriptions.
- **Object Detection**: Identify objects and potential hazards in the scene.
- **Text Extraction**: Extract readable text from images using OCR.
- **Audio Summaries**: Convert text-based results into audio for accessibility.

## Technologies Used
- **Frontend**: Streamlit for the user interface.
- **AI Models**:
  - **YOLO**: For object detection.
  - **BLIP**: For scene understanding.
  - **EasyOCR**: For text extraction.
  - **LangChain + Google Generative AI**: For generating actionable summaries.
- **Text-to-Speech**: Google Text-to-Speech (gTTS).
- **Programming Language**: Python.

## How It Works
1. **Upload an Image**: Use the Streamlit interface to upload an image in JPG or PNG format.
2. **Processing**:
   - YOLO detects objects and hazards.
   - BLIP generates a concise scene description.
   - EasyOCR extracts text from the image.
   - LangChain integrates the outputs and generates an actionable summary.
3. **Output**:
   - View a textual summary, detected objects, and extracted text.
   - Listen to an audio summary of the results.
