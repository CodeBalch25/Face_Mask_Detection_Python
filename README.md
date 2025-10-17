# COVID-19 Face Mask Detector

A real-time face mask detection system using deep learning and computer vision to identify whether people are wearing face masks. Built with TensorFlow, OpenCV, and Streamlit.

## Features

- **Real-time Detection**: Detect face masks in uploaded images with bounding boxes and confidence scores
- **User-Friendly Interface**: Interactive Streamlit web application for easy image uploads
- **High Accuracy**: Uses MobileNetV2 architecture for efficient and accurate predictions
- **Visual Feedback**: Color-coded bounding boxes (Green for mask, Red for no mask)
- **Confidence Scores**: Displays prediction confidence percentage for each detection

## Tech Stack

- **Deep Learning**: TensorFlow/Keras with MobileNetV2
- **Computer Vision**: OpenCV for image processing and face detection
- **Frontend**: Streamlit for the web interface
- **Model**: Pre-trained face detector with custom mask classification model

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/CodeBalch25/Face_Mask_Detection_Python.git
cd Face_Mask_Detection_Python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the required model files:
   - Place `face_mask_detector.model` in the root directory
   - Place face detector files in `face_detector/` directory:
     - `deploy.prototxt`
     - `res10_300x300_ssd_iter_140000.caffemodel`

4. Create the necessary directories:
```bash
mkdir images
mkdir face_detector
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the App

1. Select "Image" from the sidebar
2. Upload an image (JPG or PNG format)
3. Click "Process" to detect face masks
4. View the results with bounding boxes and confidence scores

## Project Structure

```
Face_Mask_Detection_Python/
├── app.py                          # Main Streamlit application
├── detect_mask_in_image.py         # Image detection module
├── detect_mask_in_video.py         # Video detection module (future feature)
├── searching_face.py               # Face search utilities
├── training_detection.py           # Model training script
├── requirements.txt                # Python dependencies
├── face_detector/                  # Pre-trained face detector models
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── images/                         # Temporary image storage
└── README.md                       # Project documentation
```

## How It Works

1. **Face Detection**: Uses OpenCV's DNN module with a pre-trained Caffe model to detect faces
2. **Preprocessing**: Detected faces are extracted, resized to 224x224, and preprocessed for the neural network
3. **Classification**: MobileNetV2-based model classifies each face as "Mask" or "No Mask"
4. **Visualization**: Draws bounding boxes with labels and confidence scores on the original image

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224x3
- **Output**: Binary classification (Mask / No Mask)
- **Face Detector**: SSD (Single Shot Detector) with ResNet-10 backbone

## Future Enhancements

- [  ] Real-time webcam detection
- [ ] Batch processing for multiple images
- [ ] Mobile app deployment
- [ ] Dataset expansion for improved accuracy
- [ ] Support for different mask types

## Technical Details

**Detection Pipeline:**
1. Image input → Blob conversion
2. Face detection using DNN
3. ROI extraction and preprocessing
4. Mask classification
5. Result visualization

**Performance:**
- Detection confidence threshold: 0.5
- Image preprocessing: BGR to RGB conversion
- Real-time capable on modern hardware

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

**Timothy Balch** - [@CodeBalch25](https://github.com/CodeBalch25)

## Acknowledgments

- TensorFlow team for the MobileNetV2 architecture
- OpenCV community for computer vision tools
- Streamlit for the excellent web framework

## Tags

`machine-learning` `computer-vision` `tensorflow` `opencv` `covid19` `deep-learning` `streamlit` `python` `face-detection` `image-classification`
