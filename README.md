# Instrunet-AI

Deep Learning-Based Musical Instrument Recognition System

1. Project Overview

Instrunet-AI is a deep learning–driven audio classification system designed to identify musical instruments from uploaded audio recordings. The system leverages convolutional neural networks (CNNs) trained on spectral audio features and provides an interactive web interface for inference and visualization.

The project demonstrates end-to-end machine learning workflow implementation, including data preprocessing, feature engineering, model training, and deployment using a web-based interface.

---

2. Objectives

* Develop a robust CNN-based classification model for instrument recognition.
* Implement efficient audio preprocessing and feature extraction.
* Provide an interactive deployment interface using Streamlit.
* Ensure modular, scalable, and maintainable code structure.

---

3. Key Features

3.1 Audio Processing

* Conversion to mono and normalization
* Segmentation into fixed-duration samples
* Log Mel Spectrogram extraction
* MFCC feature extraction
* Temporal landmark detection (onset-based analysis)

3.2 Model Architecture

* Convolutional Neural Network (CNN)
* Trained on spectrogram-based representations
* Multi-class classification (11 instrument categories)
* Segment-wise prediction with aggregated final inference

3.3 Deployment Interface

* Web-based UI built using Streamlit
* Audio file upload (.wav / .mp3)
* Real-time prediction output
* Visualization of waveform and spectrogram
* Probability distribution display for all instrument classes

---

4. Instrument Classes

The system classifies audio into the following categories:

* Cello
* Clarinet
* Flute
* Acoustic Guitar
* Electric Guitar
* Organ
* Piano
* Saxophone
* Trumpet
* Violin
* Human Voice

---

5. Technology Stack

| Component               | Technology          |
| ----------------------- | ------------------- |
| Programming Language    | Python              |
| Deep Learning Framework | TensorFlow / Keras  |
| Audio Processing        | Librosa             |
| Visualization           | Matplotlib / Plotly |
| Deployment              | Streamlit           |
| Data Handling           | NumPy, Pandas       |

---

6. System Architecture

6.1 Workflow

1. Audio file upload through web interface
2. Signal preprocessing and segmentation
3. Spectral feature extraction (Mel Spectrograms / MFCCs)
4. CNN-based prediction per segment
5. Aggregation of segment predictions
6. Final instrument classification output

---

7. Repository Structure


Instrunet-AI/
│
├── app.py                      # Streamlit deployment interface
├── preprocessing.py            # Audio preprocessing and feature extraction
├── models/                     # Trained CNN model files (.h5)
├── metadata.csv                # Dataset metadata and labels
├── class_counts.png            # Class distribution visualization
├── requirements.txt            # Project dependencies
└── LICENSE.md


---

8. Installation & Setup

8.1 Prerequisites

* Python 3.8 or above
* pip package manager

8.2 Installation

Clone the repository:

bash
git clone <repository-url>
cd Instrunet-AI


Install required dependencies:

bash
pip install -r requirements.txt


---

9. Running the Application

Launch the Streamlit application:

bash
streamlit run app.py


The application will open in a local web browser, allowing audio file upload and real-time prediction.

---

10. Model Training (Optional)

To retrain or preprocess data:

bash
python preprocessing.py


This script:

* Processes raw audio dataset
* Extracts spectral features
* Saves processed features for training
* Prepares metadata for supervised learning

Model training can be executed via a dedicated training script (if included), using the generated feature files.

---

11. Performance Considerations

* Segment-wise averaging improves robustness.
* Log-scaled Mel spectrograms enhance frequency resolution.
* Batch processing recommended for large datasets.
* GPU acceleration significantly reduces training time.

---

12. Scalability & Future Enhancements

* Multi-label instrument detection
* Live audio stream integration
* REST API deployment using FastAPI
* Model optimization for edge deployment
* Integration with larger audio datasets (e.g., FMA, NSynth)

---

13. License

This project is licensed under the MIT License.

--

14. Contributors

* Nandini
* Akilesh Prasad V
* Namrata More
* Nalam Venkata Surya Harshith
* Sekar Bhuvana 

---
