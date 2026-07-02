# Spot the Fake Photo

A lightweight computer vision system that distinguishes between **real photographs** and **photos of digital screens (recaptured images)** using handcrafted image-processing techniques and a Logistic Regression classifier.

> Developed for the **SalesCode AI Computer Vision & Machine Learning Take-Home Assignment**

---

## Live Demo

**Streamlit App**

https://YOUR-STREAMLIT-APP.streamlit.app

---


## Features

- FFT Frequency Analysis
- Moiré Pattern Detection
- Laplacian Sharpness
- Image Entropy
- Saturation Statistics
- Highlight / Glare Detection
- RGB Channel Correlation
- Edge Density
- Noise Estimation
- Local Binary Pattern (LBP)
- Lightweight Logistic Regression Classifier

---

## Example Prediction

Example command:

```bash
python predict.py image.jpeg
```

Output:

```text
0.94
```

Interpretation:

- **0 → Real Photo**
- **1 → Photo of a Screen**

Since **0.94** is close to **1**, the image is classified as a **Photo of a Screen**.

---

## Dataset

The dataset was manually collected using a smartphone.

| Category | Images |
|----------|-------:|
| Real Photos | 62 |
| Screen Photos | 64 |
| **Total** | **126** |

The dataset includes:

- Different lighting conditions
- Multiple viewing angles
- Various distances
- Phone screens
- Laptop screens
- Tablet screens

---

## Feature Extraction

The following handcrafted features are extracted from each image:

- FFT High Frequency Ratio
- FFT Peakiness
- Moiré Band Energy
- Laplacian Variance
- Normalized Laplacian Variance
- Image Entropy
- Saturation Mean
- Saturation Standard Deviation
- Highlight Ratio
- Largest Highlight Blob
- RGB Channel Correlation
- Edge Density
- Noise Estimation
- Local Binary Pattern (LBP) Histogram

**Feature Vector Length:** 23

---

## Model

- Feature Extraction using OpenCV & NumPy
- StandardScaler
- Logistic Regression (Scikit-learn)

---

## Performance

### Dataset

- 62 Real Images
- 64 Screen Images

### Results

| Metric | Value |
|-------|------:|
| Held-out Test Accuracy | **92.31%** |
| 5-Fold Cross Validation Accuracy | **80.00%** |

---

## Latency

Average inference latency:

**≈151 ms per image**

Measured on a Windows laptop CPU using Python.

---

## Cost

The model is designed for **on-device inference**.

- **On-device:** Approximately **$0 per image**
- **Cloud Deployment:** Very low CPU cost (well below \$1 per million images depending on infrastructure)

---

## Project Structure

```
realorscreen/
│
├── app.py
├── features.py
├── train.py
├── predict.py
├── latency.py
├── requirements.txt
├── README.md
│
├── models/
│   └── logistic_model.pkl
│
└── dataset/
    ├── real/
    └── screen/
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git

cd YOUR_REPOSITORY

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

The trained model is saved to:

```
models/logistic_model.pkl
```

---

## Prediction

```bash
python predict.py image.jpg
```

Returns a probability between **0** and **1**.

- **0 → Real Photo**
- **1 → Photo of a Screen**

---

## Streamlit Demo

Run locally:

```bash
streamlit run app.py
```

The web application allows users to:

- Capture an image using their webcam
- Upload an image
- View the prediction score instantly

---

## Future Improvements

- Collect a larger and more diverse dataset
- Include OLED, LCD and high-refresh-rate displays
- Improve robustness against new spoofing techniques
- Optimize feature extraction using OpenCV Mobile
- Explore lightweight ensemble classifiers

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## Author

**Prachi Yadav**
