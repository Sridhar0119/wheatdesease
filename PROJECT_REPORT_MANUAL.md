# Wheat Disease Detection System
## User Manual & Project Report

**Project Title:** AI-Powered Wheat Disease Detection System using Deep Learning  
**Date:** December 2025  
**Version:** 1.0

---

## 1. Executive Summary

The Wheat Disease Detection System is a web-based application designed to assist farmers and agricultural experts in the early identification and management of wheat crop diseases. Leveraging the power of Deep Learning, specifically the ResNet50 architecture, this system analyzes leaf images to detect common ailments such as Leaf Rust, Mildew, Wheat Loose Smut, and Yellow Rust. By providing instant, accurate diagnoses alongside severity estimates and treatment recommendations, the system aims to reduce crop loss and improve agricultural productivity.

---

## 2. Project Introduction

### 2.1 Problem Statement
Wheat is a staple food crop globally, but its production is severely threatened by various fungal diseases. Early detection is critical, yet manual monitoring is labor-intensive and requires expert knowledge that may not always be accessible. Misdiagnosis or delayed treatment can lead to devastating yield losses.

### 2.2 Proposed Solution
Our solution is an automated, accessible platform where users can upload images of wheat leaves. The system uses a pre-trained Convolutional Neural Network (CNN) to classify the health status of the plant. It goes beyond simple classification by calculating a "Severity Score" and providing actionable medical advice (fungicides, preventive measures) stored in a knowledge base.

---

## 3. Technical Architecture

### 3.1 Technology Stack
*   **Backend Framework:** Python Flask
*   **Deep Learning Framework:** TensorFlow / Keras
*   **Model Architecture:** ResNet50 (Transfer Learning with custom top layers)
*   **Database:** SQLite (Lightweight, serverless relational database)
*   **Frontend:** HTML5, CSS3, JavaScript (Jinja2 Templates)

### 3.2 Key Components
1.  **Authentication Module:** Secure registration and login system ensuring user data privacy.
2.  **Prediction Engine:**
    *   Preprocessing: Resizing to 224x224, normalization.
    *   Inference: The `final_wheat_disease_resnet50.h5` model predicts probabilities for 5 classes.
    *   Post-processing: Confidence thresholding (default 35%) to filter uncertainty.
3.  **Severity Estimator:** An algorithm that correlates prediction confidence and disease type to estimate the percentage of infection spread.
4.  **Recommendation System:** A dictionary-based lookup providing specific chemical treatments (e.g., Tebuconazole, Propiconazole) based on the detected disease.

---

## 4. Installation & Setup Guide

This section outlines how to set up the project on a local machine.

### 4.1 Prerequisites
*   Operating System: Windows 10/11, Linux, or macOS.
*   Python 3.8 or higher installed options.
*   pip (Python Package Installer).

### 4.2 Installation Steps
1.  **Unzip the Project File:**
    Extract the project contents to a local directory (e.g., `D:\WDD\`).

2.  **Install Dependencies:**
    Open a terminal/command prompt in the project folder and run:
    ```bash
    pip install flask tensorflow flask-bcrypt pillow numpy
    ```

3.  **Verify Model File:**
    Ensure `final_wheat_disease_resnet50.h5` is present in the root directory.

4.  **Run the Application:**
    Execute the following command:
    ```bash
    python app.py
    ```
    *Wait for the message: `Running on http://127.0.0.1:5000`*

---

## 5. User Manual

### 5.1 Getting Started

**Accessing the System:**
Open your web browser (Chrome, Firefox, or Edge) and navigate to `http://localhost:5000`.

**Registration:**
1.  Click on the "Register" link on the login page.
2.  Fill in your `Username`, `Email`, `Full Name`, and `Password`.
3.  Click "Register". You will be redirected to the login page upon success.

**Login:**
1.  Enter your registered credentials.
2.  Click "Login" to access the dashboard.

### 5.2 Analyzing a Wheat Leaf

**Step 1: Navigate to Analysis**
From the Home Dashboard, click the "Analyze Image" button.

**Step 2: Upload Image**
1.  Click the "Choose File" button.
2.  Select a clear JPG or PNG image of a wheat leaf from your computer.
    *   *Tip: Ensure the leaf occupies most of the frame for best accuracy.*
3.  Click "Analyze".

**Step 3: Review Results**
The system will display a report card containing:
*   **Diagnosis:** The name of the detected disease (e.g., "Leaf Rust") or "Healthy Wheat".
*   **Confidence Score:** How certain the AI is about the result (e.g., 98.5%).
*   **Severity Level:** Categorized as "Low", "Medium", "High", or "Critical" based on the estimated spread.
*   **Treatment Plan:** specific chemicals (like *Triazole fungicides*) and prevention tips.

### 5.3 Viewing History
Click "History" in the navigation bar to see a log of your past 10 analyses. This is useful for tracking disease progression over time.

---

## 6. Interpreting the AI Score (Confidence)

The system provides a **Confidence Score** with every prediction. Here is how to interpret it:

*   **90% - 100%:** High certainty. The model is very sure of its diagnosis.
*   **70% - 89%:** Moderate certainty. The symptoms are likely present but might be early-stage.
*   **Below 50%:** Uncertainty. If the score is low, the system may default to "Healthy" to avoid false alarms, or flag it for manual review.

*Note: The model has been calibrated to reduce false positives. If an image is blurry or contains non-leaf objects, the results may vary.*

---

## 7. Troubleshooting

| **Issue** | **Possible Cause** | **Solution** |
| :--- | :--- | :--- |
| **"Model not loaded" Error** | The `.h5` file is missing or corrupt. | Ensure `final_wheat_disease_resnet50.h5` is in the main folder. |
| **"Invalid file type"** | Uploading a PDF or Word doc. | Only `.jpg`, `.jpeg`, `.png` images are supported. |
| **Low Accuracy Results** | Blurry image or bad lighting. | Take a clear photo in daylight, focusing on the leaf symptoms. |
| **Database Locked** | Multiple app instances running. | Close other Python windows and restart the app. |

---

## 8. Conclusion

The Wheat Disease Detection System represents a significant step forward in digital agriculture. By democratizing access to expert-level disease diagnosis, we empower farmers to make data-driven decisions, ultimately securing their harvest and livelihood. The utilization of ResNet50 ensures robust performance, while the user-friendly interface makes the technology accessible to users with varying levels of technical expertise.
