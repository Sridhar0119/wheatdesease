# WHEAT DISEASE DETECTION SYSTEM - USER MANUAL

## 1. Introduction
This manual guides users effectively through the **Wheat Disease Detection System**. The application is designed to be intuitive, requiring minimal technical expertise.

## 2. Accessing the System
1.  **Launch the Application**: Ensure the Flask server is running (`python app.py`).
2.  **Open Browser**: Go to `http://localhost:5000` (or your deployed IP).
3.  **Landing Page**: You will see the Login capabilities.

## 3. Account Management
### 3.1 Registration
*   **Step 1**: Click the "Register" link on the login card.
*   **Step 2**: Enter your full name, a unique username, valid email, and a strong password.
*   **Step 3**: Confirm your password and click "Register".
*   *Note*: If the email is already taken, an error will appear.

### 3.2 Login
*   **Step 1**: Enter your registered Username and Password.
*   **Step 2**: Click "Login" to access the dashboard.

## 4. Disease Detection (Analysis)
### 4.1 Uploading an Image
1.  Navigate to the **"Analysis"** page via the customized dashboard or navigation bar.
2.  Click the **"Choose File"** button.
3.  Select an image from your computer/phone.
    *   **Supported Formats**: `.jpg`, `.jpeg`, `.png`, `.gif`.
    *   **Requirement**: Image must be clear and focused on the leaf.
4.  Click **"Upload and Analyze"**.

### 4.2 Understanding Results
The system provides a detailed report card:
*   **Status**: Healthy or Diseased (e.g., "Leaf Rust").
*   **Confidence Score**: How sure the AI is (e.g., 98%).
*   **Severity**: Low / Medium / High (Based on spread estimation).
*   **Action Plan**:
    *   *Medicine*: List of recommended chemicals/fungicides.
    *   *Prevention*: Agricultural practices to prevent recurrence.

## 5. History Feature
*   Click on **"History"** in the menu to see your last 10 scans.
*   Use this to compare current scan results with previous ones to monitor treatment effectiveness.

## 6. Troubleshooting
*   **"No file selected"**: Ensure you actually picked a file before clicking upload.
*   **"Invalid file type"**: Convert your image to JPG or PNG.
*   **"Server Error"**: Contact the administrator or check your internet connection.
