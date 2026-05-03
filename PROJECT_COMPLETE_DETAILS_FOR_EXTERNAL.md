# 🌾 WHEAT VISION AI — WHEAT DISEASE DETECTION SYSTEM
### Complete Project Details for External Explanation / Viva Voce

---

## ✅ PROJECT TITLE
**"Wheat Disease Detection System using Deep Learning (ResNet50)"**
> Also referred to as **"Wheat Vision AI"** in the application UI.

---

## 🎯 ONE-LINE STATEMENT (What is the Project?)
> An AI-powered web application that allows farmers and agricultural workers to upload a photo of a wheat leaf and instantly receive a diagnosis of what disease (if any) the plant has — along with severity levels, treatment advice, and history tracking.

---

## 🔍 PROBLEM STATEMENT

Wheat is one of the most important cereal crops in the world. Diseases like **Leaf Rust**, **Yellow Rust**, **Loose Smut**, and **Crown & Root Rot** can devastate harvests if not caught early.

**Traditional Challenges:**
- Manual inspection by farm experts is **slow**, **expensive**, and **subjective**
- Visual similarity between diseases causes **misdiagnosis** by untrained eyes
- No **digital record-keeping** of disease outbreaks in fields
- Expert pathologists are inaccessible in **rural/remote areas**

**Our Solution:** An automated, web-based AI system that diagnoses disease from a leaf image in under 1 second.

---

## 🎯 OBJECTIVES

| # | Objective |
|---|-----------|
| 1 | Build a Deep Learning model using **ResNet50** for wheat disease classification |
| 2 | Achieve **>90% accuracy** on the 5 disease classes |
| 3 | Deploy as a **Flask web application** with a clean, responsive UI |
| 4 | Implement **disease severity estimation** (Spread % + Severity Level) |
| 5 | Provide **treatment recommendations** for each detected disease |
| 6 | Build a **secure authentication system** (login, register, role-based access) |
| 7 | Store **disease history** per user in a SQLite database |

---

## 🖥️ TECHNOLOGY STACK

| Layer | Technology | Why Used |
|-------|-----------|----------|
| **AI Model** | ResNet50 (TensorFlow/Keras) | Deep residual learning, pre-trained on ImageNet |
| **Backend** | Python Flask | Lightweight, easy REST API integration |
| **Frontend** | HTML5 + CSS3 + JavaScript | Responsive, no framework needed |
| **Database** | SQLite3 | Embedded DB, perfect for field deployment |
| **Security** | Werkzeug Bcrypt | Password hashing for user credentials |
| **Fonts/Icons** | Google Fonts (Inter), Bootstrap Icons | Modern professional UI |

---

## 🏗️ SYSTEM ARCHITECTURE

The application follows the **MTV (Model-Template-View)** pattern of Flask:

```
User (Browser)
    │  HTTP Request (GET / POST)
    ▼
┌──────────────────────────────────────────┐
│            Flask App (app.py)            │
│  ─────────────────────────────────────   │
│  Route: /            → index.html UI     │
│  Route: /api/login   → Auth logic        │
│  Route: /api/register → User creation    │
│  Route: /predict     → AI Inference      │
│  Route: /api/admin/users → Admin API     │
└────────────┬─────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
ResNet50 Model   SQLite Database
(best_wheat_     (wheat_disease.db)
 disease_resnet50.h5)
```

### Request Flow for Prediction:
1. User uploads a leaf image via the browser
2. Flask receives it at the `/predict` endpoint
3. Image is saved temporarily to the `uploads/` folder
4. `prepare_image()` preprocesses it: resize(224×224) → numpy array → ResNet50 normalization
5. `model.predict()` returns a probability array of 5 values
6. `np.argmax()` finds the class with highest probability
7. Result + disease info is returned as JSON
8. Frontend renders the result with severity, symptoms, and treatment cards

---

## 🧠 AI MODEL DETAILS — ResNet50

### What is ResNet50?
- **Residual Network** with **50 layers** deep
- Developed by Microsoft Research (He et al., 2016)
- Key innovation: **Skip Connections (Residual Blocks)**
  - Bypasses 2-3 layers to prevent vanishing gradient problem
  - Allows training of very deep networks without accuracy degradation

### Why ResNet50 (and not others)?
| Model | Accuracy | Speed | Size | Chosen? |
|-------|--------|-------|------|---------|
| MobileNet | ~85% | Fast | Small | ❌ Less accurate |
| EfficientNet | ~92% | Medium | Medium | ❌ More complex |
| **ResNet50** | **~93%** | **Good** | **120MB** | ✅ **Best balance** |
| ResNet152 | ~94% | Slow | 267MB | ❌ Too heavy |

### Transfer Learning Strategy:
1. **Base Model**: ResNet50 loaded with `weights='imagenet'` (14M images pre-trained)
2. **Freeze Phase 1**: Base layers frozen → only custom head is trained (10 epochs)
3. **Fine-Tune Phase 2**: Top 40 layers unfrozen → trained at low LR (1e-5, 50 epochs)
4. **Custom Head**:
   ```
   ResNet50 Base → GlobalAveragePooling2D → Dense(1024, relu, L2 reg)
                 → Dropout(0.5) → Dense(5, softmax)
   ```

### Disease Classes (Model Outputs):
| Index | Class Name | Description |
|-------|-----------|-------------|
| 0 | Crown and Root Rot | Fungal infection at base of plant |
| 1 | Healthy Wheat | No disease detected |
| 2 | Leaf Rust | Orange pustules on leaves (Puccinia triticina) |
| 3 | Wheat Loose Smut | Black fungal spores replacing grain |
| 4 | Unknown | Low-confidence / unrecognized input |

---

## 📊 TRAINING PIPELINE

### Data Augmentation (to prevent overfitting):
```python
ImageDataGenerator(
    rotation_range=40,          # Handles rotated leaf images
    zoom_range=0.3,             # Handles zoomed in/out images
    brightness_range=[0.7, 1.3],# Handles different lightings in fields
    horizontal_flip=True,       # Mirrors image
    vertical_flip=True,         # Handles any orientation
    preprocessing_function=preprocess_input  # ResNet normalization
)
```

### Class Imbalance Handling:
- **Focal Loss** used instead of standard CrossEntropy
  - Focuses learning on hard, misclassified examples
  - Formula: `FL = α × (1 - pt)^γ × CE`  where γ=2.0, α=0.25
- **Balanced Class Weights** computed via `sklearn.utils.class_weight.compute_class_weight`
- Healthy Wheat class (minority: ~134 images) boosted by 1.5× extra weight

### Training Stages:
| Phase | Layers | Learning Rate | Epochs |
|-------|--------|--------------|--------|
| Phase 1 (Head) | Custom layers only | 0.001 | 10 |
| Phase 2 (Fine-tune) | Top 40 of ResNet | 0.00001 | 50 |

### Callbacks Used:
- `ModelCheckpoint` — saves only best `val_accuracy`
- `EarlyStopping` — stops if no improvement for 15 epochs
- `ReduceLROnPlateau` — halves LR if plateau detected

---

## 🔐 SECURITY & AUTHENTICATION

### User Management:
- Credentials stored in `secure_users_db.json`
- Passwords hashed using **Werkzeug's PBKDF2+SHA256** (industry standard)
- **Role-based access**: `admin` vs `user` roles
- New registrations forced to `user` role (cannot self-assign admin)

### API Endpoints:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Render main UI |
| `/api/register` | POST | Register new user |
| `/api/login` | POST | Authenticate & return user info |
| `/api/admin/users` | GET | List all users (admin only) |
| `/predict` | POST | Upload image → receive disease prediction |

### Default Users:
| Username | Password | Role |
|----------|----------|------|
| admin | admin123 | Admin |
| mrsrii | mrsrii123 | Admin |
| user | user123 | User |

---

## 🖼️ USER INTERFACE (UI/UX)

### Screen Layout:
- **Sidebar Navigation** — Home (Diagnose), History, Admin Panel (if admin)
- **Topbar** — Current page title + Google Translate integration
- **Content Area** — Dynamic views based on selected nav item

### Key UI Views:

#### 1. Login / Register Screen
- Gradient overlay (indigo → deep purple)
- Slide-up animation on card appearance
- Toggle between Login and Register

#### 2. Diagnose (Main Page)
- Drag-and-drop / click-to-upload zone
- Image preview before analysis
- **Analyze** button triggers prediction

#### 3. Results Display (4 components):
  1. **Uploaded Image Card** — shows your leaf photo
  2. **Diagnosis Card** — disease name, confidence badge, severity badge, disease description
  3. **Treatment Plan Card** — Symptoms list, Medicine checklist, Prevention tips
  4. **Detailed Analysis Card** — probability bar chart for all 5 classes

#### 4. History View
- Table/list of past scans per user
- Click any entry to re-view full diagnostic result

#### 5. Admin Panel
- View all registered users
- See all scan histories across users

---

## 📐 DISEASE SEVERITY ESTIMATION

A novel logic feature (not from the model, but custom-built):

```
Spread % = Base_Factor[Disease] × Confidence_Score
```

| Spread % | Severity Label | Action |
|----------|---------------|--------|
| 0% | Healthy | None needed |
| < 15% | Low | Monitor |
| < 40% | Medium | Treat Soon |
| < 70% | High | Urgent Treatment |
| > 70% | Critical | Immediate Action |

- Diseases like **Loose Smut** get a higher base factor (more aggressive spread)
- This is a **heuristic estimate** (not pixel-level segmentation)
- Provides actionable guidance even without image segmentation models

---

## 🗃️ DATABASE DESIGN

### Table 1: `users`
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| username | TEXT UNIQUE | Login name |
| password | TEXT | Hashed string |
| role | TEXT | 'admin' or 'user' |

### Table 2: `predictions` (scan_history)
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | Auto-increment |
| user_id | INTEGER FK | Links to users.id |
| filename | TEXT | Uploaded image path |
| disease | TEXT | Predicted class name |
| confidence | REAL | 0.0 – 1.0 |
| spread | REAL | Calculated spread % |
| severity | TEXT | Low/Medium/High/Critical |
| timestamp | DATETIME | When scan was performed |

---

## 📋 TEST CASES

| Test ID | Test Case | Expected Result | Status |
|---------|-----------|----------------|--------|
| TC-001 | User Registration | Account created, redirect | ✅ Pass |
| TC-002 | Duplicate Registration | Error shown | ✅ Pass |
| TC-003 | Valid Login | User authenticated, dashboard shown | ✅ Pass |
| TC-004 | Invalid Password Login | Error message | ✅ Pass |
| TC-007 | Healthy Wheat Upload | "Healthy Wheat" result with high confidence | ✅ Pass |
| TC-008 | Diseased Leaf Upload | Correct disease identified | ✅ Pass |
| TC-009 | Loose Smut Upload | "Wheat Loose Smut" detected | ✅ Pass |
| TC-011 | End-to-End Workflow | Register → Login → Upload → Get Result | ✅ Pass |

---

## ⚙️ HARDWARE & SOFTWARE REQUIREMENTS

### Development:
- CPU: Intel Core i5+ (i7 recommended for training)
- RAM: 16 GB (minimum 8 GB)
- GPU: NVIDIA GTX/RTX (for training, optional for inference)
- Storage: 500 GB SSD
- OS: Windows 10/11 or Ubuntu Linux

### Deployment (Minimum):
- vCPU: 2 cores
- RAM: 4 GB
- Storage: 10 GB (for DB + uploads)
- OS: Linux/Windows + Python 3.9+

### Python Libraries:
```
flask            - Web server
tensorflow       - AI model inference
numpy            - Array operations
pillow           - Image handling
werkzeug         - Security + file handling
sqlite3          - Database (built-in)
```

---

## 📈 RESULTS

| Metric | Value |
|--------|-------|
| Model Architecture | ResNet50 (50 layers) |
| Input Image Size | 224 × 224 × 3 |
| Number of Classes | 5 |
| Training Accuracy | ~93%+ |
| Inference Time (CPU) | ~200–500 ms |
| Healthy Leaf Confidence | Consistently >95% |

---

## 🔮 FUTURE SCOPE

| Enhancement | Description |
|------------|-------------|
| 📱 Mobile App | Convert to TensorFlow Lite + React Native for offline field use |
| 🚁 Drone Integration | Connect prediction API to agricultural drones for automated field scanning |
| 🌐 Multilingual Support | Hindi, Punjabi, Telugu translations (Flask-Babel) |
| 🔄 Active Learning | Users flag wrong predictions → retrain model with feedback |
| 🗺️ Disease Heatmap | Map disease hotspots on a GPS field layout |

---

## 🧩 KEY CHALLENGES & SOLUTIONS

| Challenge | Solution |
|-----------|----------|
| Keras version mismatch (batch_shape error) | Custom model architecture rebuilt in app.py using `build_model()` — weights loaded separately |
| Class imbalance (few Healthy samples) | Focal Loss + Balanced class weights + 1.5× boost for minority class |
| Overfitting on small dataset | Heavy data augmentation (rotation, zoom, brightness, flips) |
| Ambiguous images (poor lighting) | Model returns all 5 probabilities — top-2 usually contains correct answer |
| Admin vs. User role security | New registrations forcibly assigned `user` role; role stored server-side |

---

## 📚 REFERENCES

1. He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
2. Mohanty, S. P., et al. (2016). *Using Deep Learning for Image-Based Plant Disease Detection.* Frontiers in Plant Science.
3. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks.* arXiv:1409.1556.
4. TensorFlow Documentation (2024). `tf.keras.applications.resnet50`
5. Flask Documentation (2024). *User Guide: Sessions and Security.*
6. Singh, D., et al. (2020). *Deep Learning for Plant Stress Identification.* Indian Journal of Agricultural Sciences.

---

*© 2024 Wheat Vision AI — Wheat Disease Detection System | Built with ResNet50 + Flask*
