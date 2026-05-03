# Code Snippets for Project Report

## 1. Ensemble Prediction Logic (Backend)
This Python snippet from `app.py` demonstrates how the system combines predictions from two Deep Learning models (ResNet50 and EfficientNet) to achieve higher accuracy.

```python
# --- PREDICTION LOGIC ---
ensemble_predictions = None

# 1. Get ResNet Prediction
img_resnet = prepare_image_resnet(filepath)
pred_resnet = model_resnet.predict(img_resnet, verbose=0)[0]

# 2. Get EfficientNet Prediction (if available)
if use_ensemble and model_efficientnet is not None:
    img_eff = prepare_image_efficientnet(filepath)
    pred_eff = model_efficientnet.predict(img_eff, verbose=0)[0]
    
    # 3. Ensemble Average (Weighted if needed, here 50/50)
    ensemble_predictions = (pred_resnet + pred_eff) / 2.0
else:
    ensemble_predictions = pred_resnet

# Process Results
predicted_class_index = int(np.argmax(ensemble_predictions))
confidence = float(ensemble_predictions[predicted_class_index])
```

## 2. Dynamic Disease Severity Estimation
This function calculates the severity of the disease based on the model's confidence score and the specific type of disease detected.

```python
def estimate_disease_spread(disease_class, confidence):
    if 'healthy' in disease_class.lower():
        return 0.0
    
    spread_base = {
        'Leaf Rust': 35.5,
        'Mildew': 28.3,
        'Wheat Loose Smut': 45.8,
        'Yellow Rust': 32.7
    }.get(disease_class, 25.0)
    
    spread = spread_base * confidence
    return min(95.0, max(5.0, spread))
```

## 3. Login Interface Structure (Frontend)
A snippet of the HTML structure used for the Login page, featuring Bootstrap classes for responsive design and custom styling hooks.

```html
<form method="POST" action="{{ url_for('login') }}">
    <!-- Username Field -->
    <div class="mb-4">
        <label for="username" class="form-label visually-hidden">Username</label>
        <div class="input-icon-wrapper">
            <span class="input-group-text"><i class="bi bi-person-fill"></i></span>
            <div class="form-floating flex-grow-1">
                <input type="text" class="form-control" id="username" name="username" placeholder="Username" required>
                <label for="username">Username</label>
            </div>
        </div>
    </div>

    <!-- Password Field -->
    <div class="mb-4">
        <label for="password" class="form-label visually-hidden">Password</label>
        <div class="input-icon-wrapper">
            <span class="input-group-text"><i class="bi bi-lock-fill"></i></span>
            <div class="form-floating flex-grow-1">
                <input type="password" class="form-control" id="password" name="password" placeholder="Password" required>
                <label for="password">Password</label>
            </div>
        </div>
    </div>

    <div class="d-grid gap-2">
        <button type="submit" class="btn btn-primary shadow-sm">
            Sign In <i class="bi bi-arrow-right-short"></i>
        </button>
    </div>
</form>
```

## 4. Database Schema (SQLite)
The SQL commands used to initialize the database tables for Users and Prediction History.

```python
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        full_name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')

cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        disease TEXT,
        confidence REAL,
        spread REAL,
        severity TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
''')
```
