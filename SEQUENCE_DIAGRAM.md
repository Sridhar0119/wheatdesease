# System Design: Sequence Diagram

The following diagram details the sequence of interactions for the user login and the core wheat disease prediction process.

```mermaid
sequenceDiagram
    actor User
    participant Frontend as Web Interface (Browser)
    participant App as Flask Controller
    participant DB as SQLite Database
    participant Model as AI Model (ResNet50)

    Note over User, Model: Authentication Phase

    User->>Frontend: Access Login Page
    Frontend->>App: GET /login
    App-->>Frontend: Render login.html
    
    User->>Frontend: Enter Credentials
    Frontend->>App: POST /login (username, password)
    App->>DB: Query User by Username
    DB-->>App: Return User Record
    
    alt Valid Credentials
        App->>App: Verify Password Hash
        App->>App: Create Session
        App-->>Frontend: Redirect to /home
        Frontend->>App: GET /home
        App-->>Frontend: Render home.html
    else Invalid Credentials
        App-->>Frontend: Show Error Message
    end

    Note over User, Model: Disease Detection Phase

    User->>Frontend: Upload Leaf Image
    Frontend->>App: POST /predict (image file)
    
    activate App
    App->>App: Validate File Extension
    
    alt Invalid File
        App-->>Frontend: Return Error JSON
    else Valid File
        App->>App: Save Temp File
        
        App->>App: Preprocess Image (ResNet Input)
        App->>Model: Predict(image_array)
        activate Model
        Model-->>App: Return Probabilities
        deactivate Model
        
        App->>App: Determine Max Probability Class
        App->>App: Calculate Confidence & Spread
        
        opt Low Confidence Adjustment
            App->>App: Apply Threshold Logic (Default to Healthy if uncertain)
        end
        
        App->>App: Fetch Disease Info & Recommendations
        
        App->>DB: INSERT prediction record
        activate DB
        DB-->>App: Confirm Save
        deactivate DB
        
        App->>App: Cleanup Temp File
        
        App-->>Frontend: Return JSON (Diagnosis, Confidence, Meds)
    end
    deactivate App
    
    Frontend->>User: Display Analysis Report & Recommendations
```

### Process Description

1.  **Authentication**: The user must first authenticate. The system verifies credentials against the SQLite database using secure password hashing (Bcrypt).
2.  **Image Upload**: The user uploads a leaf image via the web interface.
3.  **Validation**: The server checks for valid file extensions (png, jpg, etc.).
4.  **Prediction**: 
    *   The image is preprocessed (resized, normalized).
    *   The loaded ResNet50 model analyzes the image.
    *   Post-processing logic applies confidence thresholds to reduce false positives.
5.  **Result Generation**: The system calculates disease severity and retrieves treatment advice from the predefined knowledge base.
6.  **Persistence**: The result is permanently stored in the database for history tracking.
7.  **Response**: The user receives a detailed report including the disease name, confidence score, severity level, and recommended actions.
