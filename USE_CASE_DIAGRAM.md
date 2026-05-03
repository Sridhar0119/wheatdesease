# System Design: Use Case Diagram

The following diagram illustrates the interactions between the **User** (Farmer/Expert) and the **Wheat Disease Detection System**.

```mermaid
usecaseDiagram
    actor "User (Farmer/Expert)" as User
    
    package "Wheat Disease Detection System" {
        usecase "Register Account" as UC1
        usecase "Login" as UC2
        usecase "Upload Leaf Image" as UC3
        usecase "View Analysis Report" as UC4
        usecase "View History" as UC5
        usecase "Logout" as UC6
        
        usecase "Validate File" as UC7
        usecase "Predict Disease (ResNet50)" as UC8
        usecase "Estimate Severity" as UC9
        usecase "Get Recommendations" as UC10
        usecase "Save to Database" as UC11
    }

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC5
    User --> UC6

    %% Relationships
    UC3 ..> UC4 : <<triggers>>
    UC3 ..> UC7 : <<include>>
    UC3 ..> UC8 : <<include>>
    UC8 ..> UC9 : <<include>>
    UC8 ..> UC10 : <<include>>
    UC8 ..> UC11 : <<include>>
```

### Description of Actors and Use Cases

*   **User**: The primary actor who interacts with the system to identify wheat diseases.
*   **Register/Login**: Ensures secure access to the system.
*   **Upload Leaf Image**: The core functionality where the user provides input data.
*   **Predict Disease**: The internal process where the ResNet50 model analyzes the image.
*   **View Analysis Report**: The output presented to the user, containing the prediction, confidence, severity, and medical advice.
*   **View History**: Allows the user to track previous analyses.
