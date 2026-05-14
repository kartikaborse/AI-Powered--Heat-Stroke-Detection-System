# AI-Powered Heat Stroke Detection System

## Overview

The **AI-Powered Heat Stroke Detection System** is an intelligent health monitoring solution designed to detect early signs of heat stroke and heat-related illnesses using machine learning and real-time physiological/environmental data.

This project aims to improve safety by continuously analyzing health indicators and generating alerts when dangerous conditions are detected.

---

# Features

* Real-time heat stroke risk detection
* AI/ML-based prediction system
* Environmental monitoring support
* Physiological data analysis
* Emergency alert system
* User-friendly dashboard/interface
* Data visualization and analytics
* Lightweight and scalable architecture

---

# Problem Statement

Heat stroke is a serious medical condition caused by prolonged exposure to high temperatures and dehydration. Delayed detection can lead to severe complications or even death.

This system helps identify early warning signs using artificial intelligence techniques and sensor data, enabling preventive action before the condition becomes critical.

---

# Technologies Used

## Programming Languages

* Python
* JavaScript (optional frontend)
* HTML/CSS (optional frontend)

## Machine Learning & AI

* TensorFlow / PyTorch
* Scikit-learn
* Pandas
* NumPy

## Backend

* Flask / FastAPI / Django

## Frontend

* React / HTML-CSS-JS

## Database

* MySQL / MongoDB / Firebase

## Hardware & Sensors (Optional)

* Temperature Sensor
* Heart Rate Sensor
* Humidity Sensor
* Wearable Devices

---

# System Architecture

1. Data Collection
2. Data Preprocessing
3. Feature Extraction
4. AI Model Prediction
5. Risk Classification
6. Alert & Notification System
7. Dashboard Monitoring

---

# How It Works

The system collects real-time environmental and physiological data such as:

* Body temperature
* Environmental temperature
* Humidity levels
* Hydration indicators

The AI model processes this information and predicts the risk level:

* No Risk
* Low Risk
* Moderate Risk
* High Risk

If abnormal conditions are detected, the system immediately sends alerts to users, caregivers, or medical staff.

# Machine Learning Model

The prediction model can be trained using historical health and environmental datasets.

## Machine Learning Algorithm Used

The system uses the **Random Forest Algorithm** for heat stroke risk prediction.

Random Forest was selected because it achieved higher accuracy and better prediction performance compared to other evaluated models such as:

* Logistic Regression
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

The model performs classification based on environmental and physiological parameters to determine the user's heat stroke risk level.

## Model Workflow

1. Data preprocessing
2. Feature scaling
3. Model training
4. Validation & testing
5. Deployment

---

# Dataset

You can use:

* Public healthcare datasets
* Sensor-generated datasets
* Custom-collected wearable data

Example features:

| Feature             | Description                     |
| ------------------- | ------------------------------- |
| Body Temperature    | Core body temperature           |
| Humidity            | Environmental humidity          |
| Ambient Temperature | Surrounding temperature         |


---

# Future Improvements

* Mobile application integration
* Smartwatch support
* Cloud deployment
* Real-time IoT integration
* Advanced deep learning models
* GPS-based emergency response
* Voice assistant support

---

# Screenshots

Add project screenshots here.

```bash
<img width="1600" height="718" alt="sensors " src="https://github.com/user-attachments/assets/188fb919-2948-45d5-b14b-0c9813f75a25" />
<img width="856" height="1448" alt="pic1" src="https://github.com/user-attachments/assets/b2c6bf7c-dc10-4c15-a5cd-486b7e169986" />
<img width="1036" height="616" alt="pic2" src="https://github.com/user-attachments/assets/56df8b4c-c9b4-433c-b760-7719f5425997" />
<img width="1032" height="592" alt="pic3" src="https://github.com/user-attachments/assets/0427c2d8-b4a8-4ff0-8465-db9794051dda" />

```

---

# API Endpoints (Optional)

| Method | Endpoint | Description              |
| ------ | -------- | ------------------------ |
| GET    | /status  | Check API status         |
| POST   | /predict | Predict heat stroke risk |
| GET    | /reports | Fetch reports            |

---

# Example Prediction Request

```json
{
  "body_temperature": 40.1,
  "heart_rate": 128,
  "humidity": 82,
  "environment_temperature": 43
}
```

# Example Response

```json
{
  "risk_level": "High Risk",
  "recommendation": "Immediate cooling and hydration required"
}
```

---

# Use Cases

* Industrial worker monitoring
* Sports and athletics
* Elderly care
* Military personnel monitoring
* Outdoor workforce safety
* Smart healthcare systems

---

# Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

---

# License

This project is licensed under the MIT License.

---

# Author

## Your Name

* GitHub: [https://github.com/your-username](https://github.com/kartikaborse)
* LinkedIn: [https://linkedin.com/in/your-profile](https://linkedin.com/in/kartika-borse

---

# Acknowledgements

* Open-source AI community
* Healthcare datasets providers
* IoT and wearable technology contributors
* Research papers on heat stroke prediction
