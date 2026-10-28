# ⚡ EV Range Prediction

A machine learning web application that estimates the real-world driving range of an electric vehicle (EV) based on vehicle specifications and selected driving-related features.

## 🚀 Live Demo

Try the deployed application:  
**[EV Range Predictor](https://ev-range-predictor-rahul.streamlit.app/)**

## 📂 GitHub Repository

[View the source code on GitHub](https://github.com/RahulYadav-75/EV_Range)

## 📌 Project Overview

Electric vehicle range can vary depending on battery capacity, motor power, vehicle weight, motor efficiency, driving speed, ambient temperature, and battery age. This project uses a regression model to estimate the vehicle's real-world range in kilometers from these inputs.

The app is built with Streamlit so users can enter vehicle details and receive a range prediction through a simple web interface.

## ✨ Features

- Predicts estimated real-world EV range in kilometers.
- Accepts vehicle make, model, and variant.
- Uses battery capacity, motor power, vehicle weight, and motor efficiency.
- Includes average driving speed, ambient temperature, road type, and battery age.
- Applies one-hot encoding to categorical inputs.
- Uses a trained machine learning regression pipeline.
- Offers an interactive Streamlit dashboard.

## 🧰 Tech Stack

- **Python**
- **Pandas** and **NumPy**
- **Scikit-learn**
- **Random Forest Regression**
- **OneHotEncoder** and **ColumnTransformer**
- **Joblib**
- **Streamlit**

## 🧾 Model Inputs

The model uses these 11 input features:

| Feature | Description |
|---|---|
| `make` | Vehicle manufacturer |
| `model` | Vehicle model |
| `variant` | Vehicle variant |
| `battery_capacity_kWh` | Battery capacity in kWh |
| `motor_power_kW` | Motor power in kW |
| `vehicle_weight_kg` | Vehicle weight in kg |
| `motor_efficiency_pct` | Motor efficiency percentage |
| `avg_driving_speed_kmh` | Average driving speed in km/h |
| `ambient_temperature_C` | Ambient temperature in °C |
| `city_road_type` | Road type/category |
| `battery_age_years` | Battery age in years |

## 📊 Model Performance

Reported evaluation results for the current 11-feature model:

| Metric | Result |
|---|---:|
| R² Score | 0.7136 (71.36%) |
| Mean Absolute Error (MAE) | 56.97 km |
| Root Mean Squared Error (RMSE) | 76.64 km |

These results are from the project's reported evaluation and may vary depending on the train/test split and model version. Predictions are estimates, not guaranteed driving ranges.

## 🖥️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/RahulYadav-75/EV_Range.git
cd EV_Range
```

### 2. Create and activate a virtual environment (optional but recommended)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Make sure the dependency file is named `requirements.txt`.

```bash
python -m pip install -r requirements.txt
```

### 4. Start the app

```bash
streamlit run app.py
```

The app should open in your browser. If it does not, open the local URL shown in the terminal.

## 📁 Project Structure

```text
EV_Range/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       └── india_ev_range_dataset.csv
├── models/
│   └── ev_range_model.pkl
└── ev_range_prediction.ipynb
```

The exact files in your repository may differ. The app requires the trained model file at the path configured in `app.py`. Include the dataset only if it is needed by your app or notebook and is suitable to publish.

## 🔮 Future Improvements

- Improve model accuracy through feature engineering and hyperparameter tuning.
- Add clearer prediction explanations and feature-importance visualizations.
- Validate inputs and provide realistic input ranges.
- Add model evaluation charts and comparisons with baseline regressors.
- Expand testing across different vehicle models and conditions.

## 👤 Author

**Rahul Yadav**

- GitHub: [RahulYadav-75](https://github.com/RahulYadav-75)

## ⚠️ Disclaimer

This project is for educational and demonstration purposes. The predicted range is an estimate based on the model and input features. Actual EV range can vary with traffic, weather, terrain, battery condition, driving style, and other factors.
