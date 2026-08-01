# Egyptian Used Car Price Prediction Using Machine Learning

> **A complete end-to-end Machine Learning project for predicting used car prices in Egypt using real-world data collected from Hatla2ee.**

---

## Project Overview

The Egyptian used car market is highly dynamic, with vehicle prices influenced by multiple factors such as manufacturer, model, mileage, transmission type, location, and available features.

The objective of this project is to build an accurate machine learning model capable of estimating the market value of a used vehicle using historical listing data.

The project covers the entire Machine Learning lifecycle:

* Data Collection
* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Training
* Hyperparameter Tuning
* Model Evaluation
* Model Comparison
* Cross Validation
* Deployment using Streamlit

---

# Demo

### Streamlit Application

The project includes an interactive Streamlit application where users can estimate the market value of a used vehicle by entering its specifications.

The application allows users to:

* Select the vehicle company
* Select the vehicle model
* Enter mileage
* Choose transmission type
* Select vehicle color
* Choose location
* Select available features
* Receive an estimated market value
* ## 🌐 Live Demo

🚀 **Try the application here:**

**https://usedcarpriceprediction-sdjdld72bg8vc4u8wdnpva.streamlit.app/**

---



---

# Dataset

Source:

**Hatla2ee Egyptian Used Cars Dataset**

Approximately **22,600** used car listings were collected and processed.

---

## Dataset Features

| Feature         | Description                  |
| --------------- | ---------------------------- |
| Company         | Vehicle manufacturer         |
| Model           | Vehicle model                |
| Year            | Manufacturing year           |
| Mileage         | Total kilometers driven      |
| Color           | Vehicle color                |
| Transmission    | Automatic / Manual / Unknown |
| Location        | Egyptian governorate or city |
| Date Posted     | Listing publication date     |
| Air Conditioner | Binary feature               |
| Power Steering  | Binary feature               |
| Remote Control  | Binary feature               |
| Price           | Target variable              |

---

# Machine Learning Pipeline

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Preprocessing
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Cross Validation
      │
      ▼
Streamlit Deployment
```

---

# Data Cleaning

The following preprocessing operations were performed:

* Removed duplicate listings
* Removed unnecessary columns
* Converted price to numeric format
* Converted mileage to numeric values
* Converted posting dates into datetime objects
* Corrected transmission extraction
* Handled missing transmission values
* Removed invalid records
* Standardized categorical variables

---

# Exploratory Data Analysis

EDA was performed to better understand the dataset.

Visualizations include:

* Price Distribution
* Log Price Distribution
* Mileage Distribution
* Company Distribution
* Model Distribution
* Color Distribution
* Correlation Heatmap
* Missing Values Analysis

---

# Feature Engineering

Several new features were created to improve prediction accuracy.

## Car Age

```text
Car Age = Current Year − Manufacturing Year
```

---

## Binary Features

The vehicle description was parsed to generate binary indicators.

Examples:

* Air Conditioner
* Power Steering
* Remote Control

---

## Transmission Correction

Transmission information was reconstructed from listing descriptions whenever possible.

Remaining missing values were labeled as **Unknown**.

---

# Final Features Used

The final model uses the following input features:

* Company
* Model
* Mileage
* Color
* Transmission
* Location
* Car Age
* Air Conditioner
* Automatic
* Power Steering
* Remote Control

---

# Models Implemented

The following regression algorithms were trained and evaluated.

## 1. Linear Regression

Baseline regression model.

---

## 2. Decision Tree Regressor

Tree-based regression algorithm.

---

## 3. Random Forest Regressor

Ensemble model using multiple decision trees.

---

## 4. XGBoost Regressor

Gradient Boosting model optimized for structured tabular datasets.

---

## 5. Tuned XGBoost ⭐

Hyperparameter optimization using RandomizedSearchCV.

This model achieved the best overall performance.

---

## 6. CatBoost Regressor

Gradient boosting model specialized for categorical variables.

---

# Model Performance

| Model               |   MAE (EGP) |  RMSE (EGP) |   R² Score |
| ------------------- | ----------: | ----------: | ---------: |
| Linear Regression   |     344,501 |     825,815 |     0.2357 |
| Decision Tree       |     175,015 |     569,046 |     0.6371 |
| Random Forest       |     130,676 |     467,833 |     0.7547 |
| CatBoost            |     129,672 |     452,269 |     0.7708 |
| XGBoost             |     119,623 |     399,401 |     0.8212 |
| **Tuned XGBoost** ⭐ | **119,246** | **385,681** | **0.8333** |

---

# Cross Validation

To evaluate model robustness, **5-Fold Cross Validation** was performed.

Results:

```
Fold R² Scores

0.9276
0.9092
0.9181
0.9155
0.9219
```

Average R²

```
0.9185
```

Standard Deviation

```
0.0062
```

These results indicate that the model generalizes well across different subsets of the dataset.

---

# Hyperparameter Tuning

The XGBoost model was optimized using **RandomizedSearchCV**.

Optimized parameters include:

* Number of estimators
* Learning rate
* Maximum depth
* Subsample ratio
* Column sampling
* Gamma
* Minimum child weight

Best model:

```
Tuned XGBoost Regressor
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* XGBoost
* CatBoost
* Joblib
* Streamlit

---

# Project Structure

```text
Egyptian_Used_Car_Price_Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── used_cars/
│       └── hatla2ee_cars_august_2025.csv
│
├── models/
│   ├── best_xgboost.pkl
│   ├── preprocessor.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── used_cars.ipynb
│
└── figures/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Egyptian_Used_Car_Price_Prediction.git
```

Move into the project

```bash
cd Egyptian_Used_Car_Price_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The application will open automatically in your default web browser.

---

# How to Use

1. Select the vehicle company.
2. Choose the corresponding model.
3. Enter the manufacturing year.
4. Enter the vehicle mileage.
5. Select the color.
6. Select the transmission type.
7. Choose the location.
8. Select available vehicle features.
9. Click **Estimate Market Value**.
10. View the predicted market price.

---

# Future Improvements

Potential future enhancements include:

* Add engine size.
* Add fuel type.
* Add vehicle body type.
* Add trim level.
* Integrate real-time market data.
* Deploy the application to Streamlit Cloud or Render.
* Develop a REST API using FastAPI.
* Add vehicle image support based on selected company and model.

---

# Results

The **Tuned XGBoost Regressor** was selected as the final production model due to its superior predictive performance.

### Final Test Performance

| Metric   |           Value |
| -------- | --------------: |
| MAE      | **119,246 EGP** |
| RMSE     | **385,681 EGP** |
| R² Score |      **0.8333** |

---

# Author

**Nour Allah**

Mechatronics & Artificial Intelligence Student

Machine Learning Project – Egyptian Used Car Price Prediction

---

# License

This project is intended for educational and research purposes.
