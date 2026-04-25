# Medical-Insurance-Charges-Prediction
A machine learning web app that predicts medical insurance charges using features like age, BMI, smoking status, children, gender, and region. It uses a Random Forest Regression model and is deployed with Streamlit for real-time predictions.

---

## Project Overview

Medical insurance companies estimate charges based on several risk factors such as age, BMI, smoking habits, and region.
This project uses a Random Forest Regression model to learn from historical data and predict future insurance costs.
The final model is deployed as a Streamlit web app to make predictions easily accessible to users.

---

## Project Objectives

* Build a regression model to predict insurance charges
* Perform data preprocessing and feature engineering
* Train and optimize a Random Forest model
* Deploy an interactive web application using Streamlit
* Provide real-time predictions for new customers

---

## Dataset Features

| Feature  | Description                     |
| -------- | ------------------------------- |
| age      | Age of the customer             |
| sex      | Male / Female                   |
| bmi      | Body Mass Index                 |
| children | Number of dependents            |
| smoker   | Smoking status                  |
| region   | Residential region              |
| charges  | Medical insurance cost (Target) |

---

## Machine Learning Model

**Model used:** Random Forest Regressor

**Why Random Forest?**

* Handles non-linear relationships well
* Works well with mixed feature types
* Reduces overfitting using ensemble learning
* Provides strong prediction accuracy

---

## Technologies Used

* Python
* Pandas & NumPy
* Scikit-Learn
* Streamlit
* Matplotlib / Seaborn

---

## Application Features

* User-friendly web interface
* Real-time prediction of insurance charges
* Interactive sliders and dropdown inputs
* Automatic preprocessing of user data
* Fast model loading using caching

---

## Project Structure

```
Regression/
│
├── ui_regression.py
├── medical_insurance.csv
└── README.md
```

---

## How to Run the App

### Install requirements

```bash
pip install streamlit scikit-learn pandas matplotlib seaborn
```

### Run the application

```bash
streamlit run ui_regression.py
```

### Open in browser

```
http://localhost:8501
```

---

## How It Works

1. Dataset is loaded and preprocessed
2. Categorical features are encoded using One-Hot Encoding
3. Data is split into training and testing sets
4. Random Forest model is trained and cached
5. User inputs new data via the web interface
6. Model predicts insurance charges instantly

---

## Future Improvements

* Add multiple model comparison
* Deploy the app on Streamlit Cloud
* Add feature importance visualization
* Improve UI/UX with multi-page layout

---

## Author

Machine Learning Project for Div Academy
