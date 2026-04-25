import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =====================================
# تحميل البيانات (مسار احترافي ثابت)
# =====================================
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "archive 0", "medical_insurance.csv")
    return pd.read_csv(file_path)

data = load_data()

st.title("Medical Insurance Charges Prediction")
st.write("Enter customer data to predict insurance charges")

# =====================================
# تدريب الموديل مرة واحدة فقط
# =====================================
@st.cache_resource
def train_model(data):

    X = data.drop("charges", axis=1)
    y = data["charges"]

    # تحويل البيانات النصية لأرقام
    X = pd.get_dummies(X, columns=['sex', 'smoker', 'region'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, X_train.columns

model, feature_columns = train_model(data)

# =====================================
# إدخال بيانات المستخدم
# =====================================
st.subheader("Predict New Customer Charges")

age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Sex", ['female', 'male'])
bmi = st.slider("BMI", 15.0, 55.0, 25.0)
children = st.slider("Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ['yes', 'no'])
region = st.selectbox("Region", ['southwest','southeast','northwest','northeast'])

# إنشاء DataFrame من الإدخال
new_data = pd.DataFrame({
    'age':[age],
    'sex':[sex],
    'bmi':[bmi],
    'children':[children],
    'smoker':[smoker],
    'region':[region]
})

# One Hot Encoding
new_data = pd.get_dummies(new_data, drop_first=True)

# توحيد الأعمدة مع التدريب
for col in feature_columns:
    if col not in new_data.columns:
        new_data[col] = 0

new_data = new_data[feature_columns]

# =====================================
# زر التنبؤ
# =====================================
if st.button("Predict Charges"):
    prediction = model.predict(new_data)[0]
    st.success(f"Predicted Insurance Charges: ${prediction:,.2f}")