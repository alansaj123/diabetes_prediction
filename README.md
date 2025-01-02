# Diabetes Prediction using K-Nearest Neighbors (KNN)

## Project Overview
This project aims to predict the likelihood of diabetes based on various health-related features using the **K-Nearest Neighbors (KNN)** machine learning algorithm. The dataset used contains information such as **Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function**, and **Age**, with the target variable being the presence or absence of diabetes.

## Key Features
- **Exploratory Data Analysis (EDA)**: In-depth analysis of the dataset to identify patterns and handle any data preprocessing needs like missing values and class imbalance.
- **Modeling**: Building a KNN classifier and evaluating its performance using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC curve.
- **SMOTE**: Addressing class imbalance using **Synthetic Minority Over-sampling Technique (SMOTE)** to improve model performance.

## Approach
1. **Data Preprocessing**:
   - The dataset was cleaned and preprocessed to handle missing values, outliers, and imbalanced classes.
   - Feature scaling was done using **StandardScaler** to standardize the features.
   
2. **KNN Classifier**:
   - The KNN classifier was trained on the data and evaluated on test data with different values of `k` (number of neighbors).
   - The model's performance was assessed using a **Confusion Matrix** and **Classification Report**.

3. **Evaluation**:
   - The final model's performance was assessed using a **ROC-AUC curve** to analyze its classification capabilities, especially for imbalanced datasets.
   - The **Classification Report** provided precision, recall, and F1-score metrics for each class.
     
## Future improvements may include:

- Hyperparameter tuning using techniques like GridSearchCV.
- Exploring other algorithms like Random Forest or Support Vector Machine for better performance.
- Implementing more advanced techniques for handling data imbalance.

## Dependencies
To run this project, you will need the following Python packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `imblearn`

Install the necessary packages with:
```bash
pip install -r requirements.txt

