# Healing with Data: Predicting Heart Disease Using Random Forest

According to the Centers for Disease Control and Prevention (CDC) one person dies every 33 seconds from cardiovascular disease in the United States. The ability to predict heart disease early can lead to timely interventions, saving lives and reducing healthcare costs. In this blog, we'll explore how we can use a machine learning algorithm—Random Forest—to predict the likelihood of heart disease based on various patient features. We'll also cover key model evaluation metrics like precision, recall, F-1 score and how to interpret them.

## What is Random Forest?

Random Forest is a machine learning method that combines many decision trees. Each tree is trained on a random part of the data, and the final prediction is made by averaging their results or taking a majority vote. It works like a group of doctors, each contributing their own knowledge to make an accurate heart disease diagnosis. Instead of relying on one doctor's opinion, we have a team of doctors (the trees in Random Forest), each with their own area of expertise. Each doctor looks at the patient's health from a different angle and gives their own diagnosis. The team votes on the diagnosis. If most doctors agree, that's the final decision. This makes the decision more reliable as it reduces the chance of a single doctor making an incorrect diagnosis. 

## Why choose Random Forest?

1. **Capturing Complex Relationships**: While Logistic Regression assumes a simple linear relationship between predictors and the target variable, Random Forest can model non-linear relationships better.
2. **Variable Importance**: If we have a large number of features, Random Forest automatically identifies and prioritizes important features, improving predictions.
3. **Feature Interactions**: Random Forest excels at detecting interactions between features (e.g., age and cholesterol levels).
4. **Overfitting and Bias**: Random Forest reduces overfitting by combining multiple decision trees, each looking at different parts of the data making it more robust and capable of generalizing better, especially in complex datasets. This approach improves accuracy and reduces the risk of overfitting. 

## The Heart Disease Dataset

To demonstrate how random forest works, we'll use the Heart Disease UCI dataset, which includes 303 records of features like age, sex, chest pain type, blood pressure, cholesterol levels, and heart rate, among others, related to a patient's medical history. The target variable indicates whether the patient has heart disease, with class 1 for "disease present" and class 0 for "no disease."

## Data Preprocessing

Before training the model, we need to preprocess the data. This involves handling any missing values, encoding categorical variables, and standardizing the data. 

## Building the Random Forest Model

Once the data is pre-processed, we can build and train the Random Forest model using the scikit-learn library in Python. We'll train the model using the training dataset and evaluate its performance on the testing dataset.

## Model Evaluation:

The table below shows the key performance metrics for our Random Forest model.

<div align="center">

| Metric | Random Forest |
|--------|--------------|
| Accuracy | 0.82 |
| Precision (Class 0) | 0.88 |
| Precision (Class 1) | 0.78 |
| Recall (Class 0) | 0.77 |
| Recall (Class 1) | 0.88 |
| F1-Score (Class 0) | 0.82 |
| F1-Score (Class 1) | 0.83 |

</div>


- **Precision**: It tells us how often the model is correct when it says someone has the disease. It's about avoiding false alarms. For Class 1 (disease), precision is 78%, meaning 78% of the predictions for disease are correct.
- **Recall**: It is the model's ability to catch everyone with the disease. For Class 1, recall is 88%, meaning the model correctly identified 88% of the people with the disease.
- **F1-Score**: The F1-score balances precision and recall. It's like asking, how good is the model at both finding cases and avoiding false alarms. It combines the two into one score to give an overall picture of the model's performance. The balanced F1-scores (82% for Class 0 and 83% for Class 1) shows the model works well for both groups.

We can also look at feature importance in our Random Forest model in the image below:

<p align="center">
  <img src="/images/feature_imp_blog2.png" alt="Feature Importance for Heart Disease Prediction">
</p>

The feature importance scores show how much each factor contributes to predicting heart disease. Chest pain type (cp), maximum heart rate achieved (thalach), and ST depression during exercise are the most important, meaning these clinical indicators strongly influence the model's predictions. Age and cholesterol levels are moderately important, while factors like gender and fasting blood sugar have less impact. This suggests the model relies heavily on detailed medical test results rather than general health or demographic factors.

## Conclusion

In conclusion, we explored how Random Forest, a powerful machine learning algorithm, can be used to predict heart disease based on patient health data. With an accuracy of 82%, the model demonstrates its potential to support disease diagnosis by effectively identifying individuals at risk. It balances precision and recall across both healthy and disease categories, making it a reliable tool for aiding early diagnosis.

For those interested in diving deeper into the code and results, you can find the complete implementation on [my GitHub repository](https://github.com/msmruthi/healing-with-data/blob/main/Heart_Disease_Diagnosis_Random_Forest.ipynb)
## Dataset Source:

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.
