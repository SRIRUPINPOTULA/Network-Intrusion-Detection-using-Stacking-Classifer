# Network Intrusion Detection Using Stacking Classifier

## Introduction
Intrusion detection is critical for maintaining the security and integrity of computer networks. This project aims to develop a robust intrusion detection model using machine learning techniques. The focus is on feature selection mechanisms to enhance model performance and efficiency.

## Method Description
The methodology consists of several key steps:
1. **Data Preprocessing:**
    - Convert categorical features into numerical using One-Hot-Encoding.
    - Scale features to prevent bias due to large value discrepancies.
2. **Feature Selection:**
    - Identify and eliminate redundant or irrelevant features.
    - Utilize Univariate feature selection with ANOVA F-test.
    - Implement the SecondPercentile method to select features based on percentile of the highest scores.
    - Apply Recursive Feature Elimination (RFE) to further refine the feature subset.
3. **Model Building:**
    - Construct a decision tree model for intrusion detection.
4. **Prediction & Evaluation (Validation):**
    - Use test data to make predictions.
    - Evaluate model performance using multiple metrics such as accuracy score, recall, f-measure, and confusion matrix.
    - Perform 10-fold cross-validation to ensure robustness of the model.

## Version Check
- Check the versions of libraries used for reproducibility.

## Dataset Loading
- Load the training and test datasets.

## Exploratory Data Analysis
- View dimensions and statistical summary of the datasets.
- Examine label distribution in both training and test sets.

## Data Preprocessing
1. **Identify Categorical Features:**
    - Explore categorical features and their distributions.
    - Identify columns requiring transformation.
2. **Label Encoding:**
    - Transform categorical features into numerical values using LabelEncoder.
3. **One-Hot-Encoding:**
    - Apply One-Hot-Encoding to convert categorical features into binary features.
4. **Dataset Splitting and Label Renaming:**
    - The dataset is initially split into training and test sets.
    - Attack labels are renamed as follows:
        - 0: 'normal'
        - 1: 'DoS'
        - 2: 'Probe'
        - 3: 'R2L'
        - 4: 'U2R'

## Dataset Preprocessing
- **Dataset Splitting and Label Renaming:** The dataset is split into training and test sets, and attack labels are renamed.
- **Feature Scaling:** The data is standardized using the StandardScaler() to ensure each feature has a mean of 0 and standard deviation of 1.

## Feature Selection
- **Univariate Feature Selection using ANOVA F-test:** Selects a percentage of the highest scoring features for each attack category.
- **Recursive Feature Elimination (RFE):** Selects a fixed number of best features using RFE with a Decision Tree classifier.

## Model Evaluation
- **Stacking Ensemble of Models:** Combines multiple base models (Logistic Regression, KNN, Decision Tree, SVM, Naive Bayes) into a meta-learner using StackingClassifier.

## Code Execution
1. **Feature Scaling:** The dataset is standardized using StandardScaler().
2. **Recursive Feature Elimination (RFE):** RFE is applied to select the best features.
3. **Model Evaluation:** Cross-validation is performed on multiple models to evaluate their performance.

## Models
The evaluation involves the following models:
- Logistic Regression (lr)
- K-Nearest Neighbors (knn)
- Support Vector Machine (svm)
- Naive Bayes (bayes)
- Stacking Classifier (stacking)

## Evaluation Results
For each dataset and model combination, the following evaluation metrics are computed and displayed:
- Accuracy: Overall accuracy of the model.
- Precision: Precision score (macro-averaged).
- Recall: Recall score (micro-averaged).

 
