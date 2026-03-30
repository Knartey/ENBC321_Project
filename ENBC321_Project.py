import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
df = pd.read_csv("ai_impact_student_performance_dataset.csv")

#Data Preprocessing:
data = df.copy()
x = data.drop('final_score', axis=1)
y = data['final_score']
#Fill NA values with the mean
data = data.fillna(data.median(numeric_only=True, skipna=True))

#Dummies for gender column
data = pd.get_dummies(data, columns=['gender', 'grade_level', 'ai_tools_used', 'ai_usage_purpose', 'performance_category'], drop_first=True, dtype=int)

#Feature selection using backward elimination
model = RandomForestClassifier(random_state=42)
backward_elim = SequentialFeatureSelector(model, n_feature_to_select='auto', tol=0.0, direction='backward', cv=5)
backward_elim.fit(x, y)
#Machine Learning Techniques (Supervised):

#Linear Regression

#Logistic Regression

#Support Vector Machines (SVM)

#Decision Trees

#Random Forests

#k-Nearest Neighbors (k-NN)

#Naive Bayes

#Machine Learning Techniques (Unsupervised):

#k-Means Clustering

#Hierarchical Clustering

#Prinicpal Component Analysis (PCA)

#Deep Learning:
