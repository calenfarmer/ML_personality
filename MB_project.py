"""
Improting the data file, parsing the data, and plotting data. 
Graph depicts the number of words used in social media posts for each personality type.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

path = '/Users/calenfarmer/Desktop/Machine Learning/Project/My Project/ML_personality/' # insert your path here. 
filename = 'mbti_1 4.csv'
data = pd.read_csv(path + filename) 

def cleanUP(text):
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', '', text)
    text = text.replace("|||"," ")
    text=text.lower()
    text = re.sub(r"http\S+", "", text, flags=re.MULTILINE)
    return text

# Parse data and remove text that is not useful. 
data['posts'] = data['posts'].apply(cleanUP)
data.head()
num_words = data['type'].value_counts()
num_words = num_words.to_dict()

types = list(num_words.keys())
count = list(num_words.values())

plt.bar(types, count, color='green', width=0.1)
plt.xlabel('Personality Types')
plt.ylabel('Number of Words')
plt.title('Number of Words per Personality Type')
plt.show()

"""
Processing and ML training/accuracy
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Dividing the 16 personalities into 4 major groups. Common practice in MyersBriggs results.
map_IE = {"I": 0, "E": 1}
map_NS = {"N": 0, "S": 1}
map_TF = {"T": 0, "F": 1}
map_JP = {"J": 0, "P": 1}

data['IE'] = data['type'].astype(str).str[0]
data['IE'] = data['IE'].map(map_IE)
data['NS'] = data['type'].astype(str).str[1]
data['NS'] = data['NS'].map(map_NS)
data['TF'] = data['type'].astype(str).str[2]
data['TF'] = data['TF'].map(map_TF)
data['JP'] = data['type'].astype(str).str[3]
data['JP'] = data['JP'].map(map_JP)

data['http_per_comment'] = data['posts'].apply(lambda x: x.count('http')/50)
data['music_per_comment'] = data['posts'].apply(lambda x: x.count('music')/50)
data['question_per_comment'] = data['posts'].apply(lambda x: x.count('?')/50)
data['img_per_comment'] = data['posts'].apply(lambda x: x.count('jpg')/50)
data['excl_per_comment'] = data['posts'].apply(lambda x: x.count('!')/50)
data['ellipsis_per_comment'] = data['posts'].apply(lambda x: x.count('...')/50)

#Building ML on 'type' column 
X = data.drop(['type','posts','IE','NS','TF','JP'], axis=1).values
y = data['type'].values

print("y shape: ", y.shape)
print("x shape: ", X.shape)
print("\n")

# Split arrays or matrices into random train and test subsets.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=5)

# *****************************************************************************************************
# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

Y_prediction = rf.predict(X_test)

rf.score(X_train, y_train)
acc_rf = round(rf.score(X_train, y_train) * 100, 2)
print("Training Data Set",round(acc_rf,2,), "%")

acc_rf = round(rf.score(X_test, y_test) * 100, 2)
print("Testing Data Set", round(acc_rf,2,), "%")

print("\n")

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
acc_log = round(lr.score(X_train, y_train) * 100, 2)
print(round(acc_log,2,), "%")

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print(round(acc_knn,2,), "%")

print("\n")

# *****************************************************************************************************
# introversion/extroversion 
X_IE = data.drop(['type','posts','IE'], axis=1).values
y_IE = data['IE'].values

X_IE_train,X_IE_test,y_IE_train,y_IE_test=train_test_split(X_IE,y_IE,test_size = 0.1, random_state=5)
print("Introversion/Extroversion Personality Types")

rf_IE = RandomForestClassifier(n_estimators=100)
rf_IE.fit(X_IE_train, y_IE_train)

rf_IE.score(X_IE_train, y_IE_train)
acc_rf_IE = round(rf_IE.score(X_IE_train, y_IE_train) * 100, 2)
print("Random Forest Predictions Model",round(acc_rf_IE,2,), "%")

# Logistic Regression
lr_IE = LogisticRegression()
lr_IE.fit(X_IE_train, y_IE_train)

acc_logg = round(lr_IE.score(X_IE_train, y_IE_train) * 100, 2)
print("Logisitic Regression Prediction Accuracy",round(acc_logg,2,), "%")

# KNN
knn_IE = KNeighborsClassifier(n_neighbors = 3)
knn_IE.fit(X_IE_train, y_IE_train)

acc_knn_IE = round(knn_IE.score(X_IE_train, y_IE_train) * 100, 2)
print("Knearnest neighbor prediction value",round(acc_knn_IE,2,), "%")
print("\n")

# *****************************************************************************************************
# Intuitive/sensing 
X_IS = data.drop(['type','posts','NS'], axis=1).values
y_IS = data['NS'].values

X_IS_train,X_IS_test,y_IS_train,y_IS_test=train_test_split(X_IS,y_IS,test_size = 0.1, random_state=5)
print("Intuitive/Sensing Personality Types")

rf_IS = RandomForestClassifier(n_estimators=100)
rf_IS.fit(X_IS_train, y_IS_train)

rf_IS.score(X_IS_train, y_IS_train)
acc_rf_IS = round(rf_IS.score(X_IS_train, y_IS_train) * 100, 2)
print("Random Forest Predictions Model",round(acc_rf_IS,2,), "%")

# Logistic Regression
lr_IS = LogisticRegression()
lr_IS.fit(X_IS_train, y_IS_train)

Y_predd = lr_IS.predict(X_IS_test)

acc_logg = round(lr_IS.score(X_IS_train, y_IS_train) * 100, 2)
print("Logisitic Regression Prediction Accuracy",round(acc_logg,2,), "%")

# KNN
knn_IS = KNeighborsClassifier(n_neighbors = 3)
knn_IS.fit(X_IS_train, y_IS_train)

acc_knn_IS = round(knn_IS.score(X_IS_train, y_IS_train) * 100, 2)
print("Knearnest neighbor prediction value",round(acc_knn_IS,2,), "%")
print("\n")

# *****************************************************************************************************
# Thinking/Feeling 
X_TF = data.drop(['type','posts','TF'], axis=1).values
y_TF = data['TF'].values

X_TF_train,X_TF_test,y_TF_train,y_TF_test=train_test_split(X_TF,y_TF,test_size = 0.1, random_state=5)
print("Thinking/Feeling Personality Types")

rf_TF = RandomForestClassifier(n_estimators=100)
rf_TF.fit(X_TF_train, y_TF_train)

rf_TF.score(X_TF_train, y_TF_train)
acc_rf_TF = round(rf_TF.score(X_TF_train, y_TF_train) * 100, 2)
print("Random Forest Predictions Model",round(acc_rf_TF,2,), "%")

# Logistic Regression
lr_TF = LogisticRegression()
lr_TF.fit(X_TF_train, y_TF_train)

acc_logg = round(lr_TF.score(X_TF_train, y_TF_train) * 100, 2)
print("Logisitic Regression Prediction Accuracy",round(acc_logg,2,), "%")

# KNN
knn_TF = KNeighborsClassifier(n_neighbors = 3)
knn_TF.fit(X_TF_train, y_TF_train)

acc_knn_TF = round(knn_TF.score(X_TF_train, y_TF_train) * 100, 2)
print("Knearnest neighbor prediction value",round(acc_knn_TF,2,), "%")
print("\n")

# *****************************************************************************************************
# Judging/perceiving
X_JP = data.drop(['type','posts','JP'], axis=1).values
y_JP = data['JP'].values

X_JP_train,X_JP_test,y_JP_train,y_JP_test=train_test_split(X_JP,y_JP,test_size = 0.1, random_state=5)
print("Judging/Perceiving Personality Types")

rf_JP = RandomForestClassifier(n_estimators=100)
rf_JP.fit(X_JP_train, y_JP_train)

rf_JP.score(X_JP_train, y_JP_train)
acc_rf_JP = round(rf_JP.score(X_JP_train, y_JP_train) * 100, 2)
print("Random Forest Predictions Model",round(acc_rf_JP,2,), "%")

# Logistic Regression
lr_JP = LogisticRegression()
lr_JP.fit(X_JP_train, y_JP_train)

acc_logg = round(lr_JP.score(X_JP_train, y_JP_train) * 100, 2)
print("Logisitic Regression Prediction Accuracy",round(acc_logg,2,), "%")

# KNN
knn_JP = KNeighborsClassifier(n_neighbors = 3)
knn_JP.fit(X_JP_train, y_JP_train)

acc_knn_JP = round(knn_JP.score(X_JP_train, y_JP_train) * 100, 2)
print("Knearnest neighbor prediction value",round(acc_knn_JP,2,), "%")
print("\n")

# *****************************************************************************************************

rf_final = {}
rf_final['IE'] = rf_IE
rf_final['IS'] = rf_IS
rf_final['TF'] = rf_TF
rf_final['JP'] = rf_JP

lr_final = {}
lr_final['IE'] = lr_IE
lr_final['IS'] = lr_IS
lr_final['TF'] = lr_TF
lr_final['JP'] = lr_JP

knn_final = {}
knn_final['IE'] = knn_IE
knn_final['IS'] = knn_IS
knn_final['TF'] = knn_TF
knn_final['JP'] = knn_JP
