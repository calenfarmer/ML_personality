"""
Improting the data file, parsing the data, and plotting data. 
Graph depicts the number of words used in social media posts for each personality type.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

path = '/Users/calenfarmer/Desktop/Machine Learning/Project/My Project/' # insert your path here. 
filename = 'mbti_1 4.csv'
data = pd.read_csv(path + filename) 

def cleanText(text):
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', '', text)
    text = text.replace("|||"," ")
    text=text.lower()
    text = re.sub(r"http\S+", "", text, flags=re.MULTILINE)
    return text

# Parse data and remove text that is not useful. 
data['posts'] = data['posts'].apply(cleanText)
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

#Dividing the 16 personalities into 4 major groups. Common practice in Myers-Briggs results.
map_IE = {"I": 0, "E": 1}
map_NS = {"N": 0, "S": 1}
map_TF = {"T": 0, "F": 1}
map_JP = {"J": 0, "P": 1}

data['I-E'] = data['type'].astype(str).str[0]
data['I-E'] = data['I-E'].map(map_IE)
data['N-S'] = data['type'].astype(str).str[1]
data['N-S'] = data['N-S'].map(map_NS)
data['T-F'] = data['type'].astype(str).str[2]
data['T-F'] = data['T-F'].map(map_TF)
data['J-P'] = data['type'].astype(str).str[3]
data['J-P'] = data['J-P'].map(map_JP)

data['http_per_comment'] = data['posts'].apply(lambda x: x.count('http')/50)
data['music_per_comment'] = data['posts'].apply(lambda x: x.count('music')/50)
data['question_per_comment'] = data['posts'].apply(lambda x: x.count('?')/50)
data['img_per_comment'] = data['posts'].apply(lambda x: x.count('jpg')/50)
data['excl_per_comment'] = data['posts'].apply(lambda x: x.count('!')/50)
data['ellipsis_per_comment'] = data['posts'].apply(lambda x: x.count('...')/50)

#Building ML on 'type' column 
X = data.drop(['type','posts','I-E','N-S','T-F','J-P'], axis=1).values
y = data['type'].values

print("y shape: ", y.shape)
print("x shape: ", X.shape)
print("\n")

# Split arrays or matrices into random train and test subsets.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state=5)

# *****************************************************************************************************
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print("Training Data Set",round(acc_random_forest,2,), "%")

acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
print("Testing Data Set", round(acc_random_forest,2,), "%")

print("\n")

# *****************************************************************************************************
# Logistic Regression
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
print(round(acc_log,2,), "%")

# *****************************************************************************************************
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print(round(acc_knn,2,), "%")

print("\n")

# *****************************************************************************************************
# introversion/extroversion 
XX = data.drop(['type','posts','I-E'], axis=1).values
yy = data['I-E'].values

XX_train,XX_test,yy_train,yy_test=train_test_split(XX,yy,test_size = 0.1, random_state=5)
print("Introversion/Extroversion Personality Types")

random_forestt = RandomForestClassifier(n_estimators=100)
random_forestt.fit(XX_train, yy_train)

random_forestt.score(XX_train, yy_train)
acc_random_forestt = round(random_forestt.score(XX_train, yy_train) * 100, 2)
print("Random Forest Predictions Model",round(acc_random_forestt,2,), "%")

# *****************************************************************************************************
# Logistic Regression
logregg = LogisticRegression()
logregg.fit(XX_train, yy_train)

acc_logg = round(logregg.score(XX_train, yy_train) * 100, 2)
print("Logisitic Regression Prediction Accuracy",round(acc_logg,2,), "%")

# KNN
knnn = KNeighborsClassifier(n_neighbors = 3)
knnn.fit(XX_train, yy_train)

acc_knnn = round(knnn.score(XX_train, yy_train) * 100, 2)
print("Knn neighbor prediction value",round(acc_knnn,2,), "%")
print("\n")

# *****************************************************************************************************
# Intuitive/sensing 
XXX = data.drop(['type','posts','N-S'], axis=1).values
yyy = data['N-S'].values

XXX_train,XXX_test,yyy_train,yyy_test=train_test_split(XXX,yyy,test_size = 0.1, random_state=5)
print("Intuitive/Sensing Personality Types")

random_forestt = RandomForestClassifier(n_estimators=100)
random_forestt.fit(XXX_train, yyy_train)

random_forestt.score(XXX_train, yyy_train)
acc_random_forestt = round(random_forestt.score(XXX_train, yyy_train) * 100, 2)
print("Random Forest Predictions Model",round(acc_random_forestt,2,), "%")

# *****************************************************************************************************
# Logistic Regression
logregg = LogisticRegression()
logregg.fit(XXX_train, yyy_train)

Y_predd = logregg.predict(XXX_test)

acc_logg = round(logregg.score(XXX_train, yyy_train) * 100, 2)
print("Logisitic Regression Prediction Accuracy",round(acc_logg,2,), "%")

# *****************************************************************************************************
# KNN
knnn = KNeighborsClassifier(n_neighbors = 3)
knnn.fit(XXX_train, yyy_train)

acc_knnn = round(knnn.score(XXX_train, yyy_train) * 100, 2)
print("Knn neighbor prediction value",round(acc_knnn,2,), "%")
print("\n")

# *****************************************************************************************************
# Thinking/Feeling 
X4 = data.drop(['type','posts','T-F'], axis=1).values
y4 = data['T-F'].values

X4_train,X4_test,y4_train,y4_test=train_test_split(X4,y4,test_size = 0.1, random_state=5)
print("Thinking/Feeling Personality Types")

random_forestt = RandomForestClassifier(n_estimators=100)
random_forestt.fit(X4_train, y4_train)

random_forestt.score(X4_train, y4_train)
acc_random_forestt = round(random_forestt.score(X4_train, y4_train) * 100, 2)
print("Random Forest Predictions Model",round(acc_random_forestt,2,), "%")

# *****************************************************************************************************
# Logistic Regression
logregg = LogisticRegression()
logregg.fit(X4_train, y4_train)

acc_logg = round(logregg.score(X4_train, y4_train) * 100, 2)
print("Logisitic Regression Prediction Accuracy",round(acc_logg,2,), "%")

# *****************************************************************************************************
# KNN
knnn = KNeighborsClassifier(n_neighbors = 3)
knnn.fit(X4_train, y4_train)

acc_knnn = round(knnn.score(X4_train, y4_train) * 100, 2)
print("Knn neighbor prediction value",round(acc_knnn,2,), "%")
print("\n")

# *****************************************************************************************************
# Judging/perceiving
X5 = data.drop(['type','posts','J-P'], axis=1).values
y5 = data['J-P'].values

X5_train,X5_test,y5_train,y5_test=train_test_split(X5,y5,test_size = 0.1, random_state=5)
print("Judging/Perceiving Personality Types")

random_forestt = RandomForestClassifier(n_estimators=100)
random_forestt.fit(X5_train, y5_train)

random_forestt.score(X5_train, y5_train)
acc_random_forestt = round(random_forestt.score(X5_train, y5_train) * 100, 2)
print("Random Forest Predictions Model",round(acc_random_forestt,2,), "%")

# *****************************************************************************************************
# Logistic Regression
logregg = LogisticRegression()
logregg.fit(X5_train, y5_train)

acc_logg = round(logregg.score(X5_train, y5_train) * 100, 2)
print("Logisitic Regression Prediction Accuracy",round(acc_logg,2,), "%")

# *****************************************************************************************************
# KNN
knnn = KNeighborsClassifier(n_neighbors = 3)
knnn.fit(X5_train, y5_train)

acc_knnn = round(knnn.score(X5_train, y5_train) * 100, 2)
print("Knn neighbor prediction value",round(acc_knnn,2,), "%")
print("\n")






# #INTJ Wordcloud
# import cv2
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# # from nltk.tokenize import word_tokenize
# dataFile_1 = data[data['type'] == 'INTJ']
# text = str(dataFile_1['posts'].tolist())

# img=cv2.imread(path + "intj.png") #Please add your path here

# rgbimg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# INTJ_mask = np.array(rgbimg)
# stopwords = set(STOPWORDS)
# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=INTJ_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)

# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(INTJ_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('INTJ', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(INTJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('ARCHITECT', loc='Center', fontsize=14)
# plt.axis("off")
# plt.show()

# #INTP
# import cv2
# dataFile_2 = data[data['type'] == 'INTP']
# text = str(dataFile_2['posts'].tolist())

# img2=cv2.imread("intp-2.png") #Please add your path here

# rgbimg2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
# INTP_mask = np.array(rgbimg2)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=INTP_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(INTP_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('INTP', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(INTP_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('LOGICIAN', loc='Center', fontsize=14)
# plt.axis("off")

# #ENTJ
# import cv2
# dataFile_3 = data[data['type'] == 'ENTJ']
# text = str(dataFile_3['posts'].tolist())
# img3=cv2.imread("entj.png") #Please add your path here

# imgr3=cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)
# ENTJ_mask = np.array(imgr3)
# stopwords = set(STOPWORDS)
# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ENTJ_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ENTJ_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ENTJ', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ENTJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('COMMANDER', loc='Center', fontsize=14)
# plt.axis("off")

# #ENTP
# dataFile_4 = data[data['type'] == 'ENTP']
# text = str(dataFile_4['posts'].tolist())
# img4=cv2.imread("entp.png") #Please add your path here
# imgr4=cv2.cvtColor(img4,cv2.COLOR_BGR2RGB)

# ENTP_mask = np.array(imgr4)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ENTP_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ENTP_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ENTP', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ENTP_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('DEBATER', loc='Center', fontsize=14)
# plt.axis("off")

# #INFJ
# dataFile_5 = data[data['type'] == 'INFJ']
# text = str(dataFile_5['posts'].tolist())
# img5=cv2.imread("infj.png") #Please add your path here
# imgr5=cv2.cvtColor(img5,cv2.COLOR_BGR2RGB)
# INFJ_mask = np.array(imgr5)
# stopwords = set(STOPWORDS)
# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=INFJ_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(INFJ_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('INFJ', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(INFJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('ADVOCATE', loc='Center', fontsize=14)
# plt.axis("off")

# #INFP
# dataFile_6 = data[data['type'] == 'INFP']
# text = str(dataFile_6['posts'].tolist())
# img6=cv2.imread("infp.png") #Please add your path here
# imgr6=cv2.cvtColor(img6,cv2.COLOR_BGR2RGB)
# INFP_mask = np.array(imgr6)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=INFP_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(INFP_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('INFP', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(INFP_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('MEDIATOR', loc='Center', fontsize=14)
# plt.axis("off")


# #ENFJ
# dataFile_7 = data[data['type'] == 'ENFJ']
# text = str(dataFile_7['posts'].tolist())
# img7=cv2.imread("enfj.png") #Please add your path here
# imgr7=cv2.cvtColor(img7,cv2.COLOR_BGR2RGB)
# ENFJ_mask = np.array(imgr7)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ENFJ_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ENFJ_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ENFJ', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ENFJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('PROTAGONIST', loc='Center', fontsize=14)
# plt.axis("off")

# #ENFP
# dataFile_8 = data[data['type'] == 'ENFP']
# text = str(dataFile_8['posts'].tolist())
# img8=cv2.imread("enfp.png") #Please add your path here
# imgr8=cv2.cvtColor(img8,cv2.COLOR_BGR2RGB)
# ENFP_mask = np.array(imgr8)
# stopwords = set(STOPWORDS)
# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ENFP_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ENFP_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ENFP', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ENFP_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('CAMPAIGNER', loc='Center', fontsize=14)
# plt.axis("off")


# #ISTJ
# dataFile_9 = data[data['type'] == 'ISTJ']
# text = str(dataFile_9['posts'].tolist())
# img9=cv2.imread("istj.png") #Please add your path here
# imgr9=cv2.cvtColor(img9,cv2.COLOR_BGR2RGB)
# ISTJ_mask = np.array(imgr9)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ISTJ_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ISTJ_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ISTJ', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ISTJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('LOGISTICIAN', loc='Center', fontsize=14)
# plt.axis("off")

# #ISFJ
# dataFile_10 = data[data['type'] == 'ISFJ']
# text = str(dataFile_10['posts'].tolist())
# img10=cv2.imread("isfj.png") #Please add your path here
# imgr10=cv2.cvtColor(img10,cv2.COLOR_BGR2RGB)
# ISFJ_mask = np.array(imgr10)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ISFJ_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ISFJ_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ISFJ', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ISFJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('DEFENDER', loc='Center', fontsize=14)
# plt.axis("off")

# #ESTJ
# dataFile_11 = data[data['type'] == 'ESTJ']
# text = str(dataFile_11['posts'].tolist())
# img11=cv2.imread("estj.png") #Please add your path here
# imgr11=cv2.cvtColor(img11,cv2.COLOR_BGR2RGB)
# ESTJ_mask = np.array(imgr11)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ESTJ_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ESTJ_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ESTJ', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ESTJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('EXECUTIVE', loc='Center', fontsize=14)
# plt.axis("off")

# #ESFJ
# dataFile_12 = data[data['type'] == 'ESFJ']
# text = str(dataFile_12['posts'].tolist())
# img12=cv2.imread("esfj.png") #Please add your path here
# imgr12=cv2.cvtColor(img12,cv2.COLOR_BGR2RGB)
# ESFJ_mask = np.array(imgr12)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ESFJ_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ESFJ_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ESFJ', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ESFJ_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('CONSUL', loc='Center', fontsize=14)
# plt.axis("off")

# #ISTP
# dataFile_13 = data[data['type'] == 'ISTP']
# text = str(dataFile_13['posts'].tolist())
# img13=cv2.imread("istp.png") #Please add your path here
# imgr13=cv2.cvtColor(img13,cv2.COLOR_BGR2RGB)
# ISTP_mask = np.array(imgr13)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ISTP_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ISTP_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ISTP', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ISTP_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('VIRTUOSO', loc='Center', fontsize=14)
# plt.axis("off")

# #ISFP
# dataFile_14 = data[data['type'] == 'ISFP']
# text = str(dataFile_14['posts'].tolist())
# img14=cv2.imread("isfp.png")#Please add your path here
# imgr14=cv2.cvtColor(img14,cv2.COLOR_BGR2RGB)
# ISFP_mask = np.array(imgr14)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ISFP_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ISFP_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ISFP', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ISFP_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('ADVENTURER', loc='Center', fontsize=14)
# plt.axis("off")

# #ESTP
# dataFile_15 = data[data['type'] == 'ESTP']
# text = str(dataFile_15['posts'].tolist())
# img15=cv2.imread("estp.png")#Please add your path here
# imgr15=cv2.cvtColor(img15,cv2.COLOR_BGR2RGB)
# ESTP_mask = np.array(imgr15)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ESTP_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)

# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ESTP_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ESTP', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ESTP_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('ENTREPRENEUR', loc='Center', fontsize=14)
# plt.axis("off")

# #ESFP
# from nltk.tokenize import word_tokenize
# dataFile_16 = data[data['type'] == 'ESFP']
# text = str(dataFile_16['posts'].tolist())
# img16=cv2.imread("esfp.png") #Please add your path here
# imgr16=cv2.cvtColor(img16,cv2.COLOR_BGR2RGB)
# ESFP_mask = np.array(imgr16)
# stopwords = set(STOPWORDS)

# text_tokens = word_tokenize(text)

# tokens_without_sw = [word for word in text_tokens if not word in stopwords]
# wc = WordCloud(background_color="white", max_words=2000, mask=ESFP_mask,
#                stopwords=stopwords)

# text = (" ").join(tokens_without_sw)
# # generate word cloud
# wc.generate(text)

# # create coloring from image
# image_colors = ImageColorGenerator(ESFP_mask)

# # show
# plt.figure(figsize=(20,10))

# plt.subplot(121)
# plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
# plt.title('ESFP', loc='Center', fontsize=14)
# plt.axis("off")

# plt.subplot(122)
# plt.imshow(ESFP_mask, cmap=plt.cm.gray, interpolation="bilinear")
# plt.title('ENTERTAINER', loc='Center', fontsize=14)
# plt.axis("off")



