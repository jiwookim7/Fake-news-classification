import pandas as pd
import numpy as np 
from sklearn.decomposition import randomized_svd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import re # removing the special character and link from the text
import string 
import os

dfTrue = pd.read_csv('C:/Users/yjk93/OneDrive/문서/Cpts315/FinalProject/True.csv')
dfFake = pd.read_csv('C:/Users/yjk93/OneDrive/문서/Cpts315/FinalProject/Fake.csv')



dfFake["class"] = 0
dfTrue["class"] = 1


# i will take 10 rows of each data set 
# remove row from the main data set

#print(df_fake.shape, df_true.shape)

dfFakeManualTest = dfFake.tail(10)
for line in range(23480, 23470, -1):
    dfFake.drop([line], axis=0, inplace=True)

dfTrueManualTest = dfTrue.tail(10)
for line in range(21416,21406, -1):
    dfTrue.drop([line], axis=0, inplace=True)

#create csv file
dfManualTesting = pd.concat([dfFakeManualTest, dfTrueManualTest], axis=0)
dfManualTesting.to_csv('manual_testing.csv')


dfMerge = pd.concat([dfFake, dfTrue], axis=0)

#delete the column
df = dfMerge.drop(["title", "subject", "date"], axis=1)

#print(df.head(10))


# suffle the dataset

df = df.sample(frac=1)

#print(df.head(10))

# check the null value is present or not

#print(df.isnull().sum())

# remove all the special character and dot other unnessary charater

def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


df["text"] = df["text"].apply(word_drop)

#print(df.head(10))

## going to define our dependent and independent variable as x and y

x = df["text"]
y = df["class"]

## I will take 25 percent data as a test set
xTrain, xTest, yTrain,yTest = train_test_split(x,y, test_size=.25)

# vectorize that x variable for any claculation 

vectorization = TfidfVectorizer()
xvTrain = vectorization.fit_transform(xTrain)
xvTest = vectorization.transform(xTest)

# logistic regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(xvTrain, yTrain)


# shoot the score of the model

#print(LR.score(xvTest, yTest)) #98% accurate

predictLR = LR.predict(xvTest)

#it will compare actual with prediction value
#print(classification_report(yTest, predictLR))

# Decision Tree classification
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xvTrain, yTrain)

#print(DT.score(xvTest, yTest)) #99% accurate

predDT = DT.predict(xvTest)

#print(classification_report(yTest, predDT))

## Gradlent Boosting classifier 

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)

GBC.fit(xvTrain, yTrain)

#print(GBC.score(xvTest, yTest))

predGBC = GBC.predict(xvTest)

#print(classification_report(yTest, predGBC))

# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)

RFC.fit(xvTrain, yTrain)

#print(RFC.score(xvTest, yTest))

predRFC = RFC.predict(xvTest)

#print(classification_report(yTest, predRFC))


# Manual Testing

def output_label(n):
    if n ==0:
        return "Fake News"
    elif n == 1:
        return "Not a Fake News"

def manual_testing(news):
    testingNews = {"text":[news]}
    newdeftest = pd.DataFrame(testingNews)
    newdeftest["text"] = newdeftest["text"].apply(word_drop)
    newxtest = newdeftest["text"]
    newxvtest = vectorization.transform(newxtest)
    predLR = LR.predict(newxvtest) # 0 or 1
    predDT = DT.predict(newxvtest) # 0 or 1
    predGBC = GBC.predict(newxvtest) # 0 or 1
    predRFC = RFC.predict(newxvtest) # 0 or 1

    return print("\n\n(First_LR): {} \n(Second_DT): {} \n(Third_GBC): {} \n(Forth_RFC) : {}".format(output_label(predLR),
    output_label(predDT),
    output_label(predGBC),
    output_label(predRFC)))




news = str(input("Enter the News Text: "))
manual_testing(news)
os.remove('manual_testing.csv')
