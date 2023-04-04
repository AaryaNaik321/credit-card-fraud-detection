from cgi import print_arguments
import numpy as np 
import pandas as pd 
import os 
os.chdir("C:/Users/saksh/OneDrive/Desktop/pratice/credit card")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
df = pd.read_csv("creditcard.csv")
fraud = df[df['Class']==1]

normal = df[df['Class']==0]
df1= df.sample(frac = 0.1,random_state=1)

df1.shape
Fraud = df1[df1['Class']==1]

Valid = df1[df1['Class']==0]
df2 = df.iloc[:50000,:31]
x =df2
x =df2.drop(['Class'],axis=1)
y = df2['Class']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
inpt=np.array([[631	,-1.058930942,	0.114078303,	1.485336984	,0.534704046,	0.754274552,	-0.496412436,	0.580911663,	0.143710655	,-0.079361502,	-0.741373462,	-0.968240369,	-0.250369967,	-1.211699997,	0.161282583,	-0.606521163,	-0.551640144	,0.088074308,	-0.716170077,	-0.862826912,	-0.014561024,	0.067736869,	0.056591336,	0.032287102,	0.035178216	,0.098191999,	-0.489206918,	0.090100667,	0.137385826,	57.64]])
lr_model.score(x_test,y_test)
def give_pred(inpt):
 inpt=np.array(inpt)
 predit =lr_model.predict(inpt)
 print(predit)
 return predit
 
give_pred(inpt)

#predit=lr_model.predict(np.array([[1.0,-0.966272,-0.185226,1.792993,-0.863291,-0.010309,1.247203,0.237609,0.377436,-1.387024,0.108300,0.005274,-0.190321,-1.175575,0.647376,-0.221929,0.062723,0.061458,123.50	]])) 
#print(predit) 