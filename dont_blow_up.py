
# CREATED BY:  TEFA101

import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Metrics Phase
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

data = pd.read_csv('scaled_data.csv')
allfigs = PdfPages('model_plots.pdf')
print(data.head())
print(data['y'][:10])
print(data.columns)

X= data.drop('y', axis = 1)
y= data['y']
# checking the feature importance
model = ExtraTreesClassifier()
model.fit(X,y)

feature_importance = pd.Series(model.feature_importances_, index = X.columns)
ax = feature_importance.nlargest(10).plot(kind = 'barh')
plt.show()


#Train and Test Split (80:20)

X=data.drop(['pdays','month','cons.price.idx','loan','housing','emp.var.rate','y'],axis=1)
y=data.y

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=1)
print("Input Training:",X_train.shape)
print("Input Test:",X_test.shape)
print("Output Training:",y_train.shape)
print("Output Test:",y_test.shape)

#------------- MODEL SELECTION---------------s-#

#creating the objects
logreg_cv = LogisticRegression(random_state=0)
dt_cv=DecisionTreeClassifier()
knn_cv=KNeighborsClassifier()
svc_cv=SVC()
nb_cv=BernoulliNB()
cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree',2:'KNN',3:'SVC',4:'Naive Bayes'}
cv_models=[logreg_cv,dt_cv,knn_cv,svc_cv,nb_cv]

# this takes a lot of time to run so i commented it out

# for i,model in enumerate(cv_models):
#     print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, X, y, cv=10, scoring ='accuracy').mean()))


# # logistic regression has the highst score 
# so we will use it for the rest of the model selection

#------------------ LOGISTIC REGRESSION -----------------#
# FOOOCUUUSSSS *** hard **** important

# Logistic regression with Hyperparameter tuning

param_grid = {'C': np.logspace(-4, 4, 50), 'penalty':['l1', 'l2']}
clf = GridSearchCV(LogisticRegression(random_state=0), param_grid,cv=5, verbose=0,n_jobs=-1)
best_model = clf.fit(X_train,y_train)
print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(X_test,y_test))

# we got 94% accuracy with logistic regression 
logreg = LogisticRegression(C=0.18420699693267145, random_state=0)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# the Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",confusion_matrix)
print("Classification Report:\n",classification_report(y_test, y_pred))

'''
From the EDA and model selection part we can clearly identify duration playing an important attribute in defining the outcome of our dataset. 
It is absolute that the more the leads are interested in starting a deposit will have higher number of calls and the call duration will be higher than the average. 
We have also figured out that job and education also acts as a crucial deciding factor and influences the outcome alot.

Here are the few recommendations for the bank than can help improve the deposit rate

Classify job roles based on corporate tiers and approach all tier 1 employees extract more information to deliver the best deposit plan, 
which can increase the duration of calls and that can lead to a deposit
Approaching the leads during the start of new bank period(May-July) will be a good choice as many have shown positive results from data history
Tune the campaign according to the national econometrics, don't chanelize the expenses on campaign when the national economy is performing poor

'''

'''
Insights:

The leads who have not made a deposit have lesser duration on calls
Comparing the average, the blue collar, entrepreneur have high duration in calls and student, retired have less duration in average
Large distribution of leads were from self employed clients and management people.

'''