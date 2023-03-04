import pandas as pd 
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

#Metrics Phase
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

data = pd.read_csv('scaled_data.csv')
print(data.head())
print(data['y'][:10])
print(data.columns)

X= data.drop('y', axis = 1)
y= data['y']
# checking the feature importance
model = ExtraTreesClassifier()
model.fit(X,y)

feature_importance = pd.Series(model.feature_importances_, index = X.columns)
feature_importance.nlargest(10).plot(kind = 'barh')
plt.show()