#created by tefa101

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


#readin data 
path = 'C:\\Users\\mosta\\Downloads\\mycodelap\\gradpro\\bank-additional-full.csv'
bank_csv = pd.read_csv( path, sep = ';')

#figures list 
allfigs = PdfPages('allplots2.pdf')

#reshape the conversion attribute to zeros and ones 
# 1 ---> converted 
# 0 ----> akkeeeed not converted 
bank_csv['conversion'] = bank_csv['y'].apply(lambda x : 1 if x == 'yes' else 0)

##notes
# no null values 
# calculating conversion rate for this campaign 
# conversion rate = converted / total clients 

conversion_rate =( bank_csv['conversion'].sum() / bank_csv.shape[0] )*100

print('conversion rate for total campaign : {}'.format(conversion_rate)) #for all the campaign 



conversion_rate_for_ages = bank_csv.groupby('age')['conversion'].sum() / bank_csv.groupby('age')['conversion'].count() *100.0
print(conversion_rate_for_ages.head())  #grouped by age 


bank_csv['age_group'] = bank_csv['age'].apply(lambda x : '(18 , 30)' if x<30 else '(30 , 40 )' if x<40 else '(40 , 50 )' if x<50 else '(50 , 60 )' if x <60 else '(60 , 70)' if x<70 else '70+' )
print(bank_csv.columns)
conversion_by_age_group = bank_csv.groupby('age_group')['conversion'].sum() / bank_csv.groupby('age_group')['conversion'].count() *100.0
conversion_by_age_group = pd.DataFrame(conversion_by_age_group)

print(conversion_by_age_group.head())

## first figure 
ax = conversion_by_age_group.plot(title='conversion for age groups' , kind='bar')
plt.xlabel('age group')
plt.ylabel('conversion %')
plt.show()
allfigs.savefig(ax.figure)


#then grouped by age and marital ()


age_marital_df = bank_csv.groupby(['age_group' , 'marital'])['conversion'].sum().unstack('marital').fillna(0)

age_marital_df = age_marital_df.divide( bank_csv.groupby('age_group')['conversion'].count(),axis=0 )

print(age_marital_df.head())    

ax = age_marital_df.plot(title='age - marital conversion rates' , kind= 'bar' )
plt.xlabel('age groups')
plt.ylabel('conversion')
plt.show()
allfigs.savefig(ax.figure)

#------------------------------------------------------

##move on to outliars
plt.figure(figsize=(15,10))
plt.style.use('seaborn')
ax = plt.subplot(221)
plt.boxplot(bank_csv['age'])
ax.set_title('age')
ax = plt.subplot(222)
plt.boxplot(bank_csv['duration'])
ax.set_title('duration')
ax = plt.subplot(223)
plt.boxplot(bank_csv['campaign'])
ax.set_title('campaign')

plt.show()
allfigs.savefig(ax.figure)

##remove the outliars

numerical_features = ['age' , 'duration' , 'campaign']
for cols in numerical_features :
    q1 = bank_csv[cols].quantile(0.25) #get quantile1
    q2 = bank_csv[cols].quantile(0.75) #get quantile2
    iqr = q2 -q1  #get the iqr
    filter = (bank_csv[cols] >= q1 - 1.5 * iqr) & (bank_csv[cols]   <= q2 + 1.5*iqr) #define the filter
    bank_csv = bank_csv.loc[filter] #apply the filter to thhe dataset

#looking at the data after removing outliers 
plt.figure(figsize=(15,10))
plt.style.use('seaborn')

ax = plt.subplot(221)
plt.boxplot(bank_csv['age'])
ax.set_title('age')

ax = plt.subplot(222)
plt.boxplot(bank_csv['duration'])
ax.set_title('duration')

ax = plt.subplot(223)
plt.boxplot(bank_csv['campaign'])
ax.set_title('campaign')

plt.show()
#allfigs.savefig(ax.figure)



allfigs.close()


#Now that we have removed outliers, we can proceed for more feature engineering techniques.

bank_features = bank_csv.copy()
#encoding education 
lst=['basic.9y','basic.6y','basic.4y']
for i in lst:
    bank_features.loc[bank_features['education'] == i, 'education'] = "middle.school"

print(bank_features['education'].value_counts())




#encoding months and days to numbers 
month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}
bank_features['month']= bank_features['month'].map(month_dict)

day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}
bank_features['day_of_week']= bank_features['day_of_week'].map(day_dict) 

#encoding the features that has yes and no values

dictionary={'yes':1,'no':0,'unknown':-1}
bank_features['housing']=bank_features['housing'].map(dictionary)
bank_features['default']=bank_features['default'].map(dictionary)
bank_features['loan']=bank_features['loan'].map(dictionary)

dictionary1={'no':0,'yes':1}
bank_features['y']=bank_features['y'].map(dictionary1)

print(bank_features.loc[:,['housing','default','loan','y']].head())

#the marital feature 

ordinal_labels=bank_features.groupby(['marital'])['y'].mean().sort_values().index
print(ordinal_labels)

ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}
ordinal_labels2
#drop marital and add marital_ordinal column 
bank_features['marital_ordinal']=bank_features['marital'].map(ordinal_labels2)
bank_features.drop(['marital'], axis=1,inplace=True)

bank_features.to_csv('C:\\Users\\mosta\\Downloads\\mycodelap' , index=False)

#-----------------------------------------
#scaling the features 

bank_scale=bank_features.copy()
Categorical_variables=['job', 'education', 'default', 'housing', 'loan', 'month','day_of_week','y', 'marital_ordinal']


feature_scale=[feature for feature in bank_scale.columns if feature not in Categorical_variables]
print(bank_scale.head())


# features = ['job', 'education', 'default', 'housing', 'loan', 'month','day_of_week','y', 'marital_ordinal']
# scaler=StandardScaler()
# scaler.fit(bank_scale[feature_scale])
# scaled_data = pd.concat([bank_scale[features].reset_index(drop=True),pd.DataFrame(scaler.transform(bank_scale[feature_scale]), columns=feature_scale)],axis=1)
# print(scaled_data.head())
