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
#C:\\Users\\mosta\\Downloads\\mycodelap\\gradpro\\
path = 'bank-additional-full.csv'
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
#Correlation plot of attributes

f,ax=plt.subplots(figsize=(10,10))
ax.set_title('Correlation plot of attributes')
sns.heatmap(bank_csv.corr(),annot=True,linewidths=0.5,linecolor="black",fmt=".1f",ax=ax)
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
allfigs.savefig(ax.figure)

allfigs.close()
#closing of the pdf


#Now that we have removed outliers, we can proceed for more feature engineering techniques.

bank_features = bank_csv.copy()
#encoding education 
lst=['basic.9y','basic.6y','basic.4y']
for i in lst:
    bank_features.loc[bank_features['education'] == i, 'education'] = "middle.school"

print('education value counts \n' , bank_features['education'].value_counts())




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
print(ordinal_labels2)
#drop marital and add marital_ordinal column 
bank_features['marital_ordinal']=bank_features['marital'].map(ordinal_labels2)
bank_features.drop(['marital'], axis=1,inplace=True)

#bank_features.to_csv('C:\\Users\\mosta\\Downloads\\mycodelap' , index=False)
print(bank_features.head())

#bank_features.to_csv('C:\\Users\\mosta\\Downloads\\mycodelap' , index=False)
print('bank feature head data frame')
print(bank_features.head())
print('bank feature columns')
print(bank_features.columns)

#---------------------------------------------------------
#saving a copy of bank features after the edits in a csv file 
# path = 'C:\\Users\\mosta\\Downloads\\mycodelap\\bank_features.csv'
# bank_features.to_csv(path , encoding='UTF-8', index=False )



#---------------------------------------------------------
#we need to encode (jop , education , contact )


contact_incoder = LabelEncoder()


 
contact_values = contact_incoder.fit_transform(bank_features['contact'])

print('contact before encoding ' , bank_features['contact'][:10])
print('contact values after encoding ' , contact_values[:10])

bank_features['new_contact'] = contact_values
print(bank_features.columns)
print(bank_features.head())
bank_features.pop('contact')
print('after droping contact')
print(bank_features.columns)
#---------------------------------------------------------------


# handling education

print(bank_features['education'].value_counts())
print(bank_features['education'].unique)

education_encoder = LabelEncoder()
education_encoded = education_encoder.fit_transform(bank_features['education'])

print('education after encoding ' , bank_features['education'][:20])
print('education after encoding ' , education_encoded[:20])
education_encoded_df = pd.DataFrame(education_encoded)
print('education encoded value counts '  , education_encoded_df.value_counts() ) # 6 catigories
bank_features['new_education'] = education_encoded
bank_features.pop('education')
print(bank_features.head())

#education col handeled

#----------------------------------------------------------------
print(bank_features['job'].value_counts())
print(bank_features['job'].unique())

job_encoder = LabelEncoder()

job_encoded = job_encoder.fit_transform(bank_features['job'])

print('job after encoding ' , job_encoded[:20])

bank_features['new_job'] = job_encoded

bank_features.pop('job')
print(bank_features.head(20))

#--------------------------------------------------------------
#handling poutcome 
poutcome_encoder = LabelEncoder()

poutcome_encoded = poutcome_encoder.fit_transform(bank_features['poutcome'])
print(poutcome_encoded[:100])

bank_features['new_poutcome'] = poutcome_encoded
bank_features.pop('poutcome')
print(bank_features.head(30))

print(bank_features['new_poutcome'].value_counts() )





#-----------------------------------------
#scaling the features 
#Standardization of numerical variables
bank_features.drop('age_group', axis=1, inplace=True)
bank_scale=bank_features.copy()

Categorical_variables=['new_job', 'new_education', 'default', 'housing', 'loan', 'month','day_of_week','y', 'marital_ordinal' , 'new_contact' ]

features_scale = [feature for feature in bank_scale.columns if feature not in Categorical_variables]

print('col for bank scale \n' , bank_scale[features_scale].columns  )

#bank_scale.pop('age_group')


scaler=StandardScaler()
scaler.fit(bank_scale[features_scale])
scaled_data = pd.concat([bank_scale[['new_job', 'new_education', 'default', 'housing', 'loan', 'month','day_of_week','y', 'marital_ordinal' , 'new_contact' ]].reset_index(drop=True),
                         pd.DataFrame(scaler.transform(bank_scale[features_scale]),columns=features_scale)],axis=1)

print(scaled_data.head(20))
scaled_data.pop('conversion')
scaled_data.to_csv('scaled_data.csv' , encoding='UTF-8', index=False )