#created by tefa101

import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_pdf import PdfPages

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
allfigs.savefig(ax.figure)



allfigs.close()