# gradpro
 applying the principles of  data analysis and machine learning

Dataset:

We have 4118 instances and 21 features. The information says there are no null values. Fishy right? anyway we will strictly scrutinize each feature and check for suspicious records and manipulate them

Attributes: Bank client data:

Age : Age of the lead (numeric)
Job : type of job (Categorical)
Marital : Marital status (Categorical)
Education : Educational Qualification of the lead (Categorical)
Default: Does the lead has any default(unpaid)credit (Categorical)
Housing: Does the lead has any housing loan? (Categorical)
loan: Does the lead has any personal loan? (Categorical)
Related with the last contact of the current campaign:

Contact: Contact communication type (Categorical)
Month: last contact month of year (Categorical)
day_of_week: last contact day of the week (categorical)
duration: last contact duration, in seconds (numeric).
Important note: Duration highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

Other attributes:

campaign: number of contacts performed during this campaign and for this client (numeric)
pdays: number of days that passed by after the client was last contacted from a previous campaign(numeric; 999 means client was not previously contacted))
previous: number of contacts performed before this campaign and for this client (numeric)
poutcome: outcome of the previous marketing campaign (categorical)
Social and economic context attributes

emp.var.rate: employment variation rate - quarterly indicator (numeric)
cons.price.idx: consumer price index - monthly indicator (numeric)
cons.conf.idx: consumer confidence index - monthly indicator (numeric)
euribor3m: euribor 3 month rate - daily indicator (numeric)
nr.employed: number of employees - quarterly indicator (numeric)
Output variable (desired target):

y - has the client subscribed a term deposit? (binary: 'yes','no')
