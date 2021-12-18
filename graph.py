# Import Libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#Add Dataset

data = pd.read_excel(r'C:\Users\navun\Documents\Project 1\Dataset.xlsx')

#For plotting data we need to remove null or fill null values if needed
#Check null or empty data 
#because Coloumns 6 has No values is dataset
#data.info()
data = data.drop(columns=['Column6'])

#Graph Total Customer Based on Family Type
dataWithFamily = data['FamilyStatus'] 
dataBasedOnFamilyType=dataWithFamily.value_counts()
labelForFamily = ['Family', 'Singles']
fig, ax = plt.subplots()
#plt.
plt.pie(dataBasedOnFamilyType, labels=labelForFamily,autopct='%.1f%%')
ax.set_title('Total Customer Based on Family Type ')
plt.legend(loc=(1.02,0))
plt.show()


#Graph  Total Customers Based on ageGroup	

dataWithAgeGroup = data['AgeGroup'] 
dataWithAgeGroup=dataWithAgeGroup.value_counts()
labels = ['40 To 50', 'Seniors','50 To 60','30 To 40','20 To 30','SuperSenior']
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title('Total Customer Based on Age Group')
plt.pie(dataWithAgeGroup, labels=labels,autopct='%.1f%%')
plt.pie(dataWithAgeGroup)
plt.legend(loc=(1.02,0))
plt.show()

#Total Customers Based on Income Bracket
dataWithIncomeBracket=data['IncomeBracket'].value_counts(ascending=True)
labels=['High','HNI','Low','Medium']
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title('Total Customer Based on IncomeBracket')

plt.pie(dataWithIncomeBracket, labels=labels,autopct='%.1f%%')
plt.legend(loc=(1.02,0))
plt.show()

#Total Customers Based on Education	
DataWithEducation = data['Education'].value_counts(ascending=True)
labels=['Basic','2n Cycle','Master','PHD','Graduation']
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title('Total Customer Based on Education')
plt.pie(DataWithEducation, labels=labels,autopct='%.1f%%')
plt.legend(loc=(1.02,0))
plt.show()


#Total number of customers / families with kids or no kids
FamilyWithKid = data['FamilyWithKid'].value_counts(ascending=True)
labels=['No','Yes']
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title('Total Customer Based on FamilyWithKid')
plt.pie(FamilyWithKid, labels=labels,autopct='%.1f%%')
plt.legend()
plt.legend(loc=(1,0))
plt.show()


#How people in various income brackets make their purchases in different products?
df = pd.read_excel(r'D:\balram\navjot\Dataset.xlsx')
df.columns

df1 = df.groupby('IncomeBracket').sum()
df = df[['IncomeBracket', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
df1 = df.groupby('IncomeBracket').sum().plot.bar()
df1

#How people in various Age Groups are making their purchases in different products?
data.columns
df2byWithAgeGroup = data[['AgeGroup', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
df2= df2byWithAgeGroup.groupby('AgeGroup').sum().plot.bar()
df2

#In Which Products People Have interest based on Disposable Income?			

data.columns
df2byWithIncomeBracket = data[['IncomeBracket', 'MntFruits' ,'MntWines','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
df3= df2byWithIncomeBracket.groupby('IncomeBracket').sum().plot.bar()
df3


#Which category is most active in terms of Purchases?		
data.columns
df2byWithIncomeBracketAndPurchages = data[['IncomeBracket', 'NumDealsPurchases' ,'NumWebPurchases','NumCatalogPurchases', 'NumStorePurchases']]
df4= df2byWithIncomeBracketAndPurchages.groupby('IncomeBracket').sum().plot.bar()
df4


#Who are buying our products?( Age & Income)		

data.columns
df2byWithIncomeBracketAndAgeGroupss = data[['IncomeBracket', 'AgeGroup','MntFruits','MntWines','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']]
df5= df2byWithIncomeBracketAndAgeGroupss.groupby(['IncomeBracket','AgeGroup']).sum().plot.bar()
df5
plotGraph1 = data.pivot_table(index='Education',columns='IncomeBracket', values='NumStorePurchases', aggfunc=np.sum)
plotGraph1.plot(kind='line',title='Graduates in High Income group,followed by Graduates in Medium Income Group.',figsize=(5,6))

plotGraph2 = data.pivot_table(index='AgeGroup',columns='IncomeBracket', values='NumStorePurchases', aggfunc=np.sum)
plotGraph2.plot(kind='line',title='Seniors in High Income group,followed by 40-50 age group in Medium Income.',figsize=(5,6))

plotGraph3 = data.pivot_table(index='AgeGroup',columns='IncomeBracket', values=['NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases'],aggfunc=np.sum)
plotGraph3.plot(kind='line',title='Problem : Participation is less by people in the age group of less than 40 yrs.',figsize=(20,12))

plotGraph4 = data.pivot_table(index='FamilyStatus',columns='IncomeBracket', values='NumWebPurchases', aggfunc=np.sum)
plotGraph4.plot(kind='bar',title='Sum of NumWebPurchases by family staus in different income bracket',figsize=(5,6))

plotGraph5 = data.pivot_table(index='IncomeBracket', values=['NumWebPurchases','NumCatalogPurchases','NumStorePurchases'], aggfunc=np.sum)
plotGraph5.plot(kind='bar',title='Sum of WebPurchases,CataloguePurchases,StorePurchaces According to income bracket',figsize=(5,6))

plotGraph6 = data.pivot_table(index='FamilyStatus', values=['NumWebPurchases','NumCatalogPurchases','NumStorePurchases'], aggfunc=np.sum)
plotGraph6.plot(kind='bar',title='Sum of WebPurchases,CataloguePurchases,StorePurchaces According to family status',figsize=(5,6))

plotGraph7 = data.pivot_table(index='AgeGroup', values=['NumWebPurchases','NumCatalogPurchases','NumStorePurchases'], aggfunc=np.sum)
plotGraph7.plot(kind='bar',title='Sum of WebPurchases,CataloguePurchases,StorePurchaces According to Age Group',figsize=(5,6))

coding of tableu dashboard 
import numpy as np
import pandas as pd
import datetime
import csv
!pip install tabpy
!pip istall tabpy python server
!pip istall tabpy-server
!pip istall tabpy-client
import tabpy as tp
from datetime import date
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
!pip install researchpy
!pip install quick-eda
import researchpy as rp
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
warnings.filterwarnings('ignore')
!pip install -U dataprep
%matplotlib inline
from statsmodels.formula.api import ols
data = pd.read_csv('drive/MyDrive/DAB103/marketing_campaign.csv', sep='\\t')
data
data.head()
data.tail()
type(data)
pandas.core.frame.DataFrame
data.shape
(2240, 29)
#For plotting data we need to remove null or fill null values if needed
#Check null or empty data 
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2240 entries, 0 to 2239
Data columns (total 29 columns):
 #   Column               Non-Null Count  Dtype  
---  ------               --------------  -----  
 0   "ID                  2240 non-null   object 
 1   Year_Birth           2240 non-null   int64  
 2   Education            2240 non-null   object 
 3   Marital_Status       2240 non-null   object 
 4   Income               2216 non-null   float64
 5   Kidhome              2240 non-null   int64  
 6   Teenhome             2240 non-null   int64  
 7   Dt_Customer          2240 non-null   object 
 8   Recency              2240 non-null   int64  
 9   MntWines             2240 non-null   int64  
 10  MntFruits            2240 non-null   int64  
 11  MntMeatProducts      2240 non-null   int64  
 12  MntFishProducts      2240 non-null   int64  
 13  MntSweetProducts     2240 non-null   int64  
 14  MntGoldProds         2240 non-null   int64  
 15  NumDealsPurchases    2240 non-null   int64  
 16  NumWebPurchases      2240 non-null   int64  
 17  NumCatalogPurchases  2240 non-null   int64  
 18  NumStorePurchases    2240 non-null   int64  
 19  NumWebVisitsMonth    2240 non-null   int64  
 20  AcceptedCmp3         2240 non-null   int64  
 21  AcceptedCmp4         2240 non-null   int64  
 22  AcceptedCmp5         2240 non-null   int64  
 23  AcceptedCmp1         2240 non-null   int64  
 24  AcceptedCmp2         2240 non-null   int64  
 25  Complain             2240 non-null   int64  
 26  Z_CostContact        2240 non-null   int64  
 27  Z_Revenue            2240 non-null   int64  
 28  Response"            2240 non-null   object 
dtypes: float64(1), int64(23), object(5)
memory usage: 507.6+ KB
[ ]
#Checking for null values
data.isnull().sum()
"ID                     0
Year_Birth              0
Education               0
Marital_Status          0
Income                 24
Kidhome                 0
Teenhome                0
Dt_Customer             0
Recency                 0
MntWines                0
MntFruits               0
MntMeatProducts         0
MntFishProducts         0
MntSweetProducts        0
MntGoldProds            0
NumDealsPurchases       0
NumWebPurchases         0
NumCatalogPurchases     0
NumStorePurchases       0
NumWebVisitsMonth       0
AcceptedCmp3            0
AcceptedCmp4            0
AcceptedCmp5            0
AcceptedCmp1            0
AcceptedCmp2            0
Complain                0
Z_CostContact           0
Z_Revenue               0
Response"               0
dtype: int64
 #Correlation Hedatmap for the Dataset
sns.heatmap(data.corr())
<matplotlib.axes._subplots.AxesSubplot at 0x7f48e7f619d0>
rom dataprep.eda import plot, plot_correlation, create_report, plot_missing

plot(data)
data.describe()
data.info
data.columns
Index(['"ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome',
       'Teenhome', 'Dt_Customer', 'Recency', 'MntWines', 'MntFruits',
       'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
       'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
       'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
       'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
       'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response"'],
      dtype='object')
Data Cleaning There are 24 NA rows in 'Income' columns, so we fill these NA with the mean income of all customers
data['Income'].fillna(np.mean(data['Income']), inplace=True)
data['Income'] = data['Income'] 
data
data.isnull().sum()
data.shape      #There is no change on shape of our data.
data.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)
data.drop(['"ID'], axis=1, inplace=True) 

data
data.drop(['Response"'], axis=1, inplace=True) 

data
Change Year_Birth to Age (as Age is more informative variable) and then dropping the Column Year_Birth from data.
data['Age'] = 2021 - data.Year_Birth.to_numpy()

data
Splitting the Age Groups to generate deep insights depending upon age of customers
Age_Labels = ['Upto 30Yrs', '31 To 40Yrs', '41 To 50Yrs', '51 To 60Yrs','Above 60Yrs']
cut_bins = [0, 30, 40,50 , 60,100]
data['AgeGroup'] = pd.cut(data['Age'], bins=cut_bins, labels=Age_Labels)
data['AgeGroup']
There are few outliers as observed in our preliminary visualization,there we can drop them.
data=data.dropna(subset=['AgeGroup'])


data
data.isnull().sum()
Modifying Education variable to make it more realistic to understand:
  data['Education'].value_counts()
  Renaming column names to make them shorter and easy to use in the dataframe
  data=data.rename(columns={'MntWines': 'Wine','MntFruits':'Fruit','MntMeatProducts':'Meat','MntFishProducts':'Fish','MntSweetProducts':'Sweet','MntGoldProds':'Gold','NumDealsPurchases':'NDP','AcceptedCmp1':'AC1','AcceptedCmp2':'AC2','AcceptedCmp3':'AC3','AcceptedCmp4':'AC4','AcceptedCmp5':'AC5','NumWebPurchases':'NWP','NumCatalogPurchases':'NCP','NumStorePurchases':'NSP','NumWebVisitsMonth':'NWVM'})
data
Creation of new variable 'Influencers' for those Homes with Kids/Teens.
data['Influencers']=data['Kidhome']+data['Teenhome']
data.drop(['Kidhome','Teenhome'], axis=1, inplace=True)

data
Transformin variable Recency(in days) into Active(made purchases with 60 days) or Dormant Customers(not made purchases for more than 60 days)
cut_labels_Recency = ['Active', 'Moderately Active', 'Dormant']
cut_bins = [0,30, 60, 100]
data['Recency'] = pd.cut(data['Recency'], bins=cut_bins, labels=cut_labels_Recency)
data['Recency']
#Graph Total Customer Based on Family Type
dataWithFamily = data['Marital_Status'] 
dataBasedOnFamilyType=dataWithFamily.value_counts()
labelForFamily = ['Couple', 'Singles']
fig, ax = plt.subplots()
#plt.pie(np.array(data['Marital_Status'].value_counts()))
plt.pie(dataBasedOnFamilyType, labels=labelForFamily,autopct='%.1f%%')
ax.set_title('Total Customer Based on Family Type ')
plt.legend(loc=(1.02,0))
plt.show()
#Graph  Total Customers Based on ageGroup	

dataWithAgeGroup = data['AgeGroup'] 
dataWithAgeGroup=dataWithAgeGroup.value_counts(ascending=True)
labels = ['Upto 30Yrs', '31 To 40Yrs' , '51 To 60 Yrs','Above 60Yrs ','41 To 50Yrs ']
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title('Total Customer Based on Age Group')
plt.pie(dataWithAgeGroup, labels=labels,autopct='%.1f%%')
plt.pie(dataWithAgeGroup)
plt.legend(loc=(1.02,0))
plt.show()
#Total Customers Based on Income Bracket
dataWithIncomeBracket=data['Income_Bracket'].value_counts(ascending=True)
labels=['High','Low','Medium']
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title('Total Customer Based on Income_Bracket')
plt.pie(dataWithIncomeBracket, labels=labels,autopct='%.1f%%')
plt.legend(loc=(1.02,0))
plt.show(
plotGraph1 = data.pivot_table(index='Education',columns='Income_Bracket', values='NSP', aggfunc=np.sum)
plotGraph1.plot(kind='line',title='Graduates in High Income group,followed by Graduates in Medium Income Group.',figsize=(5,6))
plotGraph2 = data.pivot_table(index='AgeGroup',columns='Income_Bracket', values='NSP', aggfunc=np.sum)
plotGraph2.plot(kind='line',title='Seniors in High Income group,followed by 40-50 age group in Medium Income.',figsize=(5,6))
plotGraph3 = data.pivot_table(index='AgeGroup',columns='Income_Bracket', values=['NDP','NWP','NCP','NSP'],aggfunc=np.sum)
plotGraph3.plot(kind='line',title='Problem : Participation is less by people in the age group of less than 40 yrs.',figsize=(20,12))
plotGraph4 = data.pivot_table(columns='Income_Bracket', values='NWP', aggfunc=np.sum)
plotGraph4.plot(kind='bar',title='Sum of NWP by family status in different income bracket',figsize=(5,6))
plotGraph5 = data.pivot_table(index='Income_Bracket', values=['NWP','NCP','NSP'], aggfunc=np.sum)
plotGraph5.plot(kind='bar',title='Sum of WebPurchases,CataloguePurchases,StorePurchases According to income bracket',figsize=(5,6))
plotGraph6 = data.pivot_table(index='Education',values=['NWP','NCP','NSP'], aggfunc=np.sum)
plotGraph6.plot(kind='bar',title='Sum of WebPurchases,CataloguePurchases,StorePurchaces According to family status',figsize=(5,6))

Google Colab workfile(link): https://colab.research.google.com/drive/1vaLRcLldMWaH1w74lty1C4v8Zy7rQOqY?usp=sharing
  tableu public (link) = https://public.tableau.com/app/profile/tarun.sharma6977/viz/WindsorTechnocrats_FinalProject_DAB103/Dashboard1
  

