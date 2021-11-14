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


