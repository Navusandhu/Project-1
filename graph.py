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

tableu public (link) = https://public.tableau.com/app/profile/tarun.sharma6977/viz/WindsorTechnocrats_FinalProject_DAB103/Dashboard1
  
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
Collecting tabpy
  Downloading tabpy-2.4.0-py2.py3-none-any.whl (110 kB)
     |████████████████████████████████| 110 kB 9.3 MB/s 
Requirement already satisfied: coveralls in /usr/local/lib/python3.7/dist-packages (from tabpy) (0.5)
Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from tabpy) (1.4.1)
Collecting configparser
  Downloading configparser-5.2.0-py3-none-any.whl (19 kB)
Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from tabpy) (2.23.0)
Collecting simplejson
  Downloading simplejson-3.17.6-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (130 kB)
     |████████████████████████████████| 130 kB 22.4 MB/s 
Requirement already satisfied: nltk in /usr/local/lib/python3.7/dist-packages (from tabpy) (3.2.5)
Requirement already satisfied: tornado in /usr/local/lib/python3.7/dist-packages (from tabpy) (5.1.1)
Requirement already satisfied: coverage in /usr/local/lib/python3.7/dist-packages (from tabpy) (3.7.1)
Requirement already satisfied: cloudpickle in /usr/local/lib/python3.7/dist-packages (from tabpy) (1.3.0)
Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from tabpy) (0.16.0)
Collecting mock
  Downloading mock-4.0.3-py3-none-any.whl (28 kB)
Collecting genson
  Downloading genson-1.2.2.tar.gz (34 kB)
Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from tabpy) (1.24.3)
Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from tabpy) (1.1.5)
Requirement already satisfied: textblob in /usr/local/lib/python3.7/dist-packages (from tabpy) (0.15.3)
Requirement already satisfied: jsonschema in /usr/local/lib/python3.7/dist-packages (from tabpy) (2.6.0)
Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from tabpy) (1.19.5)
Collecting pytest-cov
  Downloading pytest_cov-3.0.0-py3-none-any.whl (20 kB)
Requirement already satisfied: pytest in /usr/local/lib/python3.7/dist-packages (from tabpy) (3.6.4)
Requirement already satisfied: docopt in /usr/local/lib/python3.7/dist-packages (from tabpy) (0.6.2)
Collecting hypothesis
  Downloading hypothesis-6.30.0-py3-none-any.whl (388 kB)
     |████████████████████████████████| 388 kB 21.2 MB/s 
Requirement already satisfied: sklearn in /usr/local/lib/python3.7/dist-packages (from tabpy) (0.0)
Collecting twisted
  Downloading Twisted-21.7.0-py3-none-any.whl (3.1 MB)
     |████████████████████████████████| 3.1 MB 50.7 MB/s 
Collecting pyopenssl
  Downloading pyOpenSSL-21.0.0-py2.py3-none-any.whl (55 kB)
     |████████████████████████████████| 55 kB 3.8 MB/s 
Requirement already satisfied: PyYAML>=3.10 in /usr/local/lib/python3.7/dist-packages (from coveralls->tabpy) (3.13)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->tabpy) (2021.10.8)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->tabpy) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->tabpy) (3.0.4)
Requirement already satisfied: attrs>=19.2.0 in /usr/local/lib/python3.7/dist-packages (from hypothesis->tabpy) (21.2.0)
Requirement already satisfied: sortedcontainers<3.0.0,>=2.1.0 in /usr/local/lib/python3.7/dist-packages (from hypothesis->tabpy) (2.4.0)
Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from nltk->tabpy) (1.15.0)
Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->tabpy) (2.8.2)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas->tabpy) (2018.9)
Collecting cryptography>=3.3
  Downloading cryptography-36.0.0-cp36-abi3-manylinux_2_24_x86_64.whl (3.6 MB)
     |████████████████████████████████| 3.6 MB 59.6 MB/s 
Requirement already satisfied: cffi>=1.12 in /usr/local/lib/python3.7/dist-packages (from cryptography>=3.3->pyopenssl->tabpy) (1.15.0)
Requirement already satisfied: pycparser in /usr/local/lib/python3.7/dist-packages (from cffi>=1.12->cryptography>=3.3->pyopenssl->tabpy) (2.21)
Requirement already satisfied: atomicwrites>=1.0 in /usr/local/lib/python3.7/dist-packages (from pytest->tabpy) (1.4.0)
Requirement already satisfied: py>=1.5.0 in /usr/local/lib/python3.7/dist-packages (from pytest->tabpy) (1.11.0)
Requirement already satisfied: pluggy<0.8,>=0.5 in /usr/local/lib/python3.7/dist-packages (from pytest->tabpy) (0.7.1)
Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from pytest->tabpy) (57.4.0)
Requirement already satisfied: more-itertools>=4.0.0 in /usr/local/lib/python3.7/dist-packages (from pytest->tabpy) (8.12.0)
Collecting coverage[toml]>=5.2.1
  Downloading coverage-6.2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (213 kB)
     |████████████████████████████████| 213 kB 56.2 MB/s 
Collecting pytest
  Downloading pytest-6.2.5-py3-none-any.whl (280 kB)
     |████████████████████████████████| 280 kB 48.8 MB/s 
Collecting coverage[toml]>=5.2.1
  Downloading coverage-6.1.2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (213 kB)
     |████████████████████████████████| 213 kB 57.3 MB/s 
  Downloading coverage-6.1.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (213 kB)
     |████████████████████████████████| 213 kB 56.9 MB/s 
  Downloading coverage-6.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (213 kB)
     |████████████████████████████████| 213 kB 57.0 MB/s 
  Downloading coverage-6.0.2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (253 kB)
     |████████████████████████████████| 253 kB 64.6 MB/s 
  Downloading coverage-6.0.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (252 kB)
     |████████████████████████████████| 252 kB 50.9 MB/s 
  Downloading coverage-6.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (252 kB)
     |████████████████████████████████| 252 kB 52.1 MB/s 
  Downloading coverage-5.5-cp37-cp37m-manylinux2010_x86_64.whl (242 kB)
     |████████████████████████████████| 242 kB 47.3 MB/s 
  Downloading coverage-5.4-cp37-cp37m-manylinux2010_x86_64.whl (242 kB)
     |████████████████████████████████| 242 kB 58.9 MB/s 
  Downloading coverage-5.3.1-cp37-cp37m-manylinux2010_x86_64.whl (242 kB)
     |████████████████████████████████| 242 kB 71.3 MB/s 
  Downloading coverage-5.3-cp37-cp37m-manylinux1_x86_64.whl (229 kB)
     |████████████████████████████████| 229 kB 68.9 MB/s 
  Downloading coverage-5.2.1-cp37-cp37m-manylinux1_x86_64.whl (229 kB)
     |████████████████████████████████| 229 kB 51.5 MB/s 
INFO: pip is looking at multiple versions of pytest-cov to determine which version is compatible with other requirements. This could take a while.
Collecting pytest-cov
  Downloading pytest_cov-2.12.1-py2.py3-none-any.whl (20 kB)
  Downloading pytest_cov-2.12.0-py2.py3-none-any.whl (20 kB)
  Downloading pytest_cov-2.11.1-py2.py3-none-any.whl (20 kB)
  Downloading pytest_cov-2.11.0-py2.py3-none-any.whl (20 kB)
  Downloading pytest_cov-2.10.1-py2.py3-none-any.whl (19 kB)
  Downloading pytest_cov-2.10.0-py2.py3-none-any.whl (19 kB)
  Downloading pytest_cov-2.9.0-py2.py3-none-any.whl (19 kB)
  Downloading pytest_cov-2.8.1-py2.py3-none-any.whl (18 kB)
  Downloading pytest_cov-2.8.0-py2.py3-none-any.whl (18 kB)
  Downloading pytest_cov-2.7.1-py2.py3-none-any.whl (17 kB)
  Downloading pytest_cov-2.7.0-py2.py3-none-any.whl (17 kB)
  Downloading pytest_cov-2.6.1-py2.py3-none-any.whl (16 kB)
  Downloading pytest_cov-2.6.0-py2.py3-none-any.whl (14 kB)
  Downloading pytest_cov-2.5.1-py2.py3-none-any.whl (21 kB)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.7/dist-packages (from sklearn->tabpy) (1.0.1)
Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->sklearn->tabpy) (1.1.0)
Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn->sklearn->tabpy) (3.0.0)
Collecting hyperlink>=17.1.1
  Downloading hyperlink-21.0.0-py2.py3-none-any.whl (74 kB)
     |████████████████████████████████| 74 kB 3.3 MB/s 
Collecting incremental>=21.3.0
  Downloading incremental-21.3.0-py2.py3-none-any.whl (15 kB)
Requirement already satisfied: typing-extensions>=3.6.5 in /usr/local/lib/python3.7/dist-packages (from twisted->tabpy) (3.10.0.2)
Collecting zope.interface>=4.4.2
  Downloading zope.interface-5.4.0-cp37-cp37m-manylinux2010_x86_64.whl (251 kB)
     |████████████████████████████████| 251 kB 62.0 MB/s 
Collecting constantly>=15.1
  Downloading constantly-15.1.0-py2.py3-none-any.whl (7.9 kB)
Collecting Automat>=0.8.0
  Downloading Automat-20.2.0-py2.py3-none-any.whl (31 kB)
Building wheels for collected packages: genson
  Building wheel for genson (setup.py) ... done
  Created wheel for genson: filename=genson-1.2.2-py2.py3-none-any.whl size=21291 sha256=77ac748048ba54228715388edcd8371d0cb12f46516e86dc55fac7e4d006aa38
  Stored in directory: /root/.cache/pip/wheels/2e/34/be/0194d05d18bc4695b5c4969178790d535bdd23eabdb9d3b1e3
Successfully built genson
Installing collected packages: zope.interface, incremental, hyperlink, cryptography, constantly, Automat, twisted, simplejson, pytest-cov, pyopenssl, mock, hypothesis, genson, configparser, tabpy
Successfully installed Automat-20.2.0 configparser-5.2.0 constantly-15.1.0 cryptography-36.0.0 genson-1.2.2 hyperlink-21.0.0 hypothesis-6.30.0 incremental-21.3.0 mock-4.0.3 pyopenssl-21.0.0 pytest-cov-2.5.1 simplejson-3.17.6 tabpy-2.4.0 twisted-21.7.0 zope.interface-5.4.0
ERROR: unknown command "istall" - maybe you meant "install"
ERROR: unknown command "istall" - maybe you meant "install"
ERROR: unknown command "istall" - maybe you meant "install"
Collecting researchpy
  Downloading researchpy-0.3.2-py3-none-any.whl (15 kB)
Requirement already satisfied: statsmodels in /usr/local/lib/python3.7/dist-packages (from researchpy) (0.10.2)
Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from researchpy) (1.4.1)
Requirement already satisfied: numpy in /usr/local/lib/python3.7/dist-packages (from researchpy) (1.19.5)
Requirement already satisfied: patsy in /usr/local/lib/python3.7/dist-packages (from researchpy) (0.5.2)
Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from researchpy) (1.1.5)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas->researchpy) (2018.9)
Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->researchpy) (2.8.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->researchpy) (1.15.0)
Installing collected packages: researchpy
Successfully installed researchpy-0.3.2
Collecting quick-eda
  Downloading quick_eda-0.1.7-py3-none-any.whl (4.7 kB)
Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from quick-eda) (1.1.5)
Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas->quick-eda) (2.8.2)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas->quick-eda) (2018.9)
Requirement already satisfied: numpy>=1.15.4 in /usr/local/lib/python3.7/dist-packages (from pandas->quick-eda) (1.19.5)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas->quick-eda) (1.15.0)
Installing collected packages: quick-eda
Successfully installed quick-eda-0.1.7
Collecting dataprep
  Downloading dataprep-0.4.1-py3-none-any.whl (3.5 MB)
     |████████████████████████████████| 3.5 MB 9.4 MB/s 
Collecting regex<2021.0.0,>=2020.10.15
  Downloading regex-2020.11.13-cp37-cp37m-manylinux2014_x86_64.whl (719 kB)
     |████████████████████████████████| 719 kB 45.4 MB/s 
Collecting pydantic<2.0,>=1.6
  Downloading pydantic-1.8.2-cp37-cp37m-manylinux2014_x86_64.whl (10.1 MB)
     |████████████████████████████████| 10.1 MB 43.2 MB/s 
Collecting wordcloud<2.0,>=1.8
  Downloading wordcloud-1.8.1-cp37-cp37m-manylinux1_x86_64.whl (366 kB)
     |████████████████████████████████| 366 kB 33.7 MB/s 
Requirement already satisfied: pandas<2.0,>=1.1 in /usr/local/lib/python3.7/dist-packages (from dataprep) (1.1.5)
Requirement already satisfied: flask<2.0.0,>=1.1.4 in /usr/local/lib/python3.7/dist-packages (from dataprep) (1.1.4)
Collecting jsonpath-ng<2.0,>=1.5
  Downloading jsonpath_ng-1.5.3-py3-none-any.whl (29 kB)
Collecting varname<0.9.0,>=0.8.1
  Downloading varname-0.8.1-py3-none-any.whl (20 kB)
Collecting nltk<4.0,>=3.5
  Downloading nltk-3.6.5-py3-none-any.whl (1.5 MB)
     |████████████████████████████████| 1.5 MB 46.1 MB/s 
Collecting python-stdnum<2.0,>=1.16
  Downloading python_stdnum-1.17-py2.py3-none-any.whl (943 kB)
     |████████████████████████████████| 943 kB 45.2 MB/s 
Requirement already satisfied: ipywidgets<8.0,>=7.5 in /usr/local/lib/python3.7/dist-packages (from dataprep) (7.6.5)
Collecting usaddress<0.6.0,>=0.5.10
  Downloading usaddress-0.5.10-py2.py3-none-any.whl (63 kB)
     |████████████████████████████████| 63 kB 2.2 MB/s 
Collecting levenshtein<0.13.0,>=0.12.0
  Downloading levenshtein-0.12.0-cp37-cp37m-manylinux1_x86_64.whl (158 kB)
     |████████████████████████████████| 158 kB 50.5 MB/s 
Requirement already satisfied: tqdm<5.0,>=4.48 in /usr/local/lib/python3.7/dist-packages (from dataprep) (4.62.3)
Requirement already satisfied: numpy<2,>=1 in /usr/local/lib/python3.7/dist-packages (from dataprep) (1.19.5)
Collecting aiohttp<4.0,>=3.6
  Downloading aiohttp-3.8.1-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.1 MB)
     |████████████████████████████████| 1.1 MB 38.4 MB/s 
Requirement already satisfied: bokeh<3,>=2 in /usr/local/lib/python3.7/dist-packages (from dataprep) (2.3.3)
Collecting flask_cors<4.0.0,>=3.0.10
  Downloading Flask_Cors-3.0.10-py2.py3-none-any.whl (14 kB)
Collecting metaphone<0.7,>=0.6
  Downloading Metaphone-0.6.tar.gz (14 kB)
Collecting dask[array,dataframe,delayed]<3.0,>=2.25
  Downloading dask-2.30.0-py3-none-any.whl (848 kB)
     |████████████████████████████████| 848 kB 49.0 MB/s 
Requirement already satisfied: jinja2<3.0,>=2.11 in /usr/local/lib/python3.7/dist-packages (from dataprep) (2.11.3)
Requirement already satisfied: bottleneck<2.0,>=1.3 in /usr/local/lib/python3.7/dist-packages (from dataprep) (1.3.2)
Requirement already satisfied: scipy<2,>=1 in /usr/local/lib/python3.7/dist-packages (from dataprep) (1.4.1)
Collecting frozenlist>=1.1.1
  Downloading frozenlist-1.2.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (192 kB)
     |████████████████████████████████| 192 kB 51.3 MB/s 
Collecting aiosignal>=1.1.2
  Downloading aiosignal-1.2.0-py3-none-any.whl (8.2 kB)
Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp<4.0,>=3.6->dataprep) (2.0.8)
Collecting asynctest==0.13.0
  Downloading asynctest-0.13.0-py3-none-any.whl (26 kB)
Collecting async-timeout<5.0,>=4.0.0a3
  Downloading async_timeout-4.0.1-py3-none-any.whl (5.7 kB)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp<4.0,>=3.6->dataprep) (21.2.0)
Collecting yarl<2.0,>=1.0
  Downloading yarl-1.7.2-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (271 kB)
     |████████████████████████████████| 271 kB 59.7 MB/s 
Requirement already satisfied: typing-extensions>=3.7.4 in /usr/local/lib/python3.7/dist-packages (from aiohttp<4.0,>=3.6->dataprep) (3.10.0.2)
Collecting multidict<7.0,>=4.5
  Downloading multidict-5.2.0-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (160 kB)
     |████████████████████████████████| 160 kB 57.8 MB/s 
Requirement already satisfied: packaging>=16.8 in /usr/local/lib/python3.7/dist-packages (from bokeh<3,>=2->dataprep) (21.3)
Requirement already satisfied: PyYAML>=3.10 in /usr/local/lib/python3.7/dist-packages (from bokeh<3,>=2->dataprep) (3.13)
Requirement already satisfied: pillow>=7.1.0 in /usr/local/lib/python3.7/dist-packages (from bokeh<3,>=2->dataprep) (7.1.2)
Requirement already satisfied: tornado>=5.1 in /usr/local/lib/python3.7/dist-packages (from bokeh<3,>=2->dataprep) (5.1.1)
Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.7/dist-packages (from bokeh<3,>=2->dataprep) (2.8.2)
Collecting fsspec>=0.6.0
  Downloading fsspec-2021.11.1-py3-none-any.whl (132 kB)
     |████████████████████████████████| 132 kB 60.4 MB/s 
Requirement already satisfied: toolz>=0.8.2 in /usr/local/lib/python3.7/dist-packages (from dask[array,dataframe,delayed]<3.0,>=2.25->dataprep) (0.11.2)
Collecting partd>=0.3.10
  Downloading partd-1.2.0-py3-none-any.whl (19 kB)
Requirement already satisfied: cloudpickle>=0.2.2 in /usr/local/lib/python3.7/dist-packages (from dask[array,dataframe,delayed]<3.0,>=2.25->dataprep) (1.3.0)
Requirement already satisfied: click<8.0,>=5.1 in /usr/local/lib/python3.7/dist-packages (from flask<2.0.0,>=1.1.4->dataprep) (7.1.2)
Requirement already satisfied: Werkzeug<2.0,>=0.15 in /usr/local/lib/python3.7/dist-packages (from flask<2.0.0,>=1.1.4->dataprep) (1.0.1)
Requirement already satisfied: itsdangerous<2.0,>=0.24 in /usr/local/lib/python3.7/dist-packages (from flask<2.0.0,>=1.1.4->dataprep) (1.1.0)
Requirement already satisfied: Six in /usr/local/lib/python3.7/dist-packages (from flask_cors<4.0.0,>=3.0.10->dataprep) (1.15.0)
Requirement already satisfied: nbformat>=4.2.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets<8.0,>=7.5->dataprep) (5.1.3)
Requirement already satisfied: ipython-genutils~=0.2.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets<8.0,>=7.5->dataprep) (0.2.0)
Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets<8.0,>=7.5->dataprep) (1.0.2)
Requirement already satisfied: traitlets>=4.3.1 in /usr/local/lib/python3.7/dist-packages (from ipywidgets<8.0,>=7.5->dataprep) (5.1.1)
Requirement already satisfied: ipython>=4.0.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets<8.0,>=7.5->dataprep) (5.5.0)
Requirement already satisfied: widgetsnbextension~=3.5.0 in /usr/local/lib/python3.7/dist-packages (from ipywidgets<8.0,>=7.5->dataprep) (3.5.2)
Requirement already satisfied: ipykernel>=4.5.1 in /usr/local/lib/python3.7/dist-packages (from ipywidgets<8.0,>=7.5->dataprep) (4.10.1)
Requirement already satisfied: jupyter-client in /usr/local/lib/python3.7/dist-packages (from ipykernel>=4.5.1->ipywidgets<8.0,>=7.5->dataprep) (5.3.5)
Requirement already satisfied: pygments in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets<8.0,>=7.5->dataprep) (2.6.1)
Requirement already satisfied: pickleshare in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets<8.0,>=7.5->dataprep) (0.7.5)
Requirement already satisfied: prompt-toolkit<2.0.0,>=1.0.4 in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets<8.0,>=7.5->dataprep) (1.0.18)
Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets<8.0,>=7.5->dataprep) (57.4.0)
Requirement already satisfied: decorator in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets<8.0,>=7.5->dataprep) (4.4.2)
Requirement already satisfied: simplegeneric>0.8 in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets<8.0,>=7.5->dataprep) (0.8.1)
Requirement already satisfied: pexpect in /usr/local/lib/python3.7/dist-packages (from ipython>=4.0.0->ipywidgets<8.0,>=7.5->dataprep) (4.8.0)
Requirement already satisfied: MarkupSafe>=0.23 in /usr/local/lib/python3.7/dist-packages (from jinja2<3.0,>=2.11->dataprep) (2.0.1)
Collecting ply
  Downloading ply-3.11-py2.py3-none-any.whl (49 kB)
     |████████████████████████████████| 49 kB 6.2 MB/s 
Requirement already satisfied: jupyter-core in /usr/local/lib/python3.7/dist-packages (from nbformat>=4.2.0->ipywidgets<8.0,>=7.5->dataprep) (4.9.1)
Requirement already satisfied: jsonschema!=2.5.0,>=2.4 in /usr/local/lib/python3.7/dist-packages (from nbformat>=4.2.0->ipywidgets<8.0,>=7.5->dataprep) (2.6.0)
Collecting nltk<4.0,>=3.5
  Downloading nltk-3.6.3-py3-none-any.whl (1.5 MB)
     |████████████████████████████████| 1.5 MB 43.3 MB/s 
Requirement already satisfied: joblib in /usr/local/lib/python3.7/dist-packages (from nltk<4.0,>=3.5->dataprep) (1.1.0)
Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging>=16.8->bokeh<3,>=2->dataprep) (3.0.6)
Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas<2.0,>=1.1->dataprep) (2018.9)
Collecting locket
  Downloading locket-0.2.1-py2.py3-none-any.whl (4.1 kB)
Requirement already satisfied: wcwidth in /usr/local/lib/python3.7/dist-packages (from prompt-toolkit<2.0.0,>=1.0.4->ipython>=4.0.0->ipywidgets<8.0,>=7.5->dataprep) (0.2.5)
Collecting probableparsing
  Downloading probableparsing-0.0.1-py2.py3-none-any.whl (3.1 kB)
Collecting python-crfsuite>=0.7
  Downloading python_crfsuite-0.9.7-cp37-cp37m-manylinux1_x86_64.whl (743 kB)
     |████████████████████████████████| 743 kB 71.4 MB/s 
Requirement already satisfied: future>=0.14 in /usr/local/lib/python3.7/dist-packages (from usaddress<0.6.0,>=0.5.10->dataprep) (0.16.0)
Collecting executing
  Downloading executing-0.8.2-py2.py3-none-any.whl (16 kB)
Collecting asttokens<3.0.0,>=2.0.0
  Downloading asttokens-2.0.5-py2.py3-none-any.whl (20 kB)
Collecting pure_eval<1.0.0
  Downloading pure_eval-0.2.1-py3-none-any.whl (11 kB)
Requirement already satisfied: notebook>=4.4.1 in /usr/local/lib/python3.7/dist-packages (from widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (5.3.1)
Requirement already satisfied: terminado>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (0.12.1)
Requirement already satisfied: nbconvert in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (5.6.1)
Requirement already satisfied: Send2Trash in /usr/local/lib/python3.7/dist-packages (from notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (1.8.0)
Requirement already satisfied: pyzmq>=13 in /usr/local/lib/python3.7/dist-packages (from jupyter-client->ipykernel>=4.5.1->ipywidgets<8.0,>=7.5->dataprep) (22.3.0)
Requirement already satisfied: ptyprocess in /usr/local/lib/python3.7/dist-packages (from terminado>=0.8.1->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (0.7.0)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from wordcloud<2.0,>=1.8->dataprep) (3.2.2)
Requirement already satisfied: idna>=2.0 in /usr/local/lib/python3.7/dist-packages (from yarl<2.0,>=1.0->aiohttp<4.0,>=3.6->dataprep) (2.10)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib->wordcloud<2.0,>=1.8->dataprep) (1.3.2)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib->wordcloud<2.0,>=1.8->dataprep) (0.11.0)
Requirement already satisfied: bleach in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (4.1.0)
Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (0.8.4)
Requirement already satisfied: testpath in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (0.5.0)
Requirement already satisfied: defusedxml in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (0.7.1)
Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (0.3)
Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.7/dist-packages (from nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (1.5.0)
Requirement already satisfied: webencodings in /usr/local/lib/python3.7/dist-packages (from bleach->nbconvert->notebook>=4.4.1->widgetsnbextension~=3.5.0->ipywidgets<8.0,>=7.5->dataprep) (0.5.1)
Building wheels for collected packages: metaphone
  Building wheel for metaphone (setup.py) ... done
  Created wheel for metaphone: filename=Metaphone-0.6-py3-none-any.whl size=13919 sha256=e9e77ae5e73390e64ad83644b14009b7907f7dfcf3d5c61e12bd7153b20b7725
  Stored in directory: /root/.cache/pip/wheels/1d/a8/cb/6f8902aa5457bd71344e00665c230e9c45255b3f57f2194a0f
Successfully built metaphone
Installing collected packages: multidict, locket, frozenlist, yarl, regex, python-crfsuite, pure-eval, probableparsing, ply, partd, fsspec, executing, dask, asynctest, async-timeout, asttokens, aiosignal, wordcloud, varname, usaddress, python-stdnum, pydantic, nltk, metaphone, levenshtein, jsonpath-ng, flask-cors, aiohttp, dataprep
  Attempting uninstall: regex
    Found existing installation: regex 2019.12.20
    Uninstalling regex-2019.12.20:
      Successfully uninstalled regex-2019.12.20
  Attempting uninstall: dask
    Found existing installation: dask 2.12.0
    Uninstalling dask-2.12.0:
      Successfully uninstalled dask-2.12.0
  Attempting uninstall: wordcloud
    Found existing installation: wordcloud 1.5.0
    Uninstalling wordcloud-1.5.0:
      Successfully uninstalled wordcloud-1.5.0
  Attempting uninstall: nltk
    Found existing installation: nltk 3.2.5
    Uninstalling nltk-3.2.5:
      Successfully uninstalled nltk-3.2.5
Successfully installed aiohttp-3.8.1 aiosignal-1.2.0 asttokens-2.0.5 async-timeout-4.0.1 asynctest-0.13.0 dask-2.30.0 dataprep-0.4.1 executing-0.8.2 flask-cors-3.0.10 frozenlist-1.2.0 fsspec-2021.11.1 jsonpath-ng-1.5.3 levenshtein-0.12.0 locket-0.2.1 metaphone-0.6 multidict-5.2.0 nltk-3.6.3 partd-1.2.0 ply-3.11 probableparsing-0.0.1 pure-eval-0.2.1 pydantic-1.8.2 python-crfsuite-0.9.7 python-stdnum-1.17 regex-2020.11.13 usaddress-0.5.10 varname-0.8.1 wordcloud-1.8.1 yarl-1.7.2
from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
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

