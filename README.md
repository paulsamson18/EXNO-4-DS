## EXNO:4-DS
## AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

## ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the values vary within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns

df=pd.read_csv("/content/bmi.csv")
df1=pd.read_csv("/content/bmi.csv")
df2=pd.read_csv("/content/bmi.csv")

df.head()
```
![head_4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/32e5f3d7-e75b-4ba5-a496-3112d8f481b2)

```
df.dropna()
```
![dropna-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/1131a2ca-bbe6-4746-8a9e-937d3f24b5a9)

```
max_vals = np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![max-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/6ba78607-d401-489d-8827-a56d460d7905)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![stdscaler-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/da45021a-41a7-47d2-9af8-cedb5d6db610)


```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![minmax-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/8aca97b7-0d31-4fed-98b4-f52906675659)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
![normalizer-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/1ef0c6b8-914c-4437-ab56-ad5360281379)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![maxabs-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/8278f50e-ef00-4e58-abe6-141fd23aa3e2)

```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![robust-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/9db3bd41-cec6-4cb0-9feb-4ed85ad5c1fa)

```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![conf-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/a47d809a-8ff6-43eb-bd0d-dafcf0f1d2e6)

```
data.isnull().sum()
```
![isnull-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/d45f1996-a241-46a5-94b3-24d12c7c13e4)

```

data2=data.dropna(axis=0)
data2
```
![dropnaaxis-4](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/d315acc2-337e-4ce9-bb82-b05e0cec7313)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/5eb03112-9f83-464d-a319-ef6e83deb7c0)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/7a399c75-688f-4391-9fa8-4c5909669f96)

```
data2
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/d21c85eb-bef9-46c1-8e47-d746ad9e6418)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/15f3e5a8-e060-4d97-846d-350072e067c4)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/804aeff4-a271-4491-8834-4471f49fd4f1)

```
features = list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/120db1bd-c5ec-4753-aad3-c67577d11821)

```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/3e98a30f-2b8a-4c74-8300-6ffcdfe0675d)

```
x=new_data[features].values
print(x)
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/ca293595-6c8d-4fb7-a418-0ce8cc2f325c)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]

}
df=pd.DataFrame(data)
df
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/2cdeed50-889f-432a-aeed-35cf333a59f5)

```
X=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/67033fae-b2b9-4b11-80cd-f87aaa2d7a59)

```
selected_feature_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/da36751c-c637-4aef-b1d4-ed70c4ff73e3)

```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/c81452ff-0163-49ee-aa0b-214d976d8cb9)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/1cdf9bb8-86ba-445f-8f3e-1dfa3439ed34)

```
chi2, p, _, _=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/BALUREDDYVELAYUDHAMGOWTHAM/EXNO-4-DS/assets/119559905/3527df64-8333-42ef-b737-1d6dba433d2d)
# RESULT:
Thus to read the given data and perform Feature Scaling and Feature Selection process was performed successfully.
