# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
REG NO: 212224040239
NAME  : PRANAV BHARGAV M
```
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/eda998db-493f-4cd0-85a4-f7a6f85f2d9b)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/33d986de-47c0-4add-93d1-4c8ba0fc463e)
```
max_val=np.max(np.abs(df[['Height','Weight']]))
max_val
```
![image](https://github.com/user-attachments/assets/7bdb978b-8be6-4351-a04b-a3208a3d816c)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/c5a72bea-0fd3-41ed-9bca-e8479aa827b7)
```
from sklearn.preprocessing import Normalizer
nm=Normalizer()
df[['Height','Weight']]=nm.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/93597b17-65c2-4a98-a8d1-12a76326b5c1)
```
from sklearn.preprocessing import MaxAbsScaler
mas=MaxAbsScaler()
df[['Height','Weight']]=mas.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/e365def9-f45d-481f-8d47-88d0249b7d1f)
```
from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
df[['Height','Weight']]=rs.fit_transform(df[['Height','Weight']])
df.head(5)
```
![image](https://github.com/user-attachments/assets/38ee653b-2f3d-43df-8929-3499dc245388)
```
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/7bd5ba58-89f3-4ebf-984f-c37c46096f72)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
contingency_table
```
![image](https://github.com/user-attachments/assets/44a45dd6-3d8c-4c80-a9ed-94fae6ce8cc5)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print('Chi-square statistic:',chi2)
print('p-value:',p)
```
![image](https://github.com/user-attachments/assets/d8241742-dd10-46c7-8504-71dc5b02d160)
```
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={'Feature1' : [1,2,3,4,5],'Feature2' : ['A','B','C','A','B'],'Feature3' : [0,1,1,0,1],'Target': [0,1,1,0,1]}
df=pd.DataFrame(data)
df
```
![image](https://github.com/user-attachments/assets/ac857b5c-3b02-4884-afb6-e5fd3a55eb10)
```
x=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
print('Selected features:',x_new)
```
![image](https://github.com/user-attachments/assets/1c8b6740-fdaf-4019-8ae8-ca704b0602c6)
```
selectedFeatureIndices=selector.get_support(indices=True)
selectedFeatures=x.columns[selectedFeatureIndices]
print('Selected features:',selectedFeatures)
```
![image](https://github.com/user-attachments/assets/12cffb39-1bc6-447e-9d57-556feb02d888)
# RESULT:
```
Feature Scaling and Feature Selection process has been successfully performed on the data set.
```
