# Ex-06-Feature-Transformation
AIM
To read the given data and perform Feature Transformation process and save the data to a file.

EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

ALGORITHM
STEP 1
Read the given Data

STEP 2
Clean the Data Set using Data Cleaning Process

STEP 3
Apply Feature Transformation techniques to all the features of the data set

STEP 4
Save the data to the file

CODE
```
Name : NIVETHA K
Register Number : 212222230102
**Feature Transformation - Data_to_Transform.csv**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()
df1 = df.copy()
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)
sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()
```
OUPUT:
Feature Transformation - Data_to_Transform.csv
![o1](https://user-images.githubusercontent.com/119559844/232683181-05a217d7-d482-4108-a547-727ffb91ed34.png)
![o2](https://user-images.githubusercontent.com/119559844/232683197-39e23542-4447-459e-be35-0c79e2346785.png)
![o3](https://user-images.githubusercontent.com/119559844/232683203-dd902161-e0d1-4cd1-ada9-199d6f5be446.png)
![o4](https://user-images.githubusercontent.com/119559844/232683219-cc3b0358-2edc-40dd-8a84-6283fbed4a39.png)
![o5](https://user-images.githubusercontent.com/119559844/232683226-d6543bae-114f-42d9-ab51-cfc41c4c25ef.png)
![o6](https://user-images.githubusercontent.com/119559844/232683251-4ad1dce7-1503-4a5f-b8c5-589e7ab1075f.png)
![o7](https://user-images.githubusercontent.com/119559844/232683260-afb9bfbd-4805-4c00-8a24-e8568d6f7960.png)
![o8](https://user-images.githubusercontent.com/119559844/232683270-597ef95e-c8a3-45bf-84f2-48f6a2a76d40.png)
![o9](https://user-images.githubusercontent.com/119559844/232683278-ae1f1776-0dac-4f19-bfbf-201ed792fe01.png)
![o10](https://user-images.githubusercontent.com/119559844/232683286-a9d5b48d-c85e-41bc-8d01-ca16de25bb5b.png)
![o11](https://user-images.githubusercontent.com/119559844/232683293-dbe5f8a9-d5a3-4312-8380-6029714c55b5.png)

RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully
