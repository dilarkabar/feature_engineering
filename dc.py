import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

data=pd.read_csv("Medicaldataset.csv")
# İlk olarak bağımlı veriyi  string ifadeyi kategorik yapalım.Verimizde kişide kalp krizi varsa pozitif (1) yoksa (0) yap
le = LabelEncoder()
data['Result'] = le.fit_transform(data['Result'])   
print(data['Result'].unique())
print("*****************************************************************************************")
def data_describing(dataframe,col):
    print(dataframe.head())
    print("*******************")
    print(dataframe.tail())
    print("*******************")
    print(dataframe.shape)
    print("*******************")
    print(dataframe.info)
    print("*******************")
    print(dataframe.describe().T)
    print("*******************")
    print(dataframe.isnull().sum())
    print("*******************")
    v1=dataframe[col].value_counts()
    print(v1)
    print("*******************")
    oran=v1/len(dataframe)*100
    print(oran)
data_describing(data,"Result")

print("*****************************************************************************************")
def grap_columns(dataframe,cat=10,car=20):
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes=="0"]
    num_but_cat=[col for col in dataframe.columns if dataframe[col].dtypes!="0" and dataframe[col].nunique()< cat ]
    cat_but_car=[col for col in dataframe.columns if dataframe[col].dtypes=="0"  and dataframe[col].nunique()> cat  ]
    cat_cols=cat_cols + num_but_cat
    num_cols=[ col for col in cat_cols if col not in cat_but_car]

    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes!="0"]
    num_cols=[ col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}") # satır
    print(f"Variables: {dataframe.shape[1]}") # sütun
    print(f'cat_cols: {len(cat_cols)}') # categorik degişken sayısı
    print(f'num_cols: {len(num_cols)}') # numerik değişkenler
    print(f'cat_but_car: {len(cat_but_car)}') # categorik fakat kardinal
    print(f'num_but_cat: {len(num_but_cat)}') # numerik görünümlü kategorik
    
    return cat_cols, num_cols, cat_but_car 

cat_cols, num_cols, cat_but_car =grap_columns(data)

print("*****************************************************************************************")
#Aykırı değer analizi için iki farklı yöntem deneyeceğim
s=sns.boxplot(x=data["Age"])
s.set_title("Age  Box Plot",color="red")
plt.show()

def outliers_detection(dataframe,col,q1=0.05,q3=0.95):
    q1=dataframe[col].quantile(q1)
    q3=dataframe[col].quantile(q3)
    ıqr=q3-q1
    lowlimit=q1-(ıqr*1.5)
    uplimit=q3+ (ıqr*1.5)
    return lowlimit,uplimit
outliers_detection(data, "Age")

def thereare_outliers(dataframe,col):
    lowlimit,uplimit=outliers_detection(dataframe,col)
    if dataframe[(dataframe[col]<lowlimit)|(dataframe[col]>uplimit)].any(axis=None):
        return True
    else:
        return False
r=thereare_outliers(data,"Age")
print(r)
print("*****************************************************************************************")
#Aykırı değerlerden kurtulmak için iki farklı yöntemi deneyeceğim
#Silme
def remove_outliers(dataframe,col):
    lowlimit, uplimit = outliers_detection(dataframe, col)
    drpout=dataframe[~((dataframe[col]<lowlimit)|(dataframe[col]>uplimit))]
    return remove_outliers
for col in num_cols:
    new_df = remove_outliers(data, col)

# Baskılama Yöntemi (re-assignment with thresholds)
def re_assignment_with_thresholds(dataframe,col):
    lowlimit, uplimit = outliers_detection(dataframe, col)
    dataframe.loc[(dataframe[col] < lowlimit), col] = lowlimit
    dataframe.loc[(dataframe[col] > uplimit), col] = uplimit

cat_cols, num_cols, cat_but_car = grap_columns(data)

#Kontroller
for col in num_cols:
    print(col, thereare_outliers(data, col))

for col in num_cols:
    re_assignment_with_thresholds(data, col)
print("**********************************************")
for col in num_cols:
    print(col, thereare_outliers(data, col))

print("*****************************************************************************************")
# Feature Scaling (Özellik Ölçeklendirme)
stndrt=StandardScaler()
data["Age_StandardScaler"] = stndrt.fit_transform(data[["Age"]])
print(data.head())

RbScaler=RobustScaler()
data["Age_RobustScaler"]=RbScaler.fit_transform(data[["Age"]])
print(data.head())

mms = MinMaxScaler()
data["Age_Minmax_Scaler"] = mms.fit_transform(data[["Age"]])
print(data.head())
Age_cols = [col for col in data.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

 
    if plot:
        dataframe[numerical_col].hist(bins=20,color="red")
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in Age_cols:
    num_summary(data, col, plot=True)

































