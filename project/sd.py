import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sea
import joblib
data=pd.read_csv("C:\\Users\\payal\\Desktop\\place.csv")
#print(data.to_string())
x=data[["cgpa"]]
y=data[["placement"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression
f=LinearRegression()
f.fit(x_train,y_train)
d=f.predict([[6.89]])
r=f.score(x_test,y_test)
print("model is score is ",r*100)
joblib.dump(f, "linear")


print(d)
import joblib


dataset=sea.load_dataset("iris")
x=dataset.iloc[:,[0,1,2,3]]
x.ndim
y=dataset[["species"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.svm import SVC
dc=SVC()
dc.fit(x_train,y_train)
l=dc.predict([[5.1,3.5,1.4,0.2]])
print(l)
joblib.dump(dc, "decision")

load = joblib.load("decision")
# LASSO REGRESSION MODEL
data=pd.read_csv("C:\\Users\\payal\\Desktop\\a.csv")
#print(data)
x=data[["number_courses","time_study"]]
y=data[["Marks"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import Lasso
ab=Lasso()
ab.fit(x_train,y_train)
v=ab.score(x_test,y_test)
print(v)
joblib.dump(ab, "lasso")
#RIDGE REGRESSION MODEL
data=pd.read_csv("C:\\Users\\payal\\Desktop\\a.csv")
#print(data)
x=data[["number_courses","time_study"]]
y=data[["Marks"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import Ridge
ad=Ridge()
ad.fit(x_train,y_train)
y=ad.score(x_test,y_test)
print(y)
joblib.dump(ab, "ridge")
#DECISION TREE ALGORITHM
def encoder(data):
    import pandas as pd 
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder
    la=OrdinalEncoder()
    v=la.fit_transform(data)
    data=pd.DataFrame(v,columns=data.columns)
    return data
data=sea.load_dataset("diamonds")
print(data)
x