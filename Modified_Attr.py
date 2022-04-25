#Final code
import numpy as np
import pandas as pd
import joblib

dataset1 = pd.read_csv("Customer Sustainability DB.csv")

xt=dataset1.iloc[:,[0,2,3,4,5,6,7,8,9,10]]  #independent variables(13)
yt=dataset1.iloc[:,11]    #dependent variables

#RFC
from sklearn.model_selection import train_test_split
x_traint, x_testt, y_traint, y_testt = train_test_split(xt,yt, test_size= 0.20, random_state= 42)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_traint=sc_x.fit_transform(x_traint)
x_testt=sc_x.transform(x_testt)

from sklearn.ensemble import RandomForestClassifier
cls = RandomForestClassifier(criterion='entropy',n_estimators=300,random_state = 42)
cls.fit(x_traint,y_traint)

y_predt= cls.predict(x_testt)

print('ACCURACY is',cls.score(x_testt,y_testt)*100,'%')

filename = 'finalized_attr2.sav'
joblib.dump(cls, filename)
