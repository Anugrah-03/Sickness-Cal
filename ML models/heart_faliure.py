import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
data = pd.read_csv('Datasets\heart_failure_clinical_records_dataset.csv')
X = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]
y = data['DEATH_EVENT']
ImputedModule = SimpleImputer(missing_values = np.nan, strategy ='mean')
ImputedX = ImputedModule.fit(X)
X = ImputedX.transform(X)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, shuffle =True)
rf = RandomForestClassifier()
rf.fit(X_train,y_train) 
print(accuracy_score(rf.predict(X_test),y_test))
print(rf.predict([[50,1,168,0,38,1,276000,1.1,137,1,0,11]]))
pickle.dump(rf,open("heart_failure.pkl","wb"))
