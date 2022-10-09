import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import pickle
from sklearn.metrics import accuracy_score
data = pd.read_csv('Datasets\\alzehimer.csv')
data = data.loc[data['Visit']==1]         
data = data.reset_index(drop=True)         
data = data[['Group', 'M/F', 'Age', 'EDUC', 'SES',
            'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]
data.rename(columns={'M/F':'Gender'}, inplace=True)
data['SES'] = data['SES'].fillna(2.0)
data['Group'] = data['Group'].apply(lambda x: 1 if x == 'Demented' else 0)
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)
df=data
y = df['Group']
X = df.drop(['Group', 'ASF'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
model= LGBMClassifier() 
model.fit(X_train, y_train)
#print(model.predict(([[1.00,87.00,14.00,2.00,27.00,0.00,1987.00,0.696]])))
pickle.dump(model,open("Alzehimer.pkl","wb"))
