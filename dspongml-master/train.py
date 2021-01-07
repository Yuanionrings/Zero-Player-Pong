import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("pong_data.csv")

x = data.iloc[:,0:4] #grab first 4 col
y = data.iloc[:,5] #grab paddle_y col
#print(y.head())

reg = KNeighborsClassifier().fit(x, y)
print(reg.score(x,y))

#x.to_csv('x_test.csv',index=False)
#y.to_csv('y_test.csv', index=False)

from joblib import dump,load
dump(reg,'mymodel.joblib')
clf = load('mymodel.joblib')
