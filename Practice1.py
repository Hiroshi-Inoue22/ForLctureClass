import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('C:/Users\井上　大士\PycharmProjects\pythonProject')

df = pd.read_csv('titanic.train.csv')

print(df.head(10))
print(df.info())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

print('motimoti;',df['Embarked'].mode()[0])

x = df.drop(columns=['PassengerId','Survived','Name','Ticket','Cabin'])
y = df['Survived']

print(x.head(10))

from sklearn.preprocessing import LabelEncoder

cat_features = ['Sex','Embarked']

for col in cat_features:
    lbl_E = LabelEncoder()
    x[col] = lbl_E.fit_transform(list(df[col].values))

from sklearn.preprocessing import StandardScaler

num_features = ['Age','Fare']

for col in num_features:
    Sta_S = StandardScaler()
    x[col] = Sta_S.fit_transform(np.array(df[col].values).reshape(-1,1))

print(x.head())

from sklearn.decomposition import PCA
pca = PCA()

x_pca = pca.fit_transform(x)

print('Checker:',x.shape[0])
print('Checker:',x.shape[1])
print('Checker:',x.columns.values)

p_compo = pca.components_
print(range(len(pca.components_)),x.columns.values)
print(range(len(pca.components_)),range(1,len(pca.components_)+1))
#print(x_pca.shape)
print('DIO:',np.hstack([0,pca.explained_variance_ratio_.cumsum()]))
print('DIO:',pca.explained_variance_ratio_)

from mpl_toolkits.mplot3d import Axes3D

def plot_2d(x,y):
    plt.plot(x[:,0][y==0],x[:,1][y==0],'bo',ms=5)
    plt.plot(x[:,0][y==1],x[:,1][y==1],'r^',ms=5)
    plt.xlabel("First Principal Componet")
    plt.ylabel("Second Principal Componet")
    plt.legend(['Not Survived','Survived'],loc='best')

def plot_3d(x,y):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(projection='3d')

    ax.plot(x[:,0][y==0],x[:,1][y==0],x[:,2][y==0],'bo',ms=3)
    ax.plot(x[:, 0][y==1],x[:, 1][y==1],x[:,2][y==1],'r^', ms=3)

    ax.set_xlabel("First Principal Compornent",fontsize=10)
    ax.set_ylabel("Second Principal Compornent",fontsize=10)
    ax.set_zlabel("Second Principal Compornent",fontsize=10)
    ax.legend(['Not Survived','Survived'],loc="best")

plt.figure(figsize=(8,6))

plot_2d(x_pca,y)
plot_3d(x_pca,y)


ruiseki = pca.explained_variance_ratio_
print(ruiseki)

plt.figure(figsize=(6,4))
plt.plot(np.hstack([0,pca.explained_variance_ratio_.cumsum()]))
plt.xlabel('n_component')
plt.ylabel('explained_variance_ratio')

p_compo = pca.components_
print(p_compo)

plt.matshow(pca.components_,cmap='Oranges')
plt.yticks(range(len(pca.components_)),range(1,len(pca.components_)+1))
plt.colorbar()
plt.xticks(range(x.shape[1]),x.columns.values,rotation=60,ha='left')
plt.xlabel('Features')
plt.ylabel('Principal Component')
plt.show()
# 篠原　拓海　侵入成功
