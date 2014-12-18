import csv
import numpy as np
import random
from sklearn import tree
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
import pylab
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
#Base d'entrainement reduite prise j'ai un algo qui genere permutation en O(nln(n)) mais le data est trop grande pour qu'il marche. pour le moment je prends une plus petite base

taille_data_train=2000000

with open("bio_train.csv","rb") as f:
	X=[]
	y=[]
	i=0
	for row in csv.reader(f):
		if i<taille_data_train+1:	
			res=True
			j=0
			label=row[0]
			row_1=row[1:1170]
			while j<=len(row_1)-1:
				if row_1[j]<>'':
					j=j+1
				else:
					res=False
					j=j+1
			if res==True:
				X.append(row_1)
				
				y.append(label)
				i=i+1
			else:
				i=i+1
		else:
			break	

X=X[1:]
y=y[1:]
for i in xrange(0,len(X)):
	y[i]=int(y[i])
	for j in xrange(0,len(X[i])):
			X[i][j]=float(X[i][j])
for i in xrange(0,len(y)):
	if y[i]==0:
		y[i]=-1

print len(X)
#Erreurs de premiere et deuxieme espece
def count_misclass_0_1(y_res,y):
	c_0=0
	c_1=0
	for i in xrange(0,len(y)):
		if (y_res[i]==-1) and (y[i]==1):
			 c_0=c_0+1
		if (y_res[i]==1) and (y[i]==-1):
			 c_1=c_1+1
	return (c_0,c_1)
	

#Erreurs de premiere et deuxieme espece
def count_misclass_0_1(y_res,y):
	c_0=0
	c_1=0
	for i in xrange(0,len(y)):
		if (y_res[i]==-1) and (y[i]==1):
			c_0=c_0+1
		if (y_res[i]==1) and (y[i]==-1):
			c_1=c_1+1
	return (c_0,c_1)
	

#splitting des bases
X=preprocessing.scale(X)
(X_train, X_test, Y_train, Y_test) = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)


#Description des bases

#Base d'entrainement
res_0=0
res_1=0
for i in xrange(0,len(Y_train)):
	if Y_train[i]==1:
		res_1=res_1+1
	else:
		res_0=res_0+1
long_train=len(Y_train)
print('la taille de la base d entrainement est')
print(long_train)
print('la repartition des classes est la suivante')
print('classe 1')
print(res_1)
print('classe -1')
print(res_0)
#Base de test
res_t_0=0
res_t_1=0
for i in xrange(0,len(Y_test)):
	if Y_test[i]==1:
		res_t_1=res_t_1+1
	else:
		res_t_0=res_t_0+1
long_test=len(Y_test)
print('la taille de la base de test est')
print(long_test)
print('la repartition des classes est la suivante')
print('classe 1')
print(res_t_1)
print('classe -1')
print(res_t_0)


#differents max depth pour les arbres

liste_recall_test=[]
liste_accuracy_test=[]
liste_taille_arbre=[]

for i in xrange(1,101):
	clf_1=tree.DecisionTreeClassifier(max_depth=10)
	clf = AdaBoostClassifier(clf_1,n_estimators=i, learning_rate=1-0.001*i)
	clf = clf.fit(X_train, Y_train)
	print i
	Y_res_test=clf.predict(X_test)
	liste_taille_arbre.append(i)
	liste_accuracy_test.append(1-(count_misclass_0_1(Y_res_test,Y_test)[0]+count_misclass_0_1(Y_res_test,Y_test)[1]+0.0)/(len(Y_test)))
	liste_recall_test.append(1-(count_misclass_0_1(Y_res_test,Y_test)[0]+0.0)/res_t_1)
	
	

p_1=plt.plot(liste_taille_arbre,liste_accuracy_test)#bleu
p_5=plt.plot(liste_taille_arbre,liste_recall_test)#vert


plt.axis([1, 101, 0, 1])


matplotlib.pyplot.title("Recall rate et Precision sur la base de test")
matplotlib.pyplot.legend(["Precision","Recall Rate"])
matplotlib.pyplot.show()
