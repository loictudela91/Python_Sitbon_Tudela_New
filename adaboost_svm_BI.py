	#Chargement des packages
import csv
import math
import numpy as np
#~ import pylab as pl
from sklearn import preprocessing
import random
from sklearn import cross_validation
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation




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
	

#splitting
X=preprocessing.scale(X)
(X_train, X_test, Y_train, Y_test) = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

#Algorithme Adaboost

def prod_listes(l1,l2):
	res=0
	for i in xrange(0,len(l1)):
		res=res+l1[i]*l2[i]
	return res

def count_weighted_error(y_test,y_res,W):
	c_1=0
	c_2=0
	for i in xrange(0,len(y_test)):
		if (y_test[i]==1)&(y_res[i]==-1):
			c_1=c_1+W[i]
		if (y_test[i]==-1)&(y_res[i]==1):
			c_2=c_2+W[i]
		else:
			()
	return(c_1,c_2)


def Ada_Boost(X_train,Y_train,X_test,C_fun,gamma_min,gamma_step,gamma_init,T):
	t=0
	N=len(Y_train)
	W=[(1+0.0)/N]*N
	coeff_a=[]#Liste des coefficients de chaque classifieurs dans le classifieur final
	coeff_e=[]#Liste des erreurs normalisees
	classif_succ=[]#Valeur des classifieurs sur la base de test
	gamma_fun=gamma_init#gamma en cours
	res=[0]*(len(X_test))
	while (gamma_min<=gamma_fun)&(t<=T):
		#clf=SVC(C=C_fun,gamma=gamma_fun,kernel='rbf',class_weight={-1:1,1:(len(y_train)+0.0)/cardinal_1(y_train)}).fit(X_train,y_train)
		clf=SVC(C=C_fun,gamma=gamma_fun,kernel='rbf')
		clf.fit(X_train,Y_train)
		y_classi=clf.predict(X_train)#Resultat du classifieur sur la base d'entrainement a l'etape m
		print y_classi==Y_train
		y_res=clf.predict(X_test)#Resultat du classifieur sur la base de test a l'etape m
		(c_1,c_2)=count_weighted_error(Y_train,y_classi,W) #Erreur totale en prenant en compte les poids
		print((c_1,c_2))
		e=(c_1+c_2+0.0)/(sum(W))#Erreur normalisee
		print(e)
		if e<0.49:
			t=t+1
			a=(1+0.0)/2*(math.log((1-e)/e))
			coeff_a.append(a)
			classif_succ.append(y_res) 
			for j in xrange(0,len(W)):
				if (Y_train[j]==1)&(y_classi[j]==-1) or (Y_train[j]==-1)&(y_classi[j]==1):
					W[j]=W[j]*(math.exp(a))
			#gamma_fun=gamma_fun*gamma_step
			
		else:
			gamma_fun=gamma_fun*gamma_step
		print gamma_fun
	for i in xrange(0,len(res)):
		y_i=[classif_succ[u][i] for u in xrange(0,len(classif_succ))]
		if prod_listes(coeff_a,y_i)<0:
			res[i]=-1
		else:
			res[i]=1
	return res
y_res=Ada_Boost(X_train,Y_train,X_test,10000000,10e-10,10e-2,0.1,15)
print(count_misclass_0_1(y_res,Y_test))

print count_misclass_0_1(y_res,Y_test)

count=0
for i in xrange(0,len(Y_test)):
	if Y_test[i]==1:
		count=count+1

print('Recall rate')
print 1-(count_misclass_0_1(y_res,Y_test)[0]+0.0)/count
print('Precision')
print 1-(count_misclass_0_1(y_res,Y_test)[0]+count_misclass_0_1(y_res,Y_test)[1]+0.0)/len(Y_test)