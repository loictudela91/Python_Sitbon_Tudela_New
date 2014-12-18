from backpropagationsimple import *
import math
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from numpy import linalg
from sklearn import preprocessing
import pylab
import matplotlib
import matplotlib.pyplot as plt
import csv
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


#Donnees

#Matrice synaptique

W=np.zeros((3,1169))
for i in xrange(0,W.shape[0]):
	for j in xrange(0,W.shape[1]):
		W[i,j]=random.uniform(-0.5,0.5)

#Poids de sortie
v=[0]*3
for i in xrange(0,len(v)):
	v[i]=random.uniform(-0.5,0.5)
v=np.array(v)


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

		

#splitting
X=preprocessing.scale(X)
(X_train, X_test, Y_train, Y_test) = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
print (len(X_train))
print(sum(Y_train))
#algorithme de backpropagation

def algo_backpropagation(X,eta,seuil,nu,W,Y):
	res=[]
	iter=0
	for i in xrange(0,len(X)):
		#print X[i]
		#print W
		sortie=calcul_couche(X[i],W)
		res.append(np.inner(nu,sortie))
	error_list=[0]*len(Y)
	for i in xrange(0,len(Y)):
		error_list[i]=res[i]-Y[i]
	error_list=np.array(error_list)
	error=(1+0.0)/len(X)*linalg.norm(error_list)**2
	print error
	while seuil<=error and iter<=500:
		for i in xrange(0,len(X)):
			coeff=back_prop(W,nu,X[i],Y[i],X,eta)
			#print coeff
			for k in xrange(0,W.shape[0]):
				ref=coeff[k+1]
				ref.shape=(1,len(ref))
				W[k,:]=ref
			#print W
			ref_0=coeff[0]
			#print nu
			ref_0=np.array(ref_0)
			nu_res=[0]*3
			for i in xrange(0,3):
				nu_res[i]=ref_0[i][0]
			nu=nu_res
			#print nu
		res=[]
		for i in xrange(0,len(X)):
			sortie=calcul_couche(X[i],W)
			res.append(np.inner(nu,sortie))
		error_list=[0]*len(Y)
		for i in xrange(0,len(Y)):
			error_list[i]=res[i]-Y[i]
		error_list=np.array(error_list)
		error=(1+0.0)/len(X)*linalg.norm(error_list)**2
		iter=iter+1
		print error
	return (W,v,iter)

resultat=algo_backpropagation(X_train,0.3,0.01,v,W,Y_train)

W=resultat[0]
nu=resultat[1]

def count_misclass_0_1(y_res,y):
	c_0=0
	c_1=0
	for i in xrange(0,len(y)):
		if (y_res[i]==0) and (y[i]==1):
			c_0=c_0+1
		if (y_res[i]==1) and (y[i]==0):
			c_1=c_1+1
	return (c_0,c_1)
	
l_train_preci=[]
l_train_recall=[]
l_test_preci=[]
l_test_recall=[]

for k in xrange(0,100):
####base d'entrainement#####
	Y_res_train=[0]*len(Y_train)

	for i in xrange(0,len(Y_train)):
			sortie=calcul_couche(X_train[i],W)
			Y_res_train[i]=np.inner(nu,sortie)


	for i in xrange(0,len(Y_res_train)):
		if Y_res_train[i]<=(0.01*k):
			Y_res_train[i]=0
		else:
			Y_res_train[i]=1




#### base de test ####
	Y_res_test=[0]*len(Y_test)

	for i in xrange(0,len(Y_test)):
			sortie=calcul_couche(X_test[i],W)
			Y_res_test[i]=np.inner(nu,sortie)
	
	for i in xrange(0,len(Y_res_test)):
		if Y_res_test[i]<=(0.01*k):
			Y_res_test[i]=0
		else:
			Y_res_test[i]=1

	l_train_recall.append(1-(count_misclass_0_1(Y_res_train,Y_train)[0]+0.0)/sum(Y_train))
	l_train_preci.append(1-(count_misclass_0_1(Y_res_train,Y_train)[0]+count_misclass_0_1(Y_res_train,Y_train)[1]+0.0)/(len(Y_train)))
	l_test_recall.append(1-(count_misclass_0_1(Y_res_test,Y_test)[0]+0.0)/sum(Y_test))
	l_test_preci.append(1-(count_misclass_0_1(Y_res_test,Y_test)[0]+count_misclass_0_1(Y_res_test,Y_test)[1]+0.0)/(len(Y_test)))

for i in xrange(0,100):
	print 0.01*i
	print (l_train_preci[i],l_train_recall[i])
	print (l_test_preci[i],l_test_recall[i])
	
	