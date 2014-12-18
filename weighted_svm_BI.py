import csv
import numpy as np
import pylab as pl
import random
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from pylab import *
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


#splitting
X=preprocessing.scale(X)
(X_train, X_test, y_train, y_test) = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

#~ #Recherche initiale

#~ C_range = 10.0 ** np.arange(0,8)#10000	
#~ gamma_range = 10.0 ** np.arange(-8, 4)#100
#~ param_grid = dict(gamma=gamma_range, C=C_range)
#~ cv = StratifiedKFold(y_train,2)
#~ grid = GridSearchCV(SVC(kernel='rbf',class_weight={0:1,1:(len(y_test)+0.0)/sum(y_test)}), param_grid=param_grid, cv=cv)
#~ grid.fit(X_train, y_train)
#~ print("The best classifier is: ", grid.best_estimator_)
#~ print(grid.best_estimator_.C)
#~ print(grid.best_estimator_.gamma)

#~ score_dict = grid.grid_scores_
#~ print(score_dict)
#~ scores = [u[1] for u in score_dict]
#~ scores = np.array(scores).reshape(len(C_range), len(gamma_range))
#~ pl.figure(figsize=(8, 6))
#~ pl.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95)
#~ pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
#~ pl.xlabel('gamma')
#~ pl.ylabel('C')
#~ pl.colorbar()
#~ pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
#~ pl.yticks(np.arange(len(C_range)), C_range)

#~ pl.show()

#~ #Ce qui sort de la recherche initiale est que les meilleurs coeff sont (10e7,10e-4)



def count_misclass_0_et_1(y_test,y_res):
	c_1=0
	c_2=0
	for i in xrange(0,len(y_test)):
		if (y_test[i]==1)&(y_res[i]==0):
			c_1=c_1+1
		if (y_test[i]==0)&(y_res[i]==1):
			c_2=c_2+1
		else:
			()
	return(c_1,c_2)

def recherche_meilleur_coeff(n):
	C_res=1#trouve sur la grosse grid
	gamma_res=0.001#idem
	l_gamma=[0]*n
	l_C=[0]*n
	accuracy_train=[0]*n
	accuracy_test=[0]*n
	l_plot=xrange(1,n)
	for j in xrange(1,n):
		#a l'etape j on teste les couples situes aux sommets et sur le milieu des aretes du carres centre sur le couple trouve a l'etape j-1
		#la longueur du carre a l'etape j est 10**(1/(2**j)) puis on selectionne le meilleur
		C_range=[C_res*10**(-(1+0.0)/(2**j)),C_res,C_res*10**((1+0.0)/(2**j))]
		gamma_range=[gamma_res*10**(-(1+0.0)/(2**j)),gamma_res,gamma_res*10**((1+0.0)/(2**j))]
		cv = StratifiedKFold(y_train,2)
		param_grid = dict(gamma=gamma_range, C=C_range)
		print(param_grid)
		grid = GridSearchCV(SVC(kernel='rbf',class_weight={0:1,1:(len(y_test)+0.0)/sum(y_test)}), param_grid=param_grid, cv=cv)
		grid.fit(X_train, y_train)
		C_res=grid.best_estimator_.C
		gamma_res=grid.best_estimator_.gamma#Selection de la meilleure valeur
		print(C_res)#valeur de C a l'etape j
		l_C[j-1]=C_res#On remplit pour avoir chaque valeur de C
		print(gamma_res)#Valeur de gamma a l'etape j
		l_gamma[j-1]=gamma_res#On remplit pour avoir chaque valeur de gamma
		print(grid.best_score_)#Precision sur la base d'entrainement
		accuracy_train[j-1]=grid.best_score_#On remplit pour avoir les precisions successives sur la base d'entrainement
		print(grid.score(X_test,y_test))#Precision sur la base de test
		accuracy_test[j-1]=grid.score(X_test,y_test)#On remplit pour avoir les precisions successives sur la bases de test
	plot(l_plot,accuracy_train,label='Training')
	plot(l_plot,accuracy_test,label='Test')
	return(C_res,gamma_res,l_C,l_gamma,accuracy_train,accuracy_test)

#~ print recherche_meilleur_coeff(15)


clf=SVC(C=1.76285186855,gamma=0.0015481724539,kernel='rbf',class_weight={0:1,1:(len(y_test)+0.0)/(sum(y_test))}).fit(X_train,y_train)
y_res=clf.predict(X_test)
	
print count_misclass_0_et_1(y_test,y_res)
print(len(y_test))
print(sum(y_test))
print('Recall rate')
print 1-(count_misclass_0_et_1(y_test,y_res)[0]+0.0)/sum(y_test)
print('Precision')
print 1-(count_misclass_0_et_1(y_test,y_res)[0]+count_misclass_0_et_1(y_test,y_res)[1]+0.0)/len(y_test)

