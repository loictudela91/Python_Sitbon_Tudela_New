import math
import numpy as np
import random
from numpy import linalg
from sklearn import preprocessing
import pylab
import matplotlib
import matplotlib.pyplot as plt

#### fonction sigmoide ####
def sigmoid(x):
	return (1+0.0)/(1+math.exp(-x))


####Fonction neurone####
def calcul_neurones(w,x):
	return sigmoid(np.inner(w,x))
#print(calcul_neurones([1,1,1],[0,1,0]))

####Fonction couche####
def calcul_couche(x_sent,W):
	res=np.zeros(W.shape[0])
	for i in xrange(0,len(res)):
		res[i]=calcul_neurones(W[i,:],x_sent)
	return res



####gradient classique####

#Pour les poids de sortie
def vect(W,X):
	d=W.shape[0]
	vect=[0]*d
	for i in xrange(0,d):
		vect[i]=sigmoid(np.inner(W[i,:],X))
	return vect

def gradient_nu(y_expected,y,W,x_sent):
	return -(y_expected-y)*np.array(vect(W,x_sent))


#pour les poids de la couche avant la sortie
def gradient_wk(nu_k,y_expected,y,w_k,x_sent):
	return -(y_expected-y)*nu_k*sigmoid(np.inner(w_k,x_sent))*(1-sigmoid(np.inner(w_k,x_sent)))*np.array(x_sent)



####backpropagation####

def back_prop(W,nu,x_sent,y_expected,X,eta):
	m=W.shape[0]
	p=len(W[0,:])
	r=len(nu)
	#sortie du reseau
	sortie=calcul_couche(x_sent,W)
	y=np.inner(nu,sortie)
	#vecteur total de coeff
	coeff=[]
	ref_nu=np.array(nu)
	ref_nu.shape=(len(nu),1)
	coeff.append(ref_nu)
	for i in xrange(0,m):
		ref=W[i,:]
		ref.shape=(p,1)
		coeff.append(ref)
	#vecteur gradient classique
	#print coeff
	grad=[]
	ref_0=np.array(gradient_nu(y_expected,y,W,x_sent))
	ref_0.shape=(len(ref_0),1)
	grad.append(ref_0)
	for i in xrange(0,m):
		ref_i=np.array(gradient_wk(nu[i],y_expected,y,W[i,:],x_sent))
		ref_i.shape=(len(ref_i),1)
		grad.append(ref_i)
	#mise a jour des poids
	for i in xrange(0,len(coeff)):
		coeff[i]=coeff[i]-eta*grad[i]
	#nouveau coefficient
	return coeff



