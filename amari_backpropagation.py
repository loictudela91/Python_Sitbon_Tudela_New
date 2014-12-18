####Backpropagation AMARI ####

import math
import numpy as np
import random
from numpy import linalg

#### Fonction sigmoide ####

def sigmoid(x):
	return (1+0.0)/(1+math.exp(-x))

#### Fonction neurone ####

def calcul_neurones(w,x):
	return sigmoid(np.inner(w,x))



#### Fonction couche ####

def calcul_couche(x_sent,W):
	res=np.zeros(W.shape[0])
	for i in xrange(0,len(res)):
		res[i]=calcul_neurones(W[i,:],x_sent)
	return res
	
 #### Calcul de la matrice de fisher ####
 
#### Coefficient (w_k,w_p) ####
def coeff_fisher_wk_wp(w_k,w_p,sigma,nu_k,nu_p,X):
	d=len(w_k)
	res=np.zeros((d,d))
	N=len(X)
	i=0
	while i<=len(X)-1:
		X_=X[i]
		X_=np.array(X_)
		X_t=X[i]
		X_t=np.array(X_t)
		X_.shape=(d,1)
		X_t.shape=(1,d)
		phi_k=sigmoid(np.inner(w_k,X[i]))
		phi_p=sigmoid(np.inner(w_p,X[i]))
		res=res+phi_p*(1-phi_p)*phi_k*(1-phi_k)*np.dot(X_,X_t)
		i=i+1
	return(nu_k*nu_p+0.0)/(N*sigma**2)*res	
		
	

####coefficient (wk,wk)####
def coeff_fisher_wk_wk(w_k,sigma,nu_k,X):
	d=len(w_k)
	res=np.zeros((d,d))
	N=len(X)
	i=0
	while i<=len(X)-1:
		X_=X[i]
		X_=np.array(X_)
		X_t=X[i]
		X_t=np.array(X_t)
		X_.shape=(d,1)
		X_t.shape=(1,d)
		phi_k=sigmoid(np.dot(w_k,X[i]))
		res=res+((phi_k)*(1-phi_k))**2*np.dot(X_,X_t)
		i=i+1
	return (nu_k**2+0.0)/(sigma**2*N)*res
	
####coefficient (w_k,nu)####
def vect(W,X):
	d=W.shape[0]
	vect=[0]*d
	for i in xrange(0,d):
		vect[i]=sigmoid(np.inner(W[i,:],X))
	return vect

def coeff_fisher_wk_nu(W,w_k,nu_k,X,sigma):
	d=len(w_k)
	r=W.shape[0]
	N=len(X)
	res=np.zeros((d,r))
	for i in xrange(0,len(X)):
		vector=np.array(vect(W,X[i]))
		ref=np.array(X[i])
		ref.shape=(d,1)
		vector.shape=(1,r)
		phi_k=sigmoid(np.inner(w_k,X[i]))
		res=res+phi_k*(1-phi_k)*np.dot(ref,vector)
	return (nu_k+0.0)/(N*sigma**2)*res

#### les coeff (nu,w_k) sont les transpose de ceux la ####

#### Coefficient (nu,nu) ####
def coeff_fisher_nu_nu(W,X,sigma):
	m=W.shape[0]
	res=np.zeros((m,m))
	N=len(X)
	for i in xrange(0,N):
		vector_1=np.array(vect(W,X[i]))
		vector_2=np.array(vect(W,X[i]))
		vector_1.shape=(m,1)
		vector_2.shape=(1,m)
		res=res+np.dot(vector_1,vector_2)
	return (1+0.0)/(sigma**2*N)*res
	

def matrice_fisher(W,nu,X,sigma):
	m=W.shape[0]
	l=[[]]*(m+1)
	#premiere ligne
	T=[]
	T.append(coeff_fisher_nu_nu(W,X,sigma))
	for i in xrange(0,m):
		#print i
		T.append(coeff_fisher_wk_nu(W,W[i,:],nu[i],X,sigma).transpose())
	l[0]=T
	#autres lignes 
	for i in xrange(0,m):
		T=[]
		T.append(coeff_fisher_wk_nu(W,W[i,:],nu[i],X,sigma))
		for j in xrange(0,m):
			if i==j:
				T.append(coeff_fisher_wk_wk(W[i,:],sigma,nu[i],X))
			else:	
				T.append(coeff_fisher_wk_wp(W[i,:],W[j,:],sigma,nu[i],nu[j],X))
		#print(T)
		l[i+1]=T
	return l

####gradient classique####
def gradient_nu(y_expected,y,W,x_sent):
	return -(y_expected-y)*np.array(vect(W,x_sent))

def gradient_wk(nu_k,y_expected,y,w_k,x_sent):
	return -(y_expected-y)*nu_k*sigmoid(np.inner(w_k,x_sent))*(1-sigmoid(np.inner(w_k,x_sent)))*np.array(x_sent)

####backpropagation####
def back_prop_amari(W,nu,x_sent,y_expected,X,sigma,eta):
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
	for i in xrange(0,len(W)):
		ref=W[i,:]
		ref.shape=(len(W[i,:]),1)
		coeff.append(ref)
	#coeff_1=coeff
	#liste des matrices de fisher
	l=matrice_fisher(W,nu,X,sigma)
	#vecteur gradient classique
	grad=[]
	grad.append(np.array(gradient_nu(y_expected,y,W,x_sent)))
	for i in xrange(0,m):
		grad.append(gradient_wk(nu[i],y_expected,y,W[i,:],x_sent))
	#inversion de la matrice de fisher
	G=np.bmat(l)
	G_inverse=linalg.inv(G)
	#G_inverse=np.eye(G.shape[0])
	#print G_inverse
	#Creation de la matrice de fisher inverse au meme format que la fonction matrice_fisher
	l_inv=[[]]*(m+1)
	#remplissage de la premiere ligne de l_inv
	temp=[]
	temp.append(G_inverse[0:r,0:r])
	for k in xrange(0,m):
		temp.append(G_inverse[0:r,r+k*p:r+(k+1)*p])
	l_inv[0]=temp
	#remplissage des autres lignes
	for i in xrange(0,m):
		temp=[]
		temp.append(G_inverse[r+i*p:r+i*p+p,0:r])
		for j in xrange(0,m):
			temp.append(G_inverse[r+i*p:r+i*p+p,r+j*p:r+j*p+p])
		l_inv[i+1]=temp
	#mise a jour des coefficients
	for i in xrange(0,len(coeff)):
		res=[0]*len(coeff[i])
		res=np.array(res)
		res.shape=(len(grad[i]),1)
		for j in xrange(0,len(l_inv[i])):
			res_1=np.dot(l_inv[i][j],grad[j])
			res_1.shape=(len(grad[i]),1)
			res=res+res_1
		res_2=grad[i]
		res_2.shape=(len(grad[i]),1)
		coeff[i]=coeff[i]-eta*res
		#coeff_1[i]=coeff_1[i]-eta*res_2
	return coeff
