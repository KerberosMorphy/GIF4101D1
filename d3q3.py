# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 3, Question 3
#
###############################################################################
############################## INSTRUCTIONS ###################################
###############################################################################
#
# - RepÃ©rez les commentaires commenÃ§ant par TODO : ils indiquent une tÃ¢che que
#       vous devez effectuer.
# - Vous ne pouvez PAS changer la structure du code, importer d'autres
#       modules / sous-modules, ou ajouter d'autres fichiers Python
# - Ne touchez pas aux variables, TMAX*, ERRMAX* et _times, Ã  la fonction
#       checkTime, ni aux conditions vÃ©rifiant le bon fonctionnement de votre 
#       code. Ces structures vous permettent de savoir rapidement si vous ne 
#       respectez pas les requis minimum pour une question en particulier. 
#       Toute sous-question n'atteignant pas ces minimums se verra attribuer 
#       la note de zÃ©ro (0) pour la partie implÃ©mentation!
#
###############################################################################

import time
import numpy

from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import cdist

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from matplotlib import pyplot


# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []
def checkTime(maxduration, question):
    
    _times.append(time.time())
    _times.append(time.time())
    
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!") 

# DÃ©finition des durÃ©es d'exÃ©cution maximales pour chaque sous-question
TMAX_FIT = 2.0
TMAX_EVAL = 3.0


# Ne modifiez rien avant cette ligne!


# Question 3B
# ImplÃ©mentation du discriminant Ã  noyau
class DiscriminantANoyau:

    def __init__(self, lambda_, sigma):
        # Cette fonction est dÃ©jÃ  codÃ©e pour vous, vous n'avez qu'Ã  utiliser
        # les variables membres qu'elle dÃ©finit dans les autres fonctions de
        # cette classe.
        # Lambda et sigma sont dÃ©finis dans l'Ã©noncÃ©.
        self.lambda_ = lambda_
        self.sigma = sigma
    
    def fit(self, X, y):
        # ImplÃ©mentez la fonction d'entraÃ®nement du classifieur, selon
        # les Ã©quations que vous avez dÃ©veloppÃ©es dans votre rapport.
        X = numpy.array(X)
        y = numpy.array(y)
    
        # TODO Q3B
        # Vous devez Ã©crire une fonction nommÃ©e evaluateFunc,
        # qui reÃ§oit un seul argument en paramÃ¨tre, qui correspond aux
        # valeurs des paramÃ¨tres pour lesquels on souhaite connaÃ®tre
        # l'erreur et le gradient d'erreur pour chaque paramÃ¨tre.
        # Cette fonction sera appelÃ©e Ã  rÃ©pÃ©tition par l'optimiseur
        # de scipy, qui l'utilisera pour minimiser l'erreur et obtenir
        # un jeu de paramÃ¨tres optimal.
       
        #Tu peux juste calculer une matrice triangulaire btw
        K = cdist(X_train, X_train)
        K = K / self.sigma
        K = K ** 2
        K = numpy.exp(K)
        
        
        for t in range(len(X)):
            K[t,:] = K[t,:] * y
        
        
        def evaluateFunc(hypers):
            alphas = numpy.array(hypers[0:-1])
            omega = hypers[-1]
            
            h = numpy.zeros((len(alphas)))
            for t in range(len(alphas)):
                h[t] = numpy.sum(alphas[t] * K[t,:]) + omega
            
            
            idx = numpy.where(y * h < 1)
            e = numpy.zeros((len(idx)))
            e = 1 - numpy.array(y)[idx[0]] * numpy.array(h)[idx[0]]
            err = numpy.sum(e) + self.lambda_ * numpy.sum(numpy.array(alphas)[idx[0]])
            
            grad = numpy.zeros(hypers.shape)
            grad[idx[0]] = self.lambda_ - 1
            grad[-1] = - numpy.sum(numpy.array(y)[idx[0]])
            
            return err, grad
        
        # TODO Q3B
        # Initialisez alÃ©atoirement les paramÃ¨tres alpha et omega0
        # (l'optimiseur requiert un "initial guess", et nous ne pouvons pas
        # simplement n'utiliser que des zÃ©ros pour diffÃ©rentes raisons).
        # Stochez ces valeurs initiales alÃ©atoires dans un array numpy nommÃ©
        # "params"
        params = numpy.random.rand(len(X) + 1)
        # DÃ©terminez Ã©galement les bornes Ã  utiliser sur ces paramÃ¨tres
        # et stockez les dans une variable nommÃ©e "bounds".
        # Indice : les paramÃ¨tres peuvent-ils avoir une valeur maximale (au-
        # dessus de laquelle ils ne veulent plus rien dire)? Une valeur
        # minimale? RÃ©fÃ©rez-vous Ã  la documentation de fmin_l_bfgs_b
        # pour savoir comment indiquer l'absence de bornes.
        bounds = list(map(lambda x: (0, None), numpy.zeros((len(X)))))
        bounds.append((None,None))

        # Ã€ ce stade, trois choses devraient Ãªtre dÃ©finies :
        # - Une fonction d'Ã©valuation nommÃ©e evaluateFunc, capable de retourner
        #   l'erreur et le gradient d'erreur pour chaque paramÃ¨tre pour une
        #   configuration de paramÃ¨tres alpha et omega_0 donnÃ©e.
        # - Un tableau numpy nommÃ© params de mÃªme taille que le nombre de
        #   paramÃ¨tres Ã  entraÃ®ner.
        # - Une liste nommÃ©e bounds contenant les bornes que l'optimiseur doit 
        #   respecter pour chaque paramÃ¨tre
        # On appelle maintenant l'optimiseur avec ces informations et on stocke
        # les valeurs optimisÃ©es dans params
        _times.append(time.time())
        
        params, minval, infos = fmin_l_bfgs_b(evaluateFunc, params, bounds=bounds)
        #params, minval, infos = fmin_l_bfgs_b(evaluateFunc, params, bounds=bounds)
        _times.append(time.time())
        checkTime(TMAX_FIT, "Entrainement")

        # On affiche quelques statistiques
        print("EntraÃ®nement terminÃ© aprÃ¨s {it} itÃ©rations et "
                "{calls} appels Ã  evaluateFunc".format(it=infos['nit'], calls=infos['funcalls']))
        print("\tErreur minimale : {:.5f}".format(minval))
        print("\tL'algorithme a convergÃ©" if infos['warnflag'] == 0 else "\tL'algorithme n'a PAS convergÃ©")
        print("\tGradients des paramÃ¨tres Ã  la convergence (ou Ã  l'Ã©puisement des ressources) :")
        print(infos['grad'])

        # TODO Q3B
        # Stockez les paramÃ¨tres optimisÃ©s de la faÃ§on suivante
        # - Le vecteur alpha dans self.alphas
        # - Le biais omega0 dans self.w0
        self.alpha = params[0:-1]
        self.omega0 = params[-1:]

        # On retient Ã©galement le jeu d'entraÃ®nement, qui pourra
        # vous Ãªtre utile pour les autres fonctions Ã  implÃ©menter
        self.X, self.y = X, y
    
    def predict(self, X):
        # TODO Q3B
        # ImplÃ©mentez la fonction de prÃ©diction
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
        # et que les variables membres alphas, w0, X et y existent.
        # N'oubliez pas que ce classifieur doit retourner -1 ou 1
        K = cdist(numpy.array(X), self.X)
        K = K / self.sigma
        K = K ** 2
        K = numpy.exp(K)
        for t in range(len(X)):
            K[t,:] = K[t,:] * self.y
        
        h = numpy.zeros((len(X)))
        for t in range(len(X)):
            h[t] = numpy.sum(self.alpha * K[t,:]) + self.omega0
        
        return numpy.sign(h)
        
    def score(self, X, y):
        # TODO Q3B
        # ImplÃ©mentez la fonction retournant le score (accuracy)
        # du classifieur sur les donnÃ©es reÃ§ues en argument.
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
        # Indice : rÃ©utiliser votre implÃ©mentation de predict() rÃ©duit de
        # beaucoup la taille de cette fonction!
        pass


if __name__ == "__main__":
    # Question 3B

    # TODO Q3B
    # CrÃ©ez le jeu de donnÃ©es Ã  partir de la fonction make_moons, tel que
    # demandÃ© dans l'Ã©noncÃ©
    # N'oubliez pas de vous assurer que les valeurs possibles de y sont
    # bel et bien -1 et 1, et non 0 et 1!
    data = make_moons(1000, 0.3)
    X = data[0]
    y = list(map(lambda x: 1 if x == 1 else -1,data[1]))
    
    # TODO Q3B
    # SÃ©parez le jeu de donnÃ©es en deux parts Ã©gales, l'une pour l'entraÃ®nement
    # et l'autre pour le test
    X_train = X[0:int(len(X)/2),:]
    y_train = y[0:int(len(y)/2)]
    X_test = X[int(len(X)/2):len(X),:]
    y_test = y[int(len(y)/2):len(y)]
    
    
    _times.append(time.time())
    # TODO Q3B
    # Une fois les paramÃ¨tres lambda et sigma de votre classifieur optimisÃ©s,
    # crÃ©ez une instance de ce classifieur en utilisant ces paramÃ¨tres optimaux,
    # et calculez sa performance sur le jeu de test.
    clf = DiscriminantANoyau(0.1, 1)
    clf.fit(X_train, y_train)
    
    # TODO Q3B
    # CrÃ©ez ici une grille permettant d'afficher les rÃ©gions de
    # dÃ©cision pour chaque classifieur
    # Indice : numpy.meshgrid pourrait vous Ãªtre utile ici
    # Par la suite, affichez les rÃ©gions de dÃ©cision dans la mÃªme figure
    # que les donnÃ©es de test.
    # Note : utilisez un pas de 0.02 pour le meshgrid
    f, (ax1) = pyplot.subplots(1, 1, sharex=True)        

    x1 = numpy.arange(min(X_test[:,0]), max(X_test[:,0]), 0.02)
    x2 = numpy.arange(min(X_test[:,1]), max(X_test[:,1]), 0.02)
    xx, yy = numpy.meshgrid(x1, x2)            
    
    pred = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)
    cs = ax1.contourf(xx, yy, pred, cmap=pyplot.cm.Paired)
            
    colors = "brg"
    for i in numpy.unique(y_test) : 
        indexes = numpy.where(y_test == i)
        ax1.plot(X_test[indexes,0], X_test[indexes,1], '+', c = colors[i])
        
    
    


    # On affiche la figure
    # _times.append(time.time())
    checkTime(TMAX_FIT, "Evaluation")
    pyplot.show()
# N'Ã©crivez pas de code Ã  partir de cet endroit
