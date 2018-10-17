#!/usr/bin/env python2
# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 2, Question 2
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

from matplotlib import pyplot, patches

from sklearn.datasets import make_classification, load_breast_cancer, load_iris
from sklearn.preprocessing import minmax_scale, normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!") 

# DÃ©finition des durÃ©es d'exÃ©cution maximales pour chaque sous-question
TMAX_Q2B = 5.0
TMAX_Q2Bdisp = 10.0
TMAX_Q2C = 20
TMAX_Q2Dbc = 60
TMAX_Q2Diris = 40
TMAX_Q2Ebc = 30
TMAX_Q2Eiris = 15


# Ne modifiez rien avant cette ligne!



# Question 2B
# ImplÃ©mentation du discriminant linÃ©aire
class DiscriminantLineaire:
    def __init__(self, eta=1e-2, epsilon=1e-2, max_iter=3000):
        # Cette fonction est dÃ©jÃ  codÃ©e pour vous, vous n'avez qu'Ã  utiliser
        # les variables membres qu'elle dÃ©finit dans les autres fonctions de
        # cette classe.
        self.eta = eta
        # Epsilon et max_iter servent Ã  stocker les critÃ¨res d'arrÃªt
        # max_iter est un simple critÃ¨re considÃ©rant le nombre de mises Ã  jour
        # effectuÃ©es sur les poids (autrement dit, on cesse l'apprentissage
        # aprÃ¨s max_iter itÃ©ration de la boucle d'entraÃ®nement), alors que
        # epsilon indique la diffÃ©rence minimale qu'il doit y avoir entre
        # les erreurs de deux itÃ©rations successives pour que l'on ne
        # considÃ¨re pas l'algorithme comme ayant convergÃ©. Par exemple,
        # si epsilon=1e-2, alors tant que la diffÃ©rence entre l'erreur
        # obtenue Ã  la prÃ©cÃ©dente itÃ©ration et l'itÃ©ration courante est
        # plus grande que 0.01, on continue, sinon on arrÃªte.
        self.epsilon = epsilon
        self.max_iter = max_iter
    
    def fit(self, X, y):
        # ImplÃ©mentez la fonction d'entraÃ®nement du classifieur, selon
        # les Ã©quations que vous avez dÃ©veloppÃ©es dans votre rapport.
        #X = normalize(X, axis=0)
        print('----------------------------')        
        # On initialise les poids alÃ©atoirement et les dérivés partielles
        w = numpy.random.rand(X.shape[1]+1)
        partial_der = numpy.zeros(w.shape)          
        
        # TODO Q2B
        # Vous devez ici implÃ©menter l'entraÃ®nement.
        # Celui-ci devrait Ãªtre contenu dans la boucle suivante, qui se rÃ©pÃ¨te
        # self.max_iter fois
        # Vous Ãªtes libres d'utiliser les noms de variable de votre choix, sauf
        # pour les poids qui doivent Ãªtre contenus dans la variable w dÃ©finie plus haut
        
        #on transforme les labels dans le forme qui nous interesse
        y = numpy.array(map(lambda x: 1 if x == 1 else -1,y))
        #On peut se permettre de calculer toute les normes une fois au lieu de tout
        # recacluler a chaque itération
        norms = numpy.sum(numpy.array(map(lambda x: x*x, X)), axis = 1)

        #Valeur de l'erreur initialisée 
        tempErr = 999999999999
        for i in range(self.max_iter):
            
            if(i == self.max_iter - 1):
                print('max iter reached')
            
            #calcul de h pour tout x
            h = numpy.array(map(lambda x: sum(numpy.append(x, 1) * w), X))
            
            #Liste des index où le classifieur a faux
            errindexes = numpy.where(h * y < 0)
            
            #On arrête tout si il n'y a pas d'erreur (car la m.a.j) des poids 
            #repose sur les individus mal classés
            if (len(errindexes[0]) == 0):
                print('no misclassification')
                self.w = w
                break
        
            #extraction des individus mal classées
            Xerr = numpy.array(X[errindexes, :][0])
            herr = h[errindexes[0]]
            yerr = y[errindexes[0]]
            normerr = norms[errindexes[0]]
            
            #Calcul du critère d'erreur
            Err = 0.5 * sum((yerr - herr) * (yerr - herr) / normerr)
            #if(i%100 == 0):
            #   print(Err)
            
            
            #Critère d'arrêt
            if(abs(tempErr - Err) < self.epsilon):
                print('local minimum reached')
                #print(Err)
                #print(abs(tempErr - Err))
                break
            
            #Stockage de l'ancienne valeur d'erreur
            tempErr = Err       
            
            #reset without affectation
            partial_der = partial_der * 0
            
            #Permet de séparer le calcul ci dessous et de ne pas le refaire à chaque fois            
            temp = (herr - yerr) / normerr
            #The 3 blocks bellow are equivalent
            #Calcul des dérivés partielles
            '''
            partial_der = reduce(lambda acc, x: acc + numpy.append(numpy.array(Xerr[x,:]), [1]) * temp[x],range(len(errindexes)), partial_der)
            '''
            for t in range(len(errindexes)):
                partial_der = partial_der + numpy.append(numpy.array(Xerr[t,:]), [1]) * temp[t]
            '''
            for t in range(len(errindexes)):
                for j in range(X.shape[1]):
                    partial_der[j] = partial_der[j] + Xerr[t,j] * temp[t]
                #Ne pas oublier w0
                partial_der[-1] = partial_der[-1] + temp[t]
            '''
            
            #Mise à jour des poids
            w = w - (self.eta * partial_der)
            
            #Useful to display evolution of classificateur through iterations
            '''
            if (i%100 == 0):
                print(Err)
                self.w = w
                f, (ax1) = pyplot.subplots(1, 1, sharex=True)    
                x1 = numpy.arange(min(X[:,0]), max(X[:,0]), 0.05)
                x2 = numpy.arange(min(X[:,1]), max(X[:,1]), 0.05)
                xx, yy = numpy.meshgrid(x1, x2)        
            
                pred, h = self.predict(numpy.c_[xx.ravel(), yy.ravel()])
                pred = pred.reshape(xx.shape)
                cs = ax1.contourf(xx, yy, pred, cmap=pyplot.cm.Paired)
                
                
                #pred = self.predict(X)
                color = "rgyb"
                for c in numpy.unique(y):
                    indexes = numpy.where(y == c)
                    ax1.plot(X[indexes,0], X[indexes,1], '+', c = color[c])
                
                pyplot.show()
                pass
            '''
            
            
        #print('-----------')
        #print(w)
        
        # Ã€ ce stade, la variable w devrait contenir les poids entraÃ®nÃ©s
        # On les copie dans une variable membre pour les conserver
        self.w = w
    
    def predict(self, X):
        # TODO Q2B
        # ImplÃ©mentez la fonction de prÃ©diction
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
        
        #pour chaque x, on calcule h(x)
        h = map(lambda x: sum(numpy.append(x, 1) * self.w), X)
        
        #On renvoie la prediction 1 si x > 0, o sinon
        return numpy.array(map(lambda x: 1 if x >= 0 else 0, h)), h
    
    def score(self, X, y):
        # TODO Q2B
        # ImplÃ©mentez la fonction retournant le score (accuracy)
        # du classifieur sur les donnÃ©es reÃ§ues en argument.
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
        # Indice : rÃ©utiliser votre implÃ©mentation de predict() rÃ©duit de
        # beaucoup la taille de cette fonction!
        
        pred, h = self.predict(X)
        pred = numpy.array(pred)
        #bi-class where one class is labelled 1     
        
        #On s'assure que y soit au bon format pour tester
        y = map(lambda x: 1 if x == 1 else 0,y)
        y = numpy.array(y)

    
        acc = len(numpy.where(pred == y)[0])
        return float(acc) / float(len(X))
        
        


# Question 2B
# ImplÃ©mentation du classifieur un contre tous utilisant le discriminant linÃ©aire
# dÃ©fini plus haut
class ClassifieurUnContreTous:
    def __init__(self, n_classes, **kwargs):
        # Cette fonction est dÃ©jÃ  codÃ©e pour vous, vous n'avez qu'Ã  utiliser
        # les variables membres qu'elle dÃ©finit dans les autres fonctions de
        # cette classe.
        self.n_classes = n_classes
        self.estimators = [DiscriminantLineaire(**kwargs) for c in range(n_classes)]
    
    def fit(self, X, y):
        # TODO Q2C
        # ImplÃ©mentez ici une approche un contre tous, oÃ¹ chaque classifieur 
        # (contenu dans self.estimators) est entraÃ®nÃ© Ã  distinguer une seule classe 
        # versus toutes les autres
        
        #On crée un classifieur par classe
        #on pourrait en créer n-1
        #for i in range(self.n_classes - 1):
        for i in range(self.n_classes):
            clf = self.estimators[i]
            y_train = map(lambda x: 1 if x == i else -1, y)
            clf.fit(X, y_train)
            self.estimators[i] = clf
    
    
    def predict(self, X):
        # TODO Q2C
        # ImplÃ©mentez ici la prÃ©diction utilisant l'approche un contre tous
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©

        pred = numpy.zeros((len(X), self.n_classes))
        h = numpy.zeros((len(X), self.n_classes))
        
        for i in range(self.n_classes):
        #for i in range(self.n_classes - 1):
            clf = self.estimators[i]
            pred[:, i], h[:,i] = clf.predict(X)
        
        reconstituedPred = numpy.zeros(pred.shape[0])    
        for i in range(len(pred)):
            sumX = sum(pred[i])
            if sumX == 1:
                #print('---------')
                #print(pred[i])
                #print(numpy.argmax(pred[i]))
                reconstituedPred[i] = numpy.argmax(pred[i])
            #Option de rejet
            #elif (sumX == 0):
            #    reconstituedPred[i] = self.n_classes
            else:
                #option de rejet
                #reconstituedPred[i] = self.n_classes
                
                #Sans option de rejet
                reconstituedPred[i] = numpy.argmax(h[i])
                
                #Dans le cas n-1 classifier
                #reconstituedPred[i] = self.n_classes - 1
                
                
                
        #print(reconstituedPred)
        return numpy.array(reconstituedPred)
    
    def score(self, X, y):
        # TODO Q2C
        # ImplÃ©mentez ici le calcul du score utilisant l'approche un contre tous
        # Ce score correspond Ã  la prÃ©cision (accuracy) moyenne.
        # Vous pouvez supposer que fit() a prÃ©alablement Ã©tÃ© exÃ©cutÃ©
        pred = self.predict(X)
        
        return float(len(numpy.where(pred == y)[0])) / float(len(X))



if __name__ == '__main__':
    # Question 2C

    _times.append(time.time())
    # ProblÃ¨me Ã  2 classes
    
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1)

    # TODO Q2C
    # Testez la performance du discriminant linÃ©aire pour le problÃ¨me
    # Ã  deux classes, et tracez les rÃ©gions de dÃ©cision
    f, (ax1) = pyplot.subplots(1, 1, sharex=True)    
    x1 = numpy.arange(min(X[:,0]), max(X[:,0]), 0.05)
    x2 = numpy.arange(min(X[:,1]), max(X[:,1]), 0.05)
    xx, yy = numpy.meshgrid(x1, x2)        
    ax1.set_title('classifier bi-class')
    #Part 1 : 1V1
    clf = DiscriminantLineaire(eta=3e-2, epsilon=5e-5, max_iter=3000)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    print('score', score)
    pred, h = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)
    cs = ax1.contourf(xx, yy, pred, cmap=pyplot.cm.Paired)

    pred, h = clf.predict(X)
    color = "rgby"
    for c in numpy.unique(pred):
        indexes = numpy.where(y == c)
        ax1.plot(X[indexes,0], X[indexes,1], '+', c = color[c])
    
    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2B")
    
    pyplot.show()

    
    _times.append(time.time())
    # 3 classes
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               n_clusters_per_class=1, n_classes=3)

    f, (ax1) = pyplot.subplots(1, 1, sharex=True)    
    x1 = numpy.arange(min(X[:,0]), max(X[:,0]), 0.05)
    ax1.set_title('classifier multi-class')
    
    x2 = numpy.arange(min(X[:,1]), max(X[:,1]), 0.05)
    xx, yy = numpy.meshgrid(x1, x2)        
    
    
    
    clf = ClassifieurUnContreTous(3, eta=1e-2 , epsilon=1e-4, max_iter= 1500)
    clf.fit(X, y)

    
    score = clf.score(X, y)
    

    pred = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)
    cs = ax1.contourf(xx, yy, pred, cmap=pyplot.cm.Paired)

    pred = clf.predict(X)
    color = "rgby"
    for c in numpy.unique(y):
        indexes = numpy.where(y == c)
        ax1.plot(X[indexes,0], X[indexes,1], '+', c = color[int(c)])
    
    # TODO Q2C
    # Testez la performance du discriminant linÃ©aire pour le problÃ¨me
    # Ã  trois classes, et tracez les rÃ©gions de dÃ©cision
    print('-------')
    print('score')
    print(clf.score(X, y))

    _times.append(time.time())
    checkTime(TMAX_Q2Bdisp, "2C")
    
    pyplot.show()




    '
    # Question 2D

    _times.append(time.time())

    # TODO Q2D
    # Chargez les donnÃ©es "Breast cancer Wisconsin" et normalisez les de
    # maniÃ¨re Ã  ce que leur minimum et maximum soient de 0 et 1
    data = load_breast_cancer()        
    X = data.data
    y = data.target


    
    #Normalization
    X = minmax_scale(X)
    
    #for i in range(X.shape[1]):
    #   X[:,i] = X[:,i] - min(X[:,i])
    #   X[:,i] = X[:,i] / max(X[:,i])
    # TODO Q2D
    # Comparez les diverses approches demandÃ©es dans l'Ã©noncÃ© sur Breast Cancer
    # Initialisez votre discriminant linÃ©aire avec les paramÃ¨tres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par dÃ©faut
    # N'oubliez pas que l'Ã©valuation doit Ãªtre faite par une validation
    # croisÃ©e Ã  K=3 plis!
    clfs = [DiscriminantLineaire(eta=3e-2, epsilon=1e-4, max_iter=2000),
            LinearDiscriminantAnalysis(),
            Perceptron(),
            LogisticRegression()
            ]
    scores = numpy.zeros(len(clfs))
    scores_train = numpy.zeros(len(clfs))
    rkf = KFold(n_splits=3, shuffle=True)
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for c, i in zip(clfs, range(len(clfs))):   
            #print(clf)
            c.fit(X_train, y_train)
            scores_train[i] += c.score(X_train, y_train)
            scores[i] += c.score(X_test, y_test)
            #print(c.__class__.__name__, c.score(X_test, y_test))
    scores = scores / 3  
    scores_train = scores_train / 3    
    
    print('-------')
    print('score entraînement cancer')
    for i in range(len(scores)):
        print(clfs[i].__class__.__name__, scores_train[i])

    print('-------')
    print('score généralisation cancer')
    for i in range(len(scores)):
        print(clfs[i].__class__.__name__, scores[i])


    
    _times.append(time.time())
    checkTime(TMAX_Q2Dbc, "2Dbc")
    
    _times.append(time.time())
    # TODO Q2D
    # Chargez les donnÃ©es "Iris" et normalisez les de
    # maniÃ¨re Ã  ce que leur minimum et maximum soient de 0 et 1

    # TODO Q2D
    # Comparez les diverses approches demandÃ©es dans l'Ã©noncÃ© sur Iris
    # Pour utilisez votre discriminant linÃ©aire, utilisez l'approche Un Contre Tous
    # implÃ©mentÃ© au 2C.
    # Initialisez vos discriminants linÃ©aires avec les paramÃ¨tres suivants :
    # DiscriminantLineaire(eta=1e-4, epsilon=1e-6, max_iter=10000)
    # Pour les autres approches, conservez les valeurs par dÃ©faut
    # N'oubliez pas que l'Ã©valuation doit Ãªtre faite par une validation
    # croisÃ©e Ã  K=3 plis!
    
    clfs = [ClassifieurUnContreTous(3, eta=2e-2, epsilon=1e-4, max_iter=3000),
            LinearDiscriminantAnalysis(),
            Perceptron(),
            LogisticRegression()
            ]
    
    print('-----------------')
    data = load_iris()
    X = data.data
    y = data.target
    X = minmax_scale(X)
    scores_train = numpy.zeros(len(clfs))
    scores = numpy.zeros(len(clfs))
    
    rkf = KFold(n_splits=3, shuffle=True)
    for train_index, test_index in rkf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for c, i in zip(clfs, range(len(clfs))):   
            c.fit(X_train, y_train)
            scores_train[i] += c.score(X_train, y_train)
            scores[i] += c.score(X_test, y_test)
            
    scores = scores / 3
    scores_train = scores_train / 3    
    
    print('-------')
    print('score entraînement Iris')
    for i in range(len(scores)):
        print(clfs[i].__class__.__name__, scores_train[i])

    print('-------')
    print('score généralisation Iris')
    for i in range(len(scores)):
        print(clfs[i].__class__.__name__, scores[i])

    _times.append(time.time())
    checkTime(TMAX_Q2Diris, "2Diris")
    

   


    
    
    
    _times.append(time.time())
    # TODO Q2E
    # Testez un classifeur K plus proches voisins sur Breast Cancer
    # L'Ã©valuation doit Ãªtre faite en utilisant une approche leave-one-out
    # Testez avec k = {1, 3, 5, 7, 11, 13, 15, 25, 35, 45} et avec les valeurs
    # "uniform" et "distance" comme valeur de l'argument "weights".
    # N'oubliez pas de normaliser le jeu de donnÃ©es en utilisant minmax_scale!
    #
    # Stockez les performances obtenues (prÃ©cision moyenne pour chaque valeur de k)
    # dans deux listes, scoresUniformWeights pour weights=uniform et 
    # scoresDistanceWeights pour weights=distance
    # Le premier Ã©lÃ©ment de chacune de ces listes devrait contenir la prÃ©cision
    # pour k=1, le second la prÃ©cision pour k=3, et ainsi de suite.
    scoresUniformWeights = []
    scoresDistanceWeights = []

    data = load_breast_cancer()
    X = data.data
    y = data.target
    X = minmax_scale(X)
    loo = LeaveOneOut()
    weights = [1, 3, 5, 7, 11, 13, 15, 25, 35, 45]
    
    for k in weights:
        clf1 = KNeighborsClassifier(weights='uniform', n_neighbors= k)
        clf2 = KNeighborsClassifier(weights='distance', n_neighbors= k)
        sum1 = 0
        sum2 = 0
        for train_index, test_index in loo.split(X):
           X_train, X_test = X[train_index], X[test_index]
           y_train, y_test = y[train_index], y[test_index]
           clf1.fit(X_train, y_train)
           clf2.fit(X_train, y_train)
           sum1 += clf1.score(X_test, y_test)
           sum2 += clf2.score(X_test, y_test)
        scoresUniformWeights.append(sum1 / float(len(X)))
        scoresDistanceWeights.append(sum2 / float(len(X)))
    
    _times.append(time.time())
    checkTime(TMAX_Q2Ebc, "2Ebc")

    # TODO Q2E
    # Produisez un graphique contenant deux courbes, l'une pour weights=uniform
    # et l'autre pour weights=distance. L'axe x de la figure doit Ãªtre le nombre
    # de voisins et l'axe y la performance en leave-one-out
    f, (ax1, ax2) = pyplot.subplots(1, 2, sharex=True)
    ax1.plot(weights, scoresUniformWeights)
    ax2.plot(weights, scoresDistanceWeights)
    
    ax1.set_title('uniform cancer')
    ax2.set_title('distance cancer')

    pyplot.show()


    _times.append(time.time())
    # TODO Q2E
    # Testez un classifeur K plus proches voisins sur Iris
    # L'Ã©valuation doit Ãªtre faite en utilisant une approche leave-one-out
    # Testez avec k = {1, 3, 5, 7, 11, 13, 15, 25, 35, 45} et avec les valeurs
    # "uniform" et "distance" comme valeur de l'argument "weights".
    # N'oubliez pas de normaliser le jeu de donnÃ©es en utilisant minmax_scale!
    #
    # Stockez les performances obtenues (prÃ©cision moyenne pour chaque valeur de k)
    # dans deux listes, scoresUniformWeights pour weights=uniform et 
    # scoresDistanceWeights pour weights=distance
    # Le premier Ã©lÃ©ment de chacune de ces listes devrait contenir la prÃ©cision
    # pour k=1, le second la prÃ©cision pour k=3, et ainsi de suite.
    scoresUniformWeights = []
    scoresDistanceWeights = []

    data = load_iris()
    X = data.data
    y = data.target
    X = minmax_scale(X)
    loo = LeaveOneOut()
    weights = [1, 3, 5, 7, 11, 13, 15, 25, 35, 45]
    
    for k in weights:
        clf1 = KNeighborsClassifier(weights='uniform', n_neighbors= k)
        clf2 = KNeighborsClassifier(weights='distance', n_neighbors= k)
        sum1 = 0
        sum2 = 0
        for train_index, test_index in loo.split(X):
           X_train, X_test = X[train_index], X[test_index]
           y_train, y_test = y[train_index], y[test_index]
           clf1.fit(X_train, y_train)
           clf2.fit(X_train, y_train)
           sum1 += clf1.score(X_test, y_test)
           sum2 += clf2.score(X_test, y_test)
        scoresUniformWeights.append(sum1 / float(len(X)))
        scoresDistanceWeights.append(sum2 / float(len(X)))
    
    
    ax1.set_title('uniform Iris')
    ax2.set_title('distance Iris')


    _times.append(time.time())
    checkTime(TMAX_Q2Eiris, "2Eiris")

    
    # TODO Q2E
    # Produisez un graphique contenant deux courbes, l'une pour weights=uniform
    # et l'autre pour weights=distance. L'axe x de la figure doit Ãªtre le nombre
    # de voisins et l'axe y la performance en leave-one-out
    f, (ax1, ax2) = pyplot.subplots(1, 2, sharex=True)
    ax1.plot(weights, scoresUniformWeights)
    ax2.plot(weights, scoresDistanceWeights)

    
    pyplot.show()



# N'Ã©crivez pas de code Ã  partir de cet endroit
