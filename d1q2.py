###############################################################################
# Introduction a l'apprentissage machine
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 1, Question 2
#
###############################################################################
############################## INSTRUCTIONS ###################################
###############################################################################
#
# - Reperez les commentaires commencant par TODO : ils indiquent une tache que
#       vous devez effectuer.
# - Vous ne pouvez PAS changer la structure du code, importer d'autres
#       modules / sous-modules, ou ajouter d'autres fichiers Python
# - Ne touchez pas aux variables, TMAX*, ERRMAX* et _times, a la fonction
#       checkTime, ni aux conditions verifiant le bon fonctionnement de votre 
#       code. Ces structures vous permettent de savoir rapidement si vous ne 
#       respectez pas les requis minimum pour une question en particulier. 
#       Toute sous-question n'atteignant pas ces minimums se verra attribuer 
#       la note de zero (0) pour la partie implementation!
#
###############################################################################
import time
import numpy

from matplotlib import pyplot

# Jeu de donnees utilises
from sklearn.datasets import load_iris, make_circles

# Classifieurs utilises
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid

# Methodes d'evaluation
from sklearn.model_selection import train_test_split, RepeatedKFold

# Fonctions utilitaires liees a l'evaluation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps a s'executer! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple a show()) dans cette boucle!")

# Definition des durees d'execution maximales pour chaque sous-question
TMAX_Q2A = 0.5
TMAX_Q2B = 1.5
TMAX_Q2Cii = 0.5
TMAX_Q2Ciii = 0.5
TMAX_Q2D = 1.0

# Definition des erreurs maximales attendues pour chaque sous-question
ERRMAX_Q2B = 0.22
ERRMAX_Q2Cii = 0.07
ERRMAX_Q2Ciii = 0.07


# Ne changez rien avant cette ligne!
# Seul le code suivant le "if __name__ == '__main__':" comporte des sections a implementer

if __name__ == '__main__':
    # Question 2A
    # TODO Q2A
    # Chargez ici le dataset 'iris' dans une variable nommee data
    data = load_iris()

    # Cette ligne cree une liste contenant toutes les paires 
    # possibles entre les 4 mesures
    # Par exemple : [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]

    # Utilisons cette liste de paires pour afficher les donnees,
    # deux mesures a la fois
    # On cree une figure a plusieurs sous-graphes
    fig, subfigs = pyplot.subplots(2, 3)
    _times.append(time.time())
    
    for (f1, f2), subfig in zip(pairs, subfigs.reshape(-1)):
        # TODO Q2A
        # Affichez les donnees en utilisant f1 et f2 comme mesures
        
        #On plot une fois pour chaque classe pour bien afficher les couleurs differentes 
        colors = "brg"
        for i in numpy.unique(data.target) : 
            indexes = numpy.where(data.target == i)
            subfig.plot(data.data[indexes,f1], data.data[indexes,f2], '+', c = colors[i])
        
        subfig.set_xlabel(data.feature_names[f1])
        subfig.set_ylabel(data.feature_names[f2])

        pass
    _times.append(time.time())
    checkTime(TMAX_Q2A, "2A")

    # On affiche la figure
    pyplot.show()


    # Question 2B
    # Reprenons les paires de mesures, mais entrainons cette fois
    # differents modeles demandes avec chaque paire
    for (f1, f2) in pairs:
        # TODO Q2B
        # Creez ici un sous-dataset contenant seulement les
        # mesures designees par f1 et f2
        subData = data.data[:, (f1,f2)]

        # TODO Q2B
        # Initialisez ici les differents classifieurs, dans
        # une liste nommee "classifieurs"
        classifieurs = [
         QuadraticDiscriminantAnalysis(),
         LinearDiscriminantAnalysis(),
         GaussianNB(),
         NearestCentroid()
        ]
        

        # TODO Q2B
        # Creez ici une grille permettant d'afficher les regions de
        # decision pour chaque classifieur
        # Indice : numpy.meshgrid pourrait vous etre utile ici
        # N'utilisez pas un pas trop petit!

        x = numpy.arange(min(subData[:,0]), max(subData[:,0]), 0.05)
        y = numpy.arange(min(subData[:,1]), max(subData[:,1]), 0.05)
        xx, yy = numpy.meshgrid(x, y)        

    

        # On cree une figure a plusieurs sous-graphes
        fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
        _times.append(time.time())
        for clf,subfig in zip(classifieurs, subfigs.reshape(-1)):
            # TODO Q2B
            # Entrainez le classifieur
            clf.fit(subData, data.target)            
            
            # TODO Q2B
            # Obtenez et affichez son erreur (1 - accuracy)
            # Stockez la valeur de cette erreur dans la variable err
            err = 1 - clf.score(subData, data.target) 
            print("erreur " + str(clf.__class__.__name__) + " : " + str(err))
            
            # TODO Q2B
            # Utilisez la grille que vous avez creee plus haut
            # pour afficher les regions de decision, de meme
            # que les points colores selon leur vraie classe
            pred = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
            pred = pred.reshape(xx.shape)
            cs = subfig.contourf(xx, yy, pred, cmap=pyplot.cm.Paired)
            
            #On affiche les points par dessus les regions de decisions
            #Attention, les couleurs sont plus ou moins randoms
            colors = "brg"
            for i in numpy.unique(data.target) : 
                indexes = numpy.where(data.target == i)
                subfig.plot(subData[indexes,0], subData[indexes,1], '+', c = colors[i])
            
            
            
            # Identification des axes et des methodes
            subfig.set_xlabel(data.feature_names[f1])
            subfig.set_ylabel(data.feature_names[f2])
            subfig.set_title(clf.__class__.__name__)
            if err > ERRMAX_Q2B:
                print("[ATTENTION] Votre code pour la question 2B ne produit pas les performances attendues! "+
                      "Le taux d'erreur maximal attendu est de {0:.3f}, mais l'erreur rapportee dans votre code est de {1:.3f}!".format(ERRMAX_Q2B,err))
            
        _times.append(time.time())
        checkTime(TMAX_Q2B, "2B")
        
        # On affiche les graphiques
        pyplot.show()


    # Question 2C
    # Note : Q2C (i) peut etre repondue en utilisant le code precedent
  
    #On fait la moyenne de l'erreur
    clf = QuadraticDiscriminantAnalysis()
    err = 0
    n = 10
    for i in range(n):        
        clf.fit(data.data, data.target)
        err = err + (1 - clf.score(data.data, data.target))                
    avgError = err / n
    print("avg error train = test", avgError)
    
    
    _times.append(time.time())
    # TODO Q2Cii
    # ecrivez ici le code permettant de partitionner les donnees en jeux
    # d'entrainement / de validation et de tester la performance du classifieur
    # mentionne dans l'enonce
    # Vous devez repeter cette mesure 10 fois avec des partitions differentes
    # Stockez l'erreur moyenne sur ces 10 iterations dans une variable nommee avgError

    clf = QuadraticDiscriminantAnalysis()
    err = 0
    n = 10
    for i in range(n):
        #L'argument shuffle nous assure que les partitions sont aleatoires        
        X_train, X_test, Y_train, Y_test = train_test_split(data.data, data.target, test_size = 0.5, shuffle = True)
        clf.fit(X_train, Y_train)
        err = err + (1 - clf.score(X_test, Y_test))                
    avgError = err / n
    print("avg error 50/50 holdout", avgError)

    _times.append(time.time())
    checkTime(TMAX_Q2Cii, "2Cii")
        
    if avgError > ERRMAX_Q2Cii:
        print("[ATTENTION] Votre code pour la question 2C ii) ne produit pas les performances attendues! "+
              "Le taux d'erreur maximal attendu est de {0:.3f}, mais l'erreur rapportee dans votre code est de {1:.3f}!".format(ERRMAX_Q2Cii,avgError))

    
    _times.append(time.time())
    # TODO Q2Ciii
    # ecrivez ici le code permettant de determiner la performance du classifieur
    # avec un K-fold avec k=3.
    # Vous devez repeter le K-folding 10 fois
    # Stockez l'erreur moyenne sur ces 10 iterations dans une variable nommee avgError
    
    n_splits=3
    n_repeats=10
    rkf = RepeatedKFold(n_splits, n_repeats)
    err = 0
    
    for train_index, test_index in rkf.split(data.target):
        X_train, X_test = data.data[train_index, :], data.data[test_index, :]
        Y_train, Y_test = data.target[train_index], data.target[test_index]    
        clf.fit(X_train, Y_train)
        err = err + (1 - clf.score(X_test, Y_test))   
        
    avgError = err / (n_repeats * n_splits)    
    print("avg error cross-val", avgError)
    _times.append(time.time())
    checkTime(TMAX_Q2Ciii, "2Ciii")

    if avgError > ERRMAX_Q2Ciii:
        print("[ATTENTION] Votre code pour la question 2C iii) ne produit pas les performances attendues! "+
              "Le taux d'erreur maximal attendu est de {0:.3f}, mais l'erreur rapportee dans votre code est de {1:.3f}!".format(ERRMAX_Q2Ciii,avgError))


    # Question 2D
    # TODO Q2D
    # Initialisez ici les differents classifieurs, dans
    # une liste nommee "classifieurs"
    
    classifieurs = [
         QuadraticDiscriminantAnalysis(),
         LinearDiscriminantAnalysis(),
         GaussianNB(),
         NearestCentroid()
    ]
    # Creation du jeu de donnees
    X, y = make_circles(factor=0.3)

    # TODO Q2D
    # Creez ici une grille permettant d'afficher les regions de
    # decision pour chaque classifieur
    # Indice : numpy.meshgrid pourrait vous etre utile ici
    # N'utilisez pas un pas trop petit!
    x1 = numpy.arange(min(X[:,0]), max(X[:,0]), 0.01)
    x2 = numpy.arange(min(X[:,1]), max(X[:,1]), 0.01)
    xx1, xx2 = numpy.meshgrid(x1, x2)        

    # On cree une figure a plusieurs sous-graphes
    fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
    _times.append(time.time())
    
    
    #Permet de s'assurer de conservaer la randomisation sur les quatres classifieurs
    seed = numpy.random.randint(0, 100000)    
    
    for clf,subfig in zip(classifieurs, subfigs.reshape(-1)):
        # TODO Q2D
        # Divisez le jeu de donnees de maniere deterministe,
        # puis entrainez le classifieur
        '''pk on split pas le dataset avant ?'''
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, shuffle = True, random_state = seed)
        clf.fit(X_train, Y_train)        
        # TODO Q2D
        # Obtenez et affichez son erreur (1 - accuracy)
        # Stockez la valeur de cette erreur dans la variable err
        err = 1 - clf.score(X_test, Y_test)
        print("erreur " + clf.__class__.__name__ + " : " + str(err))
        # TODO Q2D
        # Utilisez la grille que vous avez creee plus haut
        # pour afficher les regions de decision, de meme
        # que les points colores selon leur vraie classe
        
        #Affichage des contours
        pred = clf.predict(numpy.c_[xx1.ravel(), xx2.ravel()])
        pred = pred.reshape(xx1.shape)
        cs = subfig.contourf(xx1, xx2, pred, cmap=pyplot.cm.Paired)
        
        colors = "brg"
        for i in numpy.unique(y) : 
            indexes = numpy.where(y == i)
            subfig.plot(X[indexes,0], X[indexes,1], '+', c = colors[i])
       
       
       
        subfig.set_title(clf.__class__.__name__)
        
    _times.append(time.time())
    checkTime(TMAX_Q2D, "2D")

    pyplot.show()



# N'ecrivez pas de code a partir de cet endroit
