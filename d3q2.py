# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 3, Question 2
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

import itertools
import time
import numpy
import warnings
from io import BytesIO
from http.client import HTTPConnection

from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

# Nous ne voulons pas avoir ce type d'avertissement, qui
# n'est pas utile dans le cadre de ce devoir
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!") 

# DÃ©finition des durÃ©es d'exÃ©cution maximales pour chaque sous-question
TMAX_KNN = 40
TMAX_SVM = 200
TMAX_PERCEPTRON = 400
TMAX_EVAL = 80

def fetchPendigits():
    """
    Cette fonction tÃ©lÃ©charge le jeu de donnÃ©es pendigits et le
    retourne sous forme de deux tableaux numpy. Le premier Ã©lÃ©ment
    retournÃ© par cette fonction est un tableau de 10992x16 qui
    contient les samples; le second Ã©lÃ©ment est un vecteur de 10992
    qui contient les valeurs cible (target).
    """
    host = 'vision.gel.ulaval.ca'
    url = '/~cgagne/enseignement/apprentissage/A2018/travaux/ucipendigits.npy'
    connection = HTTPConnection(host, port=80, timeout=10)
    connection.request('GET', url)

    rep = connection.getresponse()
    if rep.status != 200:
        print("ERREUR : impossible de tÃ©lÃ©charger le jeu de donnÃ©es UCI Pendigits! Code d'erreur {}".format(rep.status))
        print("VÃ©rifiez que votre ordinateur est bien connectÃ© Ã  Internet.")
        return
    stream = BytesIO(rep.read())
    dataPendigits = numpy.load(stream)
    return dataPendigits[:, :-1].astype('float32'), dataPendigits[:, -1]

# Ne modifiez rien avant cette ligne!



if __name__ == "__main__":
    # Question 2B

    # TODO Q2B
    # Chargez le jeu de donnÃ©es Pendigits. Utilisez pour cela la fonction
    # fetchPendigits fournie. N'oubliez pas de normaliser
    # les donnÃ©es d'entrÃ©e entre 0 et 1 pour toutes les dimensions.
    # Notez finalement que fetch_openml retourne les donnÃ©es d'une maniÃ¨re
    # diffÃ©rente des fonctions load_*, assurez-vous que vous utilisez
    # correctement les donnÃ©es et qu'elles sont du bon type.
    pendigits = fetchPendigits()
    y = pendigits[1]
    X = minmax_scale(pendigits[0])
    
    # TODO Q2B
    # SÃ©parez le jeu de donnÃ©es Pendigits en deux sous-jeux: entraÃ®nement (5000) et
    # test (reste des donnÃ©es). Pour la suite du code, rappelez-vous que vous ne
    # pouvez PAS vous servir du jeu de test pour dÃ©terminer la configuration
    # d'hyper-paramÃ¨tres la plus performante. Ce jeu de test ne doit Ãªtre utilisÃ©
    # qu'Ã  la toute fin, pour rapporter les rÃ©sultats finaux en gÃ©nÃ©ralisation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=5000, test_size=len(y)-5000, random_state=0)    


    # TODO Q2B
    # Pour chaque classifieur :
    # - k plus proches voisins,
    # - SVM Ã  noyau gaussien,
    # - Perceptron multicouche,
    # dÃ©terminez les valeurs optimales des hyper-paramÃ¨tres Ã  utiliser.
    # Suivez les instructions de l'Ã©noncÃ© quant au nombre d'hyper-paramÃ¨tres Ã 
    # optimiser et n'oubliez pas d'expliquer vos choix d'hyper-paramÃ¨tres
    # dans votre rapport.
    # Vous Ãªtes libres d'utiliser la mÃ©thodologie que vous souhaitez, en autant
    # que vous ne touchez pas au jeu de test.
    #
    # Note : optimisez les hyper-paramÃ¨tres des diffÃ©rentes mÃ©thodes dans
    # l'ordre dans lequel ils sont Ã©numÃ©rÃ©s plus haut, en insÃ©rant votre code
    # d'optimisation entre les commentaires le spÃ©cifiant
    

    
    _times.append(time.time())
    # TODO Q2B
    # Optimisez ici la paramÃ©trisation du kPP
    
    kvalues = numpy.arange(1,30)
    weights = ['uniform', 'distance']
    scoresKNN = numpy.zeros((len(kvalues), len(weights)))
    
    for i in range(len(kvalues)):
        for j in range(len(weights)):
            kf = KFold(n_splits=10)
            score = 0
            for train_index, validate_index in kf.split(X_train):    
                clf = KNeighborsClassifier(n_neighbors=kvalues[i], weights=weights[j])
                clf.fit(X_train[train_index,:], y_train[train_index])
                score += clf.score(X_train[validate_index,:], y_train[validate_index])
            score = score/10.0
            scoresKNN[i,j] = score
    
    maxIndxKNN = (numpy.argmax(scoresKNN)%scoresKNN.shape[0],
               numpy.argmax(scoresKNN) - (numpy.argmax(scoresKNN)%scoresKNN.shape[0])*scoresKNN.shape[1])
    
    _times.append(time.time())
    checkTime(TMAX_KNN, "K plus proches voisins")
    # TODO Q2B
    # Optimisez ici la paramÃ©trisation du SVM Ã  noyau gaussien
    
    C = list(map(lambda x: 10**float(x) , numpy.arange(10) - 5))    
    max_iter = list(map(lambda x: 1000 * x, [1]))
    #Gamma ?
    scoresSVM = numpy.zeros((len(C), len(max_iter)))
    
    for i in range(len(C)):
        for j in range(len(max_iter)):
            kf = KFold(n_splits=3)
            score = 0
            for train_index, validate_index in kf.split(X_train):    
                clf = SVC(C=C[i], max_iter=max_iter[j], gamma = 'scale')
                clf.fit(X_train[train_index,:], y_train[train_index])
                score += clf.score(X_train[validate_index,:], y_train[validate_index])
            score = score/3.0
            scoresSVM[i,j] = score
            
    maxIndxSVM = (numpy.argmax(scoresSVM)%scoresSVM.shape[0],
               numpy.argmax(scoresSVM) - (numpy.argmax(scoresSVM)%scoresSVM.shape[0])*scoresSVM.shape[1])
    


    _times.append(time.time())
    checkTime(TMAX_SVM, "SVM")
    # TODO Q2B
    # Optimisez ici la paramÃ©trisation du perceptron multicouche
    # Note : il se peut que vous obteniez ici des "ConvergenceWarning"
    # Ne vous en souciez pas et laissez le paramÃ¨tre max_iter Ã  sa
    # valeur suggÃ©rÃ©e dans l'Ã©noncÃ© (100)
    
    #numpy.array(list(itertools.permutations([1, 1, 3])))
    #itertools.combinations([1,2,3], 2)
    
    #itertools.combinations_with_replacement([1,2,3,4,5], 5)
    nbMaxLayers = numpy.arange(2,5) 
    nbMaxNeurons = numpy.arange(2,8) * 16
    
    scoresNN = numpy.zeros((len(nbMaxLayers), len(nbMaxNeurons)))
    
    
    for i in range(len(nbMaxLayers)):
        for j in range(len(nbMaxNeurons)):
            kf = KFold(n_splits=3)
            score = 0
            for train_index, validate_index in kf.split(X_train):    
                clf = MLPClassifier(hidden_layer_sizes=numpy.ones(nbMaxLayers[i], dtype=numpy.int32) * nbMaxNeurons[j], max_iter=100)
                clf.fit(X_train[train_index,:], y_train[train_index])
                score += clf.score(X_train[validate_index,:], y_train[validate_index])
            score = score/3.0
            scoresNN[i,j] = score
            
    #maxIndxNN = (numpy.argmax(scoresNN)%scoresNN.shape[0],
    #           numpy.argmax(scoresNN) - (numpy.argmax(scoresNN)%scoresNN.shape[0])*scoresNN.shape[1])
    maxIndxNN = numpy.unravel_index(numpy.argmax(scoresNN), scoresNN.shape)
    
    '''
    neoronShapes = numpy.array([])
    for i in nbMaxLayers:
        for j in nbMaxNeurons:
            for neuronShape in itertools.combinations_with_replacement(numpy.arange(1,j+1), i+1):
                print(neuronShape)
                kf = KFold(n_splits=3)
                score = 0
                for train_index, validate_index in kf.split(X_train):    
                    clf = MLPClassifier(hidden_layer_sizes=neuronShape, max_iter=100, activation='logistic')
                    clf.fit(X_train[train_index,:], y_train[train_index])
                    score += clf.score(X_train[validate_index,:], y_train[validate_index])
                score = score/3.0
                clf = MLPClassifier(hidden_layer_sizes=neuronShape, max_iter=100, activation='logistic')
                clf.fit(X_train, y_train)
                score = clf.score(X_train, y_train)
                scores = numpy.append(scores, score)
    '''

    _times.append(time.time())
    checkTime(TMAX_PERCEPTRON, "SVM")
    

    # TODO Q2B
    # Ã‰valuez les performances des meilleures paramÃ©trisations sur le jeu de test
    # et rapportez ces performances dans le rapport
    clf = KNeighborsClassifier(n_neighbors=kvalues[maxIndxKNN[0]], weights=weights[maxIndxKNN[1]])
    clf.fit(X_train, y_train)
    print('KNN : ' + str(clf.score(X_test, y_test)))
    print('k : '+str(kvalues[maxIndxKNN[0]]))
    print('weights : '+str(weights[maxIndxKNN[1]]))
    
    clf = SVC(C=C[maxIndxSVM[0]], max_iter=max_iter[maxIndxSVM[1]], gamma = 'scale')
    clf.fit(X_train, y_train)
    print('SVM : ' + str(clf.score(X_test, y_test)))
    print('C : '+str(C[maxIndxSVM[0]]))
    print('max iter : '+str(max_iter[maxIndxSVM[1]]))
    
    clf = MLPClassifier(hidden_layer_sizes=tuple(numpy.ones(nbMaxLayers[maxIndxNN[0]], dtype=numpy.int32) * nbMaxNeurons[maxIndxNN[1]]), max_iter=100)
    clf.fit(X_train, y_train)
    print('NN : ' + str(clf.score(X_test, y_test)))
    print('nb layers : '+str(nbMaxLayers[maxIndxNN[0]]))
    print('nb neuron : '+str(nbMaxNeurons[maxIndxNN[1]]))
    

    _times.append(time.time())
    checkTime(TMAX_EVAL, "Evaluation des modÃ¨les")
# N'Ã©crivez pas de code Ã  partir de cet endroit
