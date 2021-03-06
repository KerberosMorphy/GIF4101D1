#!/usr/bin/env python2
# -*- coding: utf-8 -*-
###############################################################################
# Apprentissage et reconnaissance
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 2, Question 1
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
import numpy as np

from matplotlib import pyplot

from scipy.stats import norm

from sklearn.neighbors import KernelDensity

# Fonctions utilitaires liÃ©es Ã  l'Ã©valuation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps Ã  s'exÃ©cuter! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple Ã  show()) dans cette boucle!") 

# DÃ©finition des durÃ©es d'exÃ©cution maximales pour chaque sous-question
TMAX_Q1A = 1.0
TMAX_Q1B = 2.5


# Ne changez rien avant cette ligne!


# DÃ©finition de la PDF de la densitÃ©-mÃ©lange
def pdf(X):
    return 0.4 * norm(0, 1).pdf(X[:, 0]) + 0.6 * norm(5, 1).pdf(X[:, 0])


# Question 1A

# TODO Q1A
# ComplÃ©tez la fonction sample(n), qui gÃ©nÃ¨re n
# donnÃ©es suivant la distribution mentionnÃ©e dans l'Ã©noncÃ©
def sample(n):
    arr = np.random.rand(n)
    norm1 = norm(0, 1)
    norm2 = norm(5, 1)
    
    arr = map(lambda x: norm1.rvs() if x < 0.4 else norm2.rvs(), arr)
    return arr


if __name__ == '__main__':
    # Question 1A

    _times.append(time.time())
    # TODO Q1A
    # Ã‰chantillonnez 50 et 10 000 donnÃ©es en utilisant la fonction
    # sample(n) que vous avez dÃ©finie plus haut et tracez l'histogramme
    # de cette distribution Ã©chantillonÃ©e, en utilisant 25 bins,
    # dans le domaine [-5, 10].
    # Sur les mÃªmes graphiques, tracez Ã©galement la fonction de densitÃ© rÃ©elle.
    f, (ax1, ax2) = pyplot.subplots(1, 2, sharex=True)
    
    curve = pdf(np.arange(-5, 10, 0.01).reshape((1500, 1)))
    
    samp1 = sample(50)
    ax1.hist(samp1,bins = 25, density = True)
    ax1.plot(np.arange(-5, 10, 0.01), curve, c= 'r')
    
    samp2 = sample(10000)
    ax2.hist(samp2,bins = 25, density = True)
    ax2.plot(np.arange(-5, 10, 0.01), curve, c= 'r')
    
    
    ax1.set_title('n = 50')
    ax2.set_title('n = 10000')    

    # Affichage du graphique
    _times.append(time.time())
    checkTime(TMAX_Q1A, "1A")
    pyplot.show()


    # Question 1B
    _times.append(time.time())
    
    # TODO Q1B
    # Ã‰chantillonnez 50 et 10 000 donnÃ©es, mais utilisez cette fois une
    # estimation avec noyau boxcar pour prÃ©senter les donnÃ©es. Pour chaque
    # nombre de donnÃ©es (50 et 10 000), vous devez prÃ©senter les distributions
    # estimÃ©es avec des tailles de noyau (bandwidth) de {0.3, 1, 2, 5}, dans
    # la mÃªme figure, mais tracÃ©es avec des couleurs diffÃ©rentes.
        
    f, (ax1, ax2) = pyplot.subplots(1, 2, sharex=True)
    
    kernels = map(lambda bandwidth: KernelDensity(bandwidth, kernel = 'tophat'), [0.3, 1, 2, 5])
    
    s1 = np.array(samp1).reshape(-1, 1)    
    s2 = np.array(samp2).reshape(-1, 1)    
    xgrid = np.arange(-5, 10, 0.01).reshape(-1,1)
    for (k, color) in zip(kernels, "rgby"):
        k.fit(s1)
        arr = k.score_samples(xgrid)
        arr = np.exp(arr)
        ax1.plot(xgrid, arr, c = color)
        k.fit(s2)
        arr = k.score_samples(xgrid)
        arr = np.exp(arr)
        ax2.plot(xgrid, arr, c = color)
       
        
    ax1.set_title('n = 50')
    ax2.set_title('n = 10000')    
    ax2.legend(['bw = 0.3', 'bw = 1', 'bw = 2', 'bw = 5'])
    ax1.legend(['bw = 0.3', 'bw = 1', 'bw = 2', 'bw = 5'])
    # Affichage du graphique
    _times.append(time.time())
    checkTime(TMAX_Q1B, "1B")
    pyplot.show()



# N'Ã©crivez pas de code Ã  partir de cet endroit

