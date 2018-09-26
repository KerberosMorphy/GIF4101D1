###############################################################################
# Introduction a l'apprentissage machine
# GIF-4101 / GIF-7005, Automne 2018
# Devoir 1, Question 3
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
from sklearn.datasets import make_moons, load_iris

# Fonctions utilitaires liees a l'evaluation
_times = []
def checkTime(maxduration, question):
    duration = _times[-1] - _times[-2]
    if duration > maxduration:
        print("[ATTENTION] Votre code pour la question {0} met trop de temps a s'executer! ".format(question)+
            "Le temps maximum permis est de {0:.4f} secondes, mais votre code a requis {1:.4f} secondes! ".format(maxduration,duration)+
            "Assurez-vous que vous ne faites pas d'appels bloquants (par exemple a show()) dans cette boucle!") 

TMAX_Q3D = 1.0


# Ne modifiez rien avant cette ligne!


# Question 3C
class ClassifieurAvecRejet:
    
    def __init__(self, _lambda=1):
        # _lambda est le cout de rejet
        self._lambda = _lambda
    

    def fit(self, X, y):
        # TODO Q3C
        # Implementez ici une fonction permettant d'entrainer votre modele
        # a partir des donnees fournies en argument
    
        #Initialisation des tableaux qui stockeront les parametres
        means = numpy.zeros((len(numpy.unique(y)), numpy.shape(X)[1]))
        sds = numpy.zeros(len(numpy.unique(y)))        
        proportion = numpy.zeros(len(numpy.unique(y)))        
        
        #Pour chaque classe on calcule les parametres selon la methode des EMV
        for k in numpy.unique(y):
            #On extrait les donnees de la classe k            
            indexes = numpy.where(y == k)
            classData = X[indexes]
            
            #Calcul de Moyenne
            mu = numpy.mean(classData, axis = 0)  
            
            #Calcul de Sigma
            normSum = numpy.sum((classData - mu)**2)
            sigma = normSum / (numpy.shape(classData)[0] * numpy.shape(classData)[1])
            sigma = sigma ** 0.5
            
            means[k] = mu
            sds[k] = sigma
            proportion[k] = (float)(len(indexes[0]))/(float)(len(X))
            
    
        self.proportion = proportion
        self.means = means
        self.sds = sds
        pass

    
    def predict_proba(self, X):
        # TODO Q3C
        # Implementez une fonction retournant la probabilite d'appartenance a chaque
        # classe, pour les donnees passees en argument. Cette fonction peut supposer 
        # que fit() a prealablement ete appele.
        # Indice : calculez les differents termes de l'equation de Bayes separement

        N = numpy.shape(X)[1]
        probas = numpy.zeros((len(X), len(self.sds)))
        
        #On calcule pour chaque classe k le numerateur de la formule de Bayes
        #La formule utilisee est dans le rapport
        for k in range(len(self.proportion)):
            const = (2 * numpy.pi) ** (-0.5 * N)
            det = self.sds[k] ** (-N)
            dist = X - self.means[k]
            dist = -0.5 * numpy.sum(dist ** 2, axis = 1)
            dist = dist * (1 / self.sds[k] ** 2)
            dist = numpy.exp(dist)

            probas[:, k] = self.proportion[k] * (const * det * dist)            
                
        #On divise par l'evidence pour avoir une somme des probabilites egales a 1
        for i in range(len(probas)) : 
            probas[i,:] = probas[i,:]/numpy.sum(probas[i,:])
        
        self.probas = probas
        pass


    def predict(self, X):
        # TODO Q3C
        # Implementez une fonction retournant les predictions pour les donnees
        # passees en argument. Cette fonction peut supposer que fit() a prealablement
        # ete appele.
        # Indice : vous pouvez utiliser predict_proba() pour eviter une redondance du code
    
        self.predict_proba(X)    
    
        #On a explique la regle de decision dans le rapport
        #Ici on va donc choisir la classe avec la probabilite la plus forte 
        #en incluant l'option de rejet selon lambda
        probas = self.probas
        pred = numpy.zeros(len(probas))        
        for i in range(len(probas)):
            pred[i] = numpy.argmax(probas[i,:]) if numpy.max(probas[i,:]) > (1 - self._lambda) else -1 
                
        self.prediction = pred 
        return pred
        pass
    
    
    def score(self, X, y):
        # TODO Q3C
        # Implementez une fonction retournant le score (tenant compte des donnees
        # rejetees si lambda < 1.0) pour les donnees passees en argument.
        # Cette fonction peut supposer que fit() a prealablement ete execute.
        
        
        self.predict(X)            
        pred = self.prediction
        
        #On calcule d'abord le score impute a l'option de rejet
        score_reject = self._lambda * float(len(numpy.where(pred == -1)[0])) / float(len(X))        
        #Puis le score impute a un choix de classe correct        
        self.score = float(len(numpy.where(pred == y)[0])) / float(len(X))
        #Enfin on additione
        self.score += score_reject        
                        
        print("lambda = " + str(self._lambda) + " : " + str(float(1 - self.score)))
        return self.score

        pass


# Question 3D
if __name__ == "__main__":
    
    # TODO Q3D
    # Chargez ici le dataset 'iris' dans une variable nommee data
    data = load_iris()

    # Cette ligne cree une liste contenant toutes les paires 
    # possibles entre les 4 mesures
    pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]

    # Utilisons cette liste de paires pour tester le classifieur
    # avec differents lambda
    for (f1, f2) in pairs:
        # TODO Q3D
        # Creez ici un sous-dataset contenant seulement les
        # mesures designees par f1 et f2
        subData = data.data[:, (f1,f2)]
        # TODO Q3D
        # Creez ici une grille permettant d'afficher les regions de
        # decision pour chaque classifieur
        # Indice : numpy.meshgrid pourrait vous etre utile ici
        # N'utilisez pas un pas trop petit!
        
        x = numpy.arange(min(subData[:,0]), max(subData[:,0]), 0.05)
        y = numpy.arange(min(subData[:,1]), max(subData[:,1]), 0.05)
        xx, yy = numpy.meshgrid(x, y)            
        
        # On initialise les classifieurs avec differents parametres lambda
        classifieurs = [ClassifieurAvecRejet(0.1),
                        ClassifieurAvecRejet(0.3),
                        ClassifieurAvecRejet(0.5),
                        ClassifieurAvecRejet(1)]
        
        # On cree une figure a plusieurs sous-graphes pour pouvoir montrer,
        # pour chaque configuration, les regions de decisions, incluant
        # la zone de rejet
        fig, subfigs = pyplot.subplots(2, 2, sharex='all', sharey='all')
        _times.append(time.time())
        for clf,subfig in zip(classifieurs, subfigs.reshape(-1)):
            
            # TODO Q3D
            # Entrainez le classifieur
            clf.fit(subData, data.target)
            # TODO Q3D
            # Obtenez et affichez son score
            # Stockez la valeur de cette erreur dans la variable err
            err = 1 - clf.score(subData, data.target)
            
            # TODO Q3D
            # Utilisez la grille que vous avez creee plus haut
            # pour afficher les regions de decision, INCLUANT LA
            # ZONE DE REJET, de meme que les points colores selon 
            # leur vraie classe
            pred = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
            
            pred = pred.reshape(xx.shape)
            cs = subfig.contourf(xx, yy, pred, cmap=pyplot.cm.Paired)
            
            colors = "brg"
            for i in numpy.unique(data.target) : 
                indexes = numpy.where(data.target == i)
                subfig.plot(subData[indexes,0], subData[indexes,1], '+', c = colors[i])
            
            # On ajoute un titre et des etiquettes d'axes
            subfig.set_title("lambda="+str(clf._lambda))
            subfig.set_xlabel(data.feature_names[f1])
            subfig.set_ylabel(data.feature_names[f2])
        _times.append(time.time())
        checkTime(TMAX_Q3D, "3D")
        
        # On affiche les graphiques
        pyplot.show()



# N'ecrivez pas de code a partir de cet endroit
