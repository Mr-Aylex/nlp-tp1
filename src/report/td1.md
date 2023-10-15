Rapport 1
Ce rapport va detaillé les differentes etapes de la premiere partie du TD1.

Dans un premier temps, nous avons mis en place un modele de RandomForestClassifier avec un CountVectorizer. Nous avons ensuite fait varier les parametres de ce modele (n_estimators, max_depth, max_features) et avons obtenu les resultats suivants :

Sans preprocessing des donnees :
Got accuracy [82.0, 89.5, 96.5, 94.0, 94.9748743718593] with mean 91.39497487437185%

Avec preprocessing des donnees (En supprimant les stop words et en stemmant les mots) :
Got accuracy [80.5, 88.0, 92.5, 92.5, 87.93969849246231] with mean 88.28793969849247%

Avec uniquement la stemmatisation des mots :
Got accuracy [83.0, 90.0, 96.0, 95.5, 94.9748743718593] with mean 91.89497487437187%

Avec uniquement la suppression des stop words :
Got accuracy [83.0, 90.5, 96.0, 93.5, 94.9748743718593] with mean 91.59497487437186%

Avec uniquement la suppression de la ponctuation :
Got accuracy [82.0, 88.5, 94.0, 93.0, 91.4572864321608] with mean 89.79145728643216%


Par la suite nous avons mis en place un modele GradientBoostingClassifier

Sans preprocessing des donnees :
Got accuracy [82.5, 93.5, 97.5, 95.5, 95.47738693467338] with mean 92.89547738693467% 

Avec preprocessing des donnees (En supprimant les stop words et en stemmant les mots) :
Got accuracy [83.0, 94.0, 96.5, 95.0, 94.47236180904522] with mean 92.59447236180904%

Avec uniquement la stemmatisation des mots :
Got accuracy [83.0, 94.0, 97.0, 96.5, 95.97989949748744] with mean 93.29597989949748%

Avec uniquement la suppression des stop words :
Got accuracy [82.5, 94.5, 98.5, 96.0, 94.9748743718593] with mean 93.29497487437186%

Avec uniquement la suppression de la ponctuation :
Got accuracy [82.5, 93.0, 96.5, 97.0, 95.47738693467338] with mean 92.89547738693467% 

