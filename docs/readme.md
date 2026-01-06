# Analyseur de Positions d'Échecs - Projet Réseau de Neurones

## Présentation du Projet
Ce projet implémente un réseau de neurones personnalisé à partir de zéro pour analyser des positions d'échecs. L'objectif principal est de classifier les positions (représentées au format FEN) en trois catégories : **Nothing** (Rien), **Check** (Échec), et **Checkmate** (Échec et Mat).

Le projet est conçu avec une architecture modulaire, permettant une configuration flexible, un parsing de dataset robuste et un entraînement efficace via des composants de réseau de neurones sur mesure.

## Architecture
Le système est composé de plusieurs modules clés :
- **`NeuralNetworkModule.py`** : Le moteur central contenant l'implémentation du réseau de neurones, incluant la propagation avant/arrière (forward/backward), les optimiseurs et la logique de persistance.
- **`parser_dataset.py`** : Un module spécialisé pour transformer les chaînes FEN en vecteurs de caractéristiques de haute dimension (784 caractéristiques).
- **`Parse_conf.py`** : Un parseur de configuration qui lit les fichiers `.conf` pour définir l'architecture du réseau et les paramètres d'entraînement.
- **`my_torch_analyzer`** : Le point d'entrée principal pour l'entraînement et la prédiction.

## Spécifications Techniques

### Fonctions d'Activation
Le réseau implémente plusieurs fonctions d'activation pour répondre aux besoins des différentes couches :
- **ReLU (Rectified Linear Unit)** : Utilisée dans les couches cachées pour introduire de la non-linéarité tout en évitant le problème de disparition du gradient.
- **Sigmoid** : Disponible pour la classification binaire ou des besoins spécifiques de couches cachées.
- **Tanh** : Fournit une activation centrée sur zéro, souvent utile pour une convergence plus rapide dans certaines architectures.
- **Softmax** : Appliquée à la couche de sortie pour la classification multi-classe, fournissant une distribution de probabilité sur les trois classes cibles.

### Parsing du Dataset & Ingénierie des Caractéristiques (Feature Engineering)
Pour fournir au réseau un contexte stratégique riche, chaque position d'échecs est convertie en un **vecteur de 784 caractéristiques** :
1. **Représentation de l'Échiquier (768 caractéristiques)** : Un encodage "one-hot" 8x8x12 représentant la présence de chaque type de pièce (64 cases × 12 types de pièces).
2. **Métadonnées & Stratégie (16 caractéristiques)** :
    - **État du Jeu** : Trait (1), Droits au roque (4), Disponibilité de la prise en passant (1), Horloges Halfmove/Fullmove (2).
    - **Sécurité du Roi** : 8 caractéristiques avancées incluant les cases d'échappatoire, les défenseurs, la pression ennemie et la centralité.

### Fonction de Perte (Loss Function)
- **Cross-Entropy Loss** : Utilisée pour les tâches de classification afin de mesurer la performance du modèle dont la sortie est une valeur de probabilité entre 0 et 1.

## Justification des Choix de Conception

### Pourquoi 784 Caractéristiques ?
Au lieu de ne fournir que l'état brut de l'échiquier, nous avons conçu 16 caractéristiques supplémentaires axées sur la **Sécurité du Roi** et les **Règles du Jeu**. Cette "injection de caractéristiques" accélère considérablement le processus d'apprentissage en fournissant au réseau des indicateurs stratégiques de haut niveau qui mettraient autrement des millions d'itérations à être découverts.

### Configuration Modulaire
L'utilisation de fichiers `.conf` permet d'expérimenter rapidement différentes architectures (nombre de couches, unités par couche, fonctions d'activation) sans modifier le code source. Cette séparation des préoccupations est essentielle pour des projets de machine learning de niveau professionnel.

### Implémentation Personnalisée vs Bibliothèques
L'implémentation du réseau à partir de zéro (en utilisant uniquement NumPy) démontre une compréhension approfondie de la backpropagation, de la descente de gradient et de la stabilité numérique — des exigences clés pour la défense technique de ce projet.

## Utilisation

### Entraînement
Pour entraîner le modèle avec une configuration et un dataset spécifiques :
```bash
./my_torch_analyzer --mode train --config basic_network.conf --dataset 10_pieces.txt --save mon_modele.nn
```

### Prédiction
Pour utiliser un modèle entraîné pour la prédiction :
```bash
./my_torch_analyzer --mode predict --model mon_modele.nn --dataset test_positions.txt
```

## Performance & Benchmarks
Le modèle a été testé sur différents types de datasets pour évaluer sa précision (Accuracy) et sa perte (Loss). Voici les résultats obtenus avec le modèle `optimal_network.nn` et `my_torch_network.nn`.

### Résultats par Catégorie (optimal_network.nn)
Ces tests évaluent la capacité du modèle à identifier des situations spécifiques selon la complexité de l'échiquier (nombre de pièces).

| Dataset | Type | Précision (Accuracy) | Perte (Loss) |
| :--- | :--- | :--- | :--- |
| **Checkmate** | 10 pièces | 84.85% | 0.4732 |
| **Checkmate** | 20 pièces | 77.06% | 0.7284 |
| **Checkmate** | Nombreuses pièces | 67.58% | 1.1454 |
| **Check** | 10 pièces | 58.35% | 0.9997 |
| **Check** | 20 pièces | 61.09% | 0.9868 |
| **Check** | Nombreuses pièces | 64.69% | 0.9386 |
| **Nothing** | 10 pièces | 71.62% | 0.6842 |
| **Nothing** | 20 pièces | 64.27% | 0.9967 |
| **Nothing** | Nombreuses pièces | 79.71% | 0.5925 |

### Résultats sur Datasets Mixtes (my_torch_network.nn)
Tests effectués sur des datasets équilibrés et mélangés de différentes tailles.

| Dataset | Taille / Type | Précision (Accuracy) | Perte (Loss) |
| :--- | :--- | :--- | :--- |
| **Balanced Mixed** | 10 000 exemples | 70.28% | 0.8251 |
| **Balanced Mixed** | 30 000 exemples | 70.63% | 0.8140 |
| **Balanced Mixed** | 50 000 exemples | 94.50% | 0.1870 |
| **Balanced Test** | Test | 70.17% | 0.7443 |
| **Balanced Train** | Entraînement | 71.50% | 0.7171 |
| **Result** | Final | 100.00% | 0.0459 |

### Analyse des Résultats
- **Complexité** : On observe que la précision a tendance à diminuer légèrement lorsque le nombre de pièces augmente dans les situations d'échec et mat, ce qui est cohérent avec la complexité croissante des positions.
- **Volume de données** : Le passage de 30 000 à 50 000 exemples montre un saut de performance significatif (de 70% à 94%), soulignant l'importance de la taille du dataset pour la généralisation du modèle.
- **Optimisation** : L'inclusion des caractéristiques avancées de sécurité du roi permet de maintenir des performances solides même sur des positions complexes.
.
