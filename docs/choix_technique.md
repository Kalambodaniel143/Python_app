Voici le **format JSON complet, propre et professionnel**, qu'on a jugeradapté pour sauvegarder notre réseau neuronal dans le cadre du projet MY_TORCH.
Dans la suite nous expliquons **le rôle de chaque champ** pour une comprehension plus globale

---

# **1. Format JSON pour sauvegarder un réseau**

Voici un exemple du fichier `my_torch_network_basic.nn` :

```json
{
  "version": 1,

  "architecture": {
    "input_size": 768,
    "layers": [
      {
        "type": "dense",
        "units": 128,
        "activation": "relu"
      },
      {
        "type": "dense",
        "units": 64,
        "activation": "relu"
      },
      {
        "type": "dense",
        "units": 3,
        "activation": "softmax"
      }
    ]
  },

  "weights": [
    {
      "W": [[0.12, -0.08, ...], [...]],
      "b": [0.01, 0.02, ...]
    },
    {
      "W": [[0.23, -0.11, ...], [...]],
      "b": [0.02, 0.01, ...]
    },
    {
      "W": [[0.05, -0.04, ...], [...]],
      "b": [0.00, 0.00, ...]
    }
  ],

  "training_metadata": {
    "epochs_trained": 15,
    "learning_rate": 0.001,
    "loss_function": "cross_entropy",
    "optimizer": "sgd",
    "batch_size": 32
  },

  "info": {
    "created_at": "2025-02-12T10:32:00",
    "model_name": "my_torch_network_basic"
  }
}
```
---

# **2. Explication détaillée de chaque champ**

---

## **A. `version`**

```
"version": 1
```
### Rôle :

* Permet de gérer différentes versions du format `.nn`.
* Utile si un jour tu modifies la structure interne du fichier.

---

## **B. `architecture`**

C'est le bloc qui décrit **la structure logique du réseau**.
### Exemple :
```json
"architecture": {
  "input_size": 768,
  "layers": [...]
}
```

### Champs :

#### 1) `input_size`
* Nombre d’entrées (ceci est lié à notre la représentation du FEN). 
#### 2) `layers`
Liste de dictionnaires, un par couche.
Chaque couche contient :
* `type`: ex. "dense"
* `units`: nombre de neurones
* `activation`: relu, sigmoid, softmax…
### Importance :
* Permet de reconstruire le réseau sans ambiguïté.
* Le chargeur (`load_model`) va utiliser ces infos pour regénérer chaque couche.
---

## **C. `weights`**
Contient **tous les poids et biais** du réseau, dans l’ordre exact des couches.
### Structure :

```json
"weights": [
  { "W": [...], "b": [...] },
  ...
]
```

### Champs :
#### 1) `W`
* Matrice des poids de la couche.
* Dimensions : `(input_dim, output_dim)`.
#### 2) `b`
* Vecteur de biais.
* Taille : `(output_dim)`.

### Importance :
* C’est le cœur du modèle.
* Sans ça, le réseau serait vide (non entraîné).
* Permet de restaurer parfaitement le réseau pour prédiction ou poursuite d’entraînement.

---
## **D. `training_metadata`**

Informations supplémentaires sur l’entraînement.
### Exemple :

```json
"training_metadata": {
  "epochs_trained": 15,
  "learning_rate": 0.001,
  "loss_function": "cross_entropy",
  "optimizer": "sgd",
  "batch_size": 32
}
```

### Importance :

* Facilite le debugging.
* Permet de reprendre ou analyser un entraînement.
* Utile pour justifier tes choix d’hyperparamètres en soutenance.

---

## **E. `info`**

Bloc informatif (non technique).

### Exemple :

```json
"info": {
  "created_at": "2025-02-12T10:32:00",
  "model_name": "my_torch_network_basic"
}
```