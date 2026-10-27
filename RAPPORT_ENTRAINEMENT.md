# Rapport sur l'entraînement du modèle DynamicViT-Simple

## 1. Objectif

Ce projet entraîne un Vision Transformer (ViT) pour classifier les images de CIFAR-10 tout en apprenant à réduire le nombre de patchs traités par le modèle. L'entraînement se fait en deux étapes :

1. un **Teacher**, qui apprend à classifier les images sans élagage ;
2. un **Student**, initialisé à partir du Teacher, qui apprend à classifier les images tout en décidant quels patchs conserver.

Le Student utilise trois modules de prédiction d'élagage. La sélection des patchs s'appuie sur le Gumbel-Softmax pendant l'entraînement et sur un classement des scores à l'inférence. Le code est principalement défini dans `model.py` et `train.py`.

## 2. Architecture et données

Les images CIFAR-10 ont une taille de 32 × 32 pixels. Une convolution de taille 4 × 4 et de pas 4 les transforme en une grille de **64 patchs**. Un token de classification (`CLS`) est ajouté à cette séquence. Le Transformer comporte 8 blocs, une dimension d'embedding de 128 et 4 têtes d'attention.

Les trois décisions d'élagage sont effectuées avant les blocs d'indice 2, 4 et 6 (indexation Python, donc avant les 3e, 5e et 7e blocs). Le Student vise un ratio de conservation de `0.7` à chaque étage, relativement au nombre initial de patchs. Les ratios cibles cumulés sont donc :

| Étage | Ratio cible | Patchs visés sur 64 |
|---|---:|---:|
| 1 | 0,7 | 44,8, soit environ 44 à 45 |
| 2 | 0,7² = 0,49 | 31,36, soit environ 31 |
| 3 | 0,7³ = 0,343 | 21,95, soit environ 21 |

À l'entraînement, les décisions sont représentées par des masques et les tenseurs de patchs gardent leur longueur de 64. À l'inférence, les patchs sont effectivement sélectionnés et retirés de la séquence.

Les données sont téléchargées automatiquement dans `data/` si nécessaire. L'entraînement applique un recadrage aléatoire avec padding de 4 pixels, un retournement horizontal aléatoire, puis une normalisation avec les moyennes `(0.4914, 0.4822, 0.4465)` et écarts-types `(0.2470, 0.2435, 0.2616)`. Le jeu de test utilise la même normalisation sans augmentation.

## 3. Entraînement du Teacher

Le Teacher utilise la même architecture ViT, mais sans modules de prédiction ni élagage (`use_pruning=False`). Il est entraîné avec l'entropie croisée entre ses logits et les étiquettes CIFAR-10.

Paramètres principaux :

- 15 époques ;
- batch de 64 images ;
- optimiseur AdamW ;
- taux d'apprentissage `1e-3` et weight decay `0.01`.

Le Teacher est évalué sur le jeu de test à chaque époque. Ses poids sont enregistrés dans `checkpoints/teacher.pth`.

## 4. Entraînement du Student

Le Student est créé avec l'élagage activé et un ratio de conservation de `0.7`. Il reçoit les poids du Teacher pour les couches communes. Le chargement utilise `strict=False`, car les modules de prédiction du Student n'existent pas dans le Teacher.

Le Student est ensuite entraîné pendant 10 époques. Le Teacher est placé en mode évaluation, et ses sorties sont calculées sans gradient. Deux groupes de paramètres sont optimisés avec AdamW (`weight_decay=0.01`) :

- backbone : taux d'apprentissage `1e-4` ;
- modules de prédiction : taux d'apprentissage `1e-3`.

Le taux d'apprentissage du backbone est fixé à zéro pendant les 5 premières époques, puis passe à `1e-4`. Les modules de prédiction sont optimisés dès le début. Le Student est évalué à chaque époque et ses poids sont enregistrés dans `checkpoints/student.pth`.

## 5. Gumbel-Softmax pendant l'entraînement

À chaque étage d'élagage, le module de prédiction attribue à chaque patch deux logits : **supprimer** (classe 0) et **conserver** (classe 1). Ces logits sont produits à partir d'une représentation locale du patch et d'un contexte global calculé comme une moyenne pondérée des patchs encore conservés.

Pendant l'entraînement, le code applique :

```python
F.gumbel_softmax(logits_s, tau=1.0, hard=True, dim=-1)
```

Le Gumbel-Softmax ajoute un bruit de Gumbel aux logits, puis les transforme en choix entre les deux classes. Avec `hard=True`, le choix utilisé dans la propagation avant est discret (one-hot), ce qui donne un masque de conservation binaire. L'option hard utilise une estimation de gradient straight-through : le calcul avant utilise le choix discret, tandis que le gradient est estimé à partir de la relaxation soft du Gumbel-Softmax. Cela permet d'entraîner les modules de prédiction malgré la décision discrète.

Le masque de chaque étage est multiplié par le masque cumulatif précédent. Un patch supprimé à un étage reste donc supprimé aux étages suivants. La température est fixée à `1.0` dans cette implémentation ; elle n'est pas diminuée au fil des époques.

Les décisions cumulatives alimentent également la pénalité de ratio. Le masque final est détaché du graphe de calcul pour le terme de distillation des tokens, mais la pénalité de ratio reste reliée aux décisions et peut entraîner les modules de prédiction.

## 6. Attention masking

Lors de l'entraînement du Student, les patchs ne sont pas physiquement retirés : l'attention est masquée selon la politique cumulative de conservation. Pour un patch supprimé, sa colonne est exclue des destinations possibles pour les autres tokens. La diagonale est toutefois conservée, ce qui autorise chaque token à s'attentionner lui-même.

Plus précisément, pour un token requête `i` et un token clé `j`, le masque utilisé est :

- `1` si `i == j` ;
- sinon, la décision de conservation du token `j`.

Les scores d'attention sont exponentiés après soustraction de leur maximum, multipliés par ce masque, puis normalisés. Le token `CLS` est toujours conservé et peut agréger l'information des patchs encore actifs. Le masquage est mis à jour après chaque étape d'élagage et utilisé dans les blocs suivants.

À l'inférence, le code sélectionne effectivement les patchs : il calcule la probabilité de conservation par softmax, garde les meilleurs scores (`top-k`) et réduit la séquence avant de poursuivre le Transformer. Le nombre retenu à l'étage `s` est calculé à partir des 64 patchs initiaux et du ratio `0.7^(s+1)`, avec au moins un patch conservé. Cette sélection physique rend les blocs ultérieurs moins coûteux en calcul.

## 7. Fonctions de perte du Student

La perte totale est une somme pondérée de quatre termes :

```text
L = L_cls + 0.5 L_KL + 0.5 L_distill + 2.0 L_ratio
```

### 7.1 Classification — entropie croisée

```text
L_cls = CrossEntropy(logits_student, labels)
```

Ce terme entraîne le Student à prédire la classe CIFAR-10 correcte.

### 7.2 Distillation des prédictions — divergence KL

```text
L_KL = KL(
    softmax(logits_teacher),
    log_softmax(logits_student)
)
```

Le Student apprend à rapprocher sa distribution de classes de celle du Teacher. Le code utilise `F.kl_div` avec `reduction="batchmean"` et sans température de distillation. Le Teacher est évalué sans gradient. Le poids de ce terme dans la perte totale est `0.5`.

### 7.3 Distillation des représentations de patchs

Le Student et le Teacher produisent leurs représentations finales de patchs. Le code calcule l'erreur quadratique moyenne par patch et par dimension, puis la moyenne uniquement sur les patchs conservés au dernier étage :

```text
L_distill = somme(erreur_patch × masque_final)
            / max(somme(masque_final), 1)
```

La représentation du Teacher et le masque final sont détachés pour ce calcul. Cette perte encourage le Student à préserver, pour les patchs conservés, des représentations proches de celles du Teacher. Son poids est `0.5`.

### 7.4 Régularisation des ratios de conservation

Pour chaque étage, le taux de conservation prédit est comparé à sa cible `0.7^(s+1)`. La perte est la moyenne des erreurs quadratiques entre taux observés par image et ratios cibles :

```text
L_ratio = moyenne((ratio_cible_étage - taux_conservation_image_étage)²)
```

Ce terme encourage l'élagage à respecter les ratios voulus. Son poids est `2.0`, supérieur à celui des deux termes de distillation.

## 8. Évaluation et résultats enregistrés

L'évaluation met le modèle en mode `eval()`. Pour le Student, cela active la sélection top-k et l'élagage physique. L'exactitude de classification est calculée sur le jeu de test.

Pendant l'entraînement du Student, les logs affichent la perte totale et ses quatre composantes, l'exactitude, les taux de conservation moyens par étage et le temps d'époque. À la fin, les exactitudes du Teacher et du Student sont affichées.

Les poids sont sauvegardés comme `state_dict` :

- `checkpoints/teacher.pth` : Teacher entraîné ;
- `checkpoints/student.pth` : Student entraîné.

Ces fichiers contiennent les paramètres du modèle, pas les métriques ni l'historique complet des époques.

## 9. Lancer l'entraînement

Depuis le répertoire du projet :

```bash
python train.py
```

Le script choisit CUDA si disponible, sinon le CPU. Il fixe les graines aléatoires PyTorch et Python à `42`. Les durées d'entraînement peuvent varier selon le matériel.
