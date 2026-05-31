# 🏆 PRB222 — Option sur Best Of (Maximum de Panier)

Projet numérique portant sur la valorisation d'options sur le **maximum d'un panier d'actifs** (Best Of) dans le modèle de Black-Scholes, par technique de Monte Carlo avec réduction de variance.

---

## 🎯 Objectif

Calculer et analyser trois types de produits sur Best Of :

| Produit | Payoff |
|---------|--------|
| **Forward Best Of** | `max(Si(T)) − K` |
| **Put Best Of** | `(K − max(Si(T)))+` |
| **Call Best Of** | `(max(Si(T)) − K)+` |

Avec les méthodes : MC standard, **variables antithétiques**, et **variable de contrôle**.

---

## 📂 Structure

```
├── best_of_commented.py   # Script principal commenté
```

---

## ⚙️ Modèle

Black-Scholes en dimension 3, matrice de corrélation équicorrélée :

```
Γ = [[1, ρ, ρ],
     [ρ, 1, ρ],
     [ρ, ρ, 1]]    avec ρ ∈ ]-0.5, 1[
```

Simulation via **décomposition de Cholesky** : `W(T) = √T · ε @ L^T`, avec `ε ~ N(0, I3)`.

---

## ⚙️ Paramètres

| Paramètre | Valeur |
|-----------|--------|
| `Si,0` | 1 |
| `σi` | 0.30 |
| `ρ` | 0.3 (sauf Q6, Q10) |
| `K` | 1 (sauf Q10) |
| `r` | 0.02 |
| `T` | 1.5 ans |

> Les simulations utilisent uniquement des lois **Uniformes** via la méthode de Box-Muller.

---

## 📋 Questions traitées

| Q | Contenu |
|---|---------|
| Q1 | Solution de l'EDS Si(t) |
| Q2 | Minoration de P1 et cas d'égalité (σ identiques, ρ=1) |
| Q3 | Simulation de W(T) + MC standard pour P1 |
| Q4 | Variables antithétiques pour P1 |
| Q5 | Variances empiriques et IC à 90% des deux estimateurs |
| Q6 | P1 en fonction de ρ ∈ ]−0.5, 1[ |
| Q7 | Put européen vanille PE,i (formule analytique B&S) |
| Q8 | Majorant de P2 par les puts vanilles |
| Q9 | MC antithétique pour P2, comparaison ρ=0.3 vs ρ=1 |
| Q10 | P2 en fonction de ρ pour K ∈ {1, 1.2, 1.5} |
| Q11 | P2 vs PE,i en fonction de S1,0 ∈ [0, 2] |
| Q12 | Variable de contrôle (put sur S1) pour P2 |
| Q13 | P2,N en fonction de N ∈ {1,3,10,25} et ρ ∈ {0, 0.5, 1} |
| Q14 | Relation Forward = Call − Put sur Best Of |

---

## 🔑 Idées clés

- **ρ → 1** : les sous-jacents deviennent identiques → Forward → 0 (ATM), Put → put vanille
- **N grand, ρ=0** : le max d'actifs indépendants est très élevé → put Best Of quasi nul
- **Variable de contrôle** : le put sur S1 est corrélé au put Best Of → réduction de variance efficace

---

## 🚀 Lancement

```bash
pip install numpy matplotlib
python best_of_commented.py
```

---

## 📦 Dépendances

- `numpy`
- `matplotlib`
