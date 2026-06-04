# Best-Of Option Pricing via Monte Carlo

> **Personal project**, done independently outside coursework.

Pricing of **Best-Of options** (options on the maximum of a basket of assets) in a 3-dimensional Black-Scholes model, with variance reduction techniques and control variates.

---

## Products

### Forward Best-Of
$$P^1 = e^{-rT} \mathbb{E}\left[\max_{i=1..3} S_i(T) - K\right]$$

### Put Best-Of
$$P^2 = e^{-rT} \mathbb{E}\left[\left(K - \max_{i=1..3} S_i(T)\right)^+\right]$$

### Call Best-Of
$$P^3 = e^{-rT} \mathbb{E}\left[\left(\max_{i=1..3} S_i(T) - K\right)^+\right]$$

**Put-Call-Forward parity:**
$$P^1 = P^3 - P^2$$

---

## Model

3-dimensional Black-Scholes with equicorrelated Brownian motions:

$$dS_i(t) = S_i(t)\left(r\*dt + \sigma_i\*dW_i(t)\right)$$

$$
\Gamma=
\begin{bmatrix}
1 & \rho & \rho \\
\rho & 1 & \rho \\
\rho & \rho & 1
\end{bmatrix},
\qquad
\rho\in\left(-\frac12,1\right)
$$

Correlated Brownian motions simulated via **Cholesky decomposition**: $W(T) = \sqrt{T}\,\varepsilon\* L^\top$ with $\varepsilon \sim \mathcal{N}(0, I_3)$.

---

## Parameters

| Parameter | Value |
|-----------|-------|
| `Sᵢ,₀` | 1 |
| `σᵢ` | 0.30 |
| `ρ` | 0.3 |
| `K` | 1 |
| `r` | 0.02 |
| `T` | 1.5 years |

> All simulations use **Uniform random variables only** via Box-Muller transform.

---

## Variance reduction techniques

### Antithetic variables
For each draw $\varepsilon$, also evaluate at $-\varepsilon$:
$$\hat{P} = \frac{1}{2}\left(f(\varepsilon) + f(-\varepsilon)\right)$$

### Control variate (Q12)
Use the vanilla European put on $S_1$ as a control variate (known analytically):
$$Z = X - \hat{\beta}(Y - \mathbb{E}[Y]), \quad \hat{\beta} = \frac{\text{Cov}(X,Y)}{\text{Var}(Y)}$$

---

## Key results

**Effect of ρ on P²:**

- As $\rho \to 1$: all assets become identical → $P^2$ converges to the vanilla put price
- As $\rho \to -1/2$: assets are near-independent → the maximum is large → $P^2 \approx 0$

**Effect of N (number of underlyings) on P^{2,N}:**

- $\rho = 0$: independent assets → maximum grows with $N$ → $P^{2,N} \to 0$
- $\rho = 1$: all assets identical regardless of $N$ → $P^{2,N}$ constant

**Upper bound:**
$$P^2 \leq \min_i P^{E,i}$$
since $(K - \max S_i)^+ \leq (K - S_i)^+$ for all $i$.

---

## Topics covered

| Q | Content |
|---|---------|
| Q3 | MC standard estimator for P¹ |
| Q4 | Antithetic variables for P¹ |
| Q5 | Empirical variances and 90% confidence intervals |
| Q6 | P¹ as a function of ρ |
| Q7 | Analytical European put PE,i (Black-Scholes) |
| Q8 | Upper bound for P² |
| Q9 | MC antithetic for P², comparison ρ=0.3 vs ρ=1 |
| Q10 | P² vs ρ for K ∈ {1, 1.2, 1.5} |
| Q11 | P² vs PE,i as a function of S₁,₀ |
| Q12 | Control variate estimator for P² |
| Q13 | P²,N vs N ∈ {1,3,10,25} for ρ ∈ {0, 0.5, 1} |
| Q14 | Forward = Call − Put parity |

---

## Run

```bash
pip install numpy matplotlib
python best_of_commented.py
```

## Dependencies

`numpy` · `matplotlib`

