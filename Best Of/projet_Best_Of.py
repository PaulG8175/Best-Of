import numpy as np
import matplotlib.pyplot as plt

# Box-Muller en dimension 3 : génère 3 gaussiennes indépendantes (Z1,Z2 partagent le même U)
def bx_mu_R3(n):
    U,V,X,Y = np.random.rand(n), np.random.rand(n),np.random.rand(n),np.random.rand(n)
    Z1 = np.sqrt(-2*np.log(U))*np.cos(2*np.pi*V)
    Z2 = np.sqrt(-2*np.log(U))*np.sin(2*np.pi*V)
    Z3 = np.sqrt(-2*np.log(X))*np.cos(2*np.pi*Y)
    return np.column_stack((Z1,Z2,Z3))

# Box-Muller 1D
def bx_mu(n):
    U,V = np.random.rand(n),np.random.rand(n)
    return np.sqrt(-2*np.log(U))*np.cos(2*np.pi*V)

# CDF de la loi normale centrée réduite (Abramowitz & Stegun)
def Abramowitz(x):
    if x<0:
        return 1-Abramowitz(-x)
    else:
        b0 = 0.2316419
        b1 = 0.319381530
        b2 = -0.356563782
        b3 = 1.781477937
        b4 = -1.821255978
        b5 = 1.330274429
        t = 1/(1+b0*x)
        return 1 - np.exp(-x**2 /2) * (b1*t+b2* t**2 +b3* t**3 +b4* t**4 +b5 * t**5)/np.sqrt(2*np.pi)

# Variance empirique sans biais
def var_empi(X):
    return np.sum((X-np.mean(X))**2)/(np.size(X)-1)


# Q1 : S_i(t) = S0_i * exp((r-sig_i**2 /2)*t+ sig_i*Wt)

# Q2 : on a exp(-r*T)*S_i(T) = S0_i * exp(-0.5*sig_i**2 + sig_i*Wt), donc en prenant l'espérance, on a : 
# E[exp(-r*T)*S_i(T)] = S0_i * exp(-0.5*T*sig_i**2) * exp(sig_i**2 * T/2) = S0_i, on a donc directement l'inégalité
# on a égalité lorsque tout les sous-jacents à maturité sont égaux p.s à maturité, c'est à dire lorsque sig1=sig2=sig3 et rho=1

# ============================================================
# Q3) Forward Best Of par MC standard
# P1 = e^{-rT} * E[max(S1(T), S2(T), S3(T)) - K]
# ============================================================

sig = 0.3
S0 = np.array([1,1,1])
rho=0.3
K=1
r=0.02
T=1.5
q_95=1.645

# Matrice de corrélation équicorrélée, décomposée par Cholesky
gamma = np.array([[1,rho,rho],[rho,1,rho],[rho,rho,1]])
L = np.linalg.cholesky(gamma)

# Simulation de W(T) corrélé : W = sqrt(T) * eps @ L^T avec eps ~ N(0,I3)
def mbs(T,L,eps):
    return np.sqrt(T)*(eps@np.transpose(L))

# S(T) = S0 * exp((r - 0.5*sig²)*T + sig*W(T))
def ST(S0,sig,r,W,t):
    return S0*np.exp((r-0.5* sig**2)*t + sig*W)

# Estimateur MC standard de P1 : payoff = max(Si(T)) - K
def P1_MC(T,L,S0,sig,eps):
    W = mbs(T,L,eps)
    S = ST(S0,sig,r,W,T)

    P1_list = np.exp(-r*T)*(np.max(S,axis=1)-K)
    P1 = np.mean(P1_list)

    var = var_empi(P1_list)
    IC = np.array([P1-q_95*np.sqrt(var/np.size(P1_list)),P1+q_95*np.sqrt(var/np.size(P1_list))])

    return P1,IC


# ============================================================
# Q4) Réduction de variance par variables antithétiques
# On moyenne les payoffs de eps et -eps
# ============================================================

def P1_MC_anti(T,L,S0,sig,eps):
    W_plus = mbs(T,L,eps)
    W_moins = mbs(T,L,-eps)
    S_plus = ST(S0,sig,r,W_plus,T)
    S_moins = ST(S0,sig,r,W_moins,T)

    # Estimateur antithétique : moyenne des deux max
    P1_list = np.exp(-r*T)*(0.5*(np.max(S_plus,axis=1)+np.max(S_moins,axis=1))-K)
    P1 = np.mean(P1_list)

    var = var_empi(P1_list)
    IC = np.array([P1-q_95*np.sqrt(var/np.size(P1_list)),P1+q_95*np.sqrt(var/np.size(P1_list))])

    return P1,IC
    
# Q5 : la fonction variance empirique est au début et les intervalles de confiance dans les fonctions Q3 et Q4

N_list = np.array([1000,3000,5000,10000,30000,50000,100000,300000,500000,1000000])
N_taille=np.size(N_list)

P1_MC_list = np.zeros(N_taille)
P1_MC_anti_list = np.zeros(N_taille)

IC_list = np.zeros((N_taille,2))
IC_anti_list = np.zeros((N_taille,2))

for i,N in enumerate(N_list):
    eps = bx_mu_R3(N)
    P1_MC_list[i],IC_list[i,:] = P1_MC(T,L,S0,sig,eps)
    P1_MC_anti_list[i],IC_anti_list[i,:] = P1_MC_anti(T,L,S0,sig,eps)

plt.figure()
plt.title("Estimateurs de P1 en fonction de N")
plt.plot(N_list,P1_MC_list,label="Monte Carlo")
plt.plot(N_list,P1_MC_anti_list,label="Monte Carlo anti")
plt.fill_between(N_list,IC_list[:,0],IC_list[:,1],alpha=0.3,label="IC 90%")
plt.fill_between(N_list,IC_anti_list[:,0],IC_anti_list[:,1],alpha=0.3,label="IC anti 90%")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("prix option")
plt.show()


# ============================================================
# Q6) P1 en fonction de rho : quand rho→1, les sous-jacents convergent
# vers le même actif → max(Si(T)) → S(T) → P1 → S0 - e^{-rT}*K → 0 (ATM)
# ============================================================

# N=1e5 : critère de précision validé sur le graphique Q5
N=100000
S0=2
nbr_rho = 40
rho_list = np.linspace(-0.49,0.99,nbr_rho)

P1_anti_rho = np.zeros(nbr_rho)

for i,rho in enumerate(rho_list):
    gamma = np.array([[1,rho,rho],[rho,1,rho],[rho,rho,1]])
    L = np.linalg.cholesky(gamma)
    eps = bx_mu_R3(N)
    P1_anti_rho[i] = P1_MC_anti(T,L,S0,sig,eps)[0]

plt.figure()
plt.title("Prix de l'option en fonction de rho")
plt.plot(rho_list,P1_anti_rho)
plt.ylabel("prix option")
plt.xlabel("rho")
plt.show()



# on voit que le prix de l'option diminue fortement vers 0 lorsque rho tend vers 1, ce qui est normal
# en effet lorsque rho tend vers 1, les sous jacents sont de plus en plus corrélés entre eux et on a vue Q2, que cela,
# en résulte à une égalité. Or, K=1 et S0=1, on a donc P1 qui tend vers 0.


# ============================================================
# Q7) Put européen vanille sur Si : formule analytique B&S
# PE,i = -S0*Φ(-d1) + K*e^{-rT}*Φ(-d2)
# ============================================================

# Q7 : il s'agait de la formule classique d'un put sous le modèle de Black & Sholes pour un seul sous-jacent
# on PE,i = -S0*phi(-d1) + K*exp(-r*T)*phi(-d2) avec phi fonctiond e répartition de N(0,1) et 
# d1 = (ln(S0/K)+(r+0.5*sig**2)*T)/(sig*sqrt(T)) et d2 = d1 - sig*sqrt(T)

def PE_i(T,r,sig,K,S0):
    d1 = (np.log(S0/K)+(r+ 0.5 *sig**2)*T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return -S0*Abramowitz(-d1) + K*np.exp(-r*T)*Abramowitz(-d2)

# Q8 : on a (K - max(Si(T))+ <= (K-Si(T))+ , donc : P2 <= max(PE_i)

# ============================================================
# Q9) Put Best Of par MC antithétique
# P2 = e^{-rT} * E[(K - max(Si(T)))+]
# rho=1 : tous les sous-jacents identiques → P2 = put vanille classique
# ============================================================

def P2_MC_anti(T,L,S0,sig,eps):
    W_plus = mbs(T,L,eps)
    W_moins = mbs(T,L,-eps)
    S_plus = ST(S0,sig,r,W_plus,T)
    S_moins = ST(S0,sig,r,W_moins,T)

    # Payoff : (K - max(Si(T)))+ moyenné sur les deux trajectoires antithétiques
    P2_list = np.exp(-r*T) * (np.maximum(K-np.max(S_plus,axis=1),0)+np.maximum(K-np.max(S_moins,axis=1),0))/2
    P2 = np.mean(P2_list)

    var = var_empi(P2_list)
    IC = np.array([P2-q_95*np.sqrt(var/np.size(P2_list)),P2+q_95*np.sqrt(var/np.size(P2_list))])

    return P2,IC



N_list = np.array([1000,3000,5000,10000,30000,50000,100000,300000,500000,1000000])
N_taille=np.size(N_list)

rho1 = 0.3
rho2 = 0.9999    # pas 1 car matrice de covariance non inversible sinon mais avec rho=1, on a P2 qui est égale au prix d'un put classique avec un seul sous jacent

gamma1 = np.array([[1,rho1,rho1],[rho1,1,rho1],[rho1,rho1,1]])
L1 = np.linalg.cholesky(gamma1)

gamma2 = np.array([[1,rho2,rho2],[rho2,1,rho2],[rho2,rho2,1]])
L2 = np.linalg.cholesky(gamma2)

P2_MC_1_anti_list = np.zeros(N_taille)
IC_anti_1_list = np.zeros((N_taille,2))

P2_MC_2_anti_list = np.zeros(N_taille)
IC_anti_2_list = np.zeros((N_taille,2))

for i,N in enumerate(N_list):
    eps = bx_mu_R3(N)
    P2_MC_1_anti_list[i],IC_anti_1_list[i,:] = P2_MC_anti(T,L1,S0,sig,eps)
    P2_MC_2_anti_list[i],IC_anti_2_list[i,:] = P2_MC_anti(T,L2,S0,sig,eps)

# Longueur normalisée de l'IC pour vérifier la précision
print(np.abs(IC_anti_1_list[-2,0] - IC_anti_1_list[-2,1])/P2_MC_1_anti_list[-2])


plt.figure()
plt.title("Estimateurs de P2 en fonction de N")
plt.plot(N_list,P2_MC_1_anti_list,label="Monte Carlo anti rho=0.3")
plt.fill_between(N_list,IC_anti_1_list[:,0],IC_anti_1_list[:,1],alpha=0.3,label="IC anti 90%")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("prix option")
plt.show()


plt.figure()
plt.title("Estimateurs de P2 en fonction de N")
plt.plot(N_list,P2_MC_2_anti_list,label="Monte Carlo anti rho=1")
plt.fill_between(N_list,IC_anti_2_list[:,0],IC_anti_2_list[:,1],alpha=0.3,label="IC anti 90%")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("prix option")
plt.show()


# ============================================================
# Q10) P2 en fonction de rho pour plusieurs K
# P2 croît avec rho : corrélation élevée → max proche d'un seul actif → plus de chance d'être ITM
# P2 croît avec K : strike plus élevé → option plus dans la monnaie
# ============================================================

# N=500.000 : longueur d'IC normalisée ≈ 0.007 → précision suffisante
N=500000


nbr_rho = 50
rho_list = np.linspace(-0.499,0.999,50)
K_list = np.array([1,1.2,1.5])
plt.figure()
plt.title("prix P2 en fonction de rho")

for K in K_list:

    P2_MC_anti_rho = np.zeros(nbr_rho)

    for i,rho in enumerate(rho_list):
        eps = bx_mu_R3(N)
        gamma_rho = np.array([[1,rho,rho],[rho,1,rho],[rho,rho,1]])
        L_rho = np.linalg.cholesky(gamma_rho)
        P2_MC_anti_rho[i] = P2_MC_anti(T,L_rho,S0,sig,eps)[0]
    plt.plot(rho_list,P2_MC_anti_rho,label=f"P2(rho) pour K={K}")

plt.xlabel("rho")
plt.ylabel("prix option P2")
plt.legend()
plt.show()


# ============================================================
# Q11) P2 vs PE,i en fonction de S1,0
# PE,i >= P2 car (K - max(Si))+ <= (K - S1)+ (Q8)
# P2 → 0 quand S1,0 → 2 (le max est forcément élevé → put OTM)
# ============================================================

nbr_S0_1 = 50

S0_1_list = np.linspace(0.001,2,nbr_S0_1)
K=1
rho=0.3
P2_MC_anti_S0_list = np.zeros(nbr_S0_1)
PE_list = np.zeros(nbr_S0_1)

gamma = np.array([[1,rho,rho],[rho,1,rho],[rho,rho,1]])
L = np.linalg.cholesky(gamma)


for i,S0_1 in enumerate(S0_1_list):
    S0 = np.array([S0_1,1,1])
    eps = bx_mu_R3(N)
    PE_list[i] = PE_i(T,r,sig,K,S0_1)
    P2_MC_anti_S0_list[i] = P2_MC_anti(T,L,S0,sig,eps)[0]

plt.figure()
plt.title("Prix option e, fonction de S0")
plt.plot(S0_1_list,PE_list,label="prix PE_i")
plt.plot(S0_1_list,P2_MC_anti_S0_list,label="prix P2")
plt.legend()
plt.xlabel("S0")
plt.ylabel("prix option")
plt.show()

# on voit bien que PE_i>= P2, De plus P2 est très proche de 0 lorsque S0_1 tend vers 2.


# ============================================================
# Q12) Variable de contrôle : put vanille sur S1
# Z = X - opt*(Y - E[Y]) avec opt = Cov(X,Y)/Var(Y)
# réduit la variance si X et Y sont corrélés (le put sur S1 est proche du put Best Of)
# ============================================================

S0=np.array([1,1,1])

def P2_controle(T,L,S0,sig,eps):
    W = mbs(T,L,eps)
    S = ST(S0,sig,r,W,T)

    X = np.exp(-r*T)*np.maximum(K-np.max(S,axis=1),0)  # payoff put Best Of
    Y = np.exp(-r*T)*np.maximum(K-S[:,0],0)             # variable de contrôle : put sur S1

    E_Y = PE_i(T,r,sig,K,S0[0])  # espérance analytique connue de Y
    opt = np.cov(X,Y,ddof=1)[0,1]/var_empi(Y)  # coefficient optimal
    Z = X-opt*(Y-E_Y)  # estimateur corrigé
    print(np.corrcoef(X,Y)[0,1])  # corrélation entre X et Y (doit être élevée)
    P2 = np.mean(Z)
    var = var_empi(Z)
    IC = np.array([P2 - q_95*np.sqrt(var/np.size(Z)),P2+q_95*np.sqrt(var/np.size(Z))])
    return P2,IC



N_list = np.array([1000,3000,5000,10000,30000,50000,100000,300000,500000,1000000])
N_taille=np.size(N_list)

P2_controle_list = np.zeros(N_taille)
P2_no_controle_list = np.zeros(N_taille)

IC_controle_list = np.zeros((N_taille,2))
IC_no_controle_list = np.zeros((N_taille,2))

for i,N in enumerate(N_list):
    eps = bx_mu_R3(N)
    P2_controle_list[i],IC_controle_list[i,:] = P2_controle(T,L,S0,sig,eps)
    P2_no_controle_list[i],IC_no_controle_list[i,:] = P2_MC_anti(T,L,S0,sig,eps)

plt.figure()
plt.title("Estimateurs de P2 en fonction de N")
plt.plot(N_list,P2_controle_list,label="controlé")
plt.plot(N_list,P2_no_controle_list,label="non controlé")
plt.fill_between(N_list,IC_controle_list[:,0],IC_controle_list[:,1],alpha=0.3,label="IC controlé 90%")
plt.fill_between(N_list,IC_no_controle_list[:,0],IC_no_controle_list[:,1],alpha=0.3,label="IC non controlé 90%")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("prix option")
plt.show()


# ============================================================
# Q13) Put Best Of sur N sous-jacents en fonction de N et rho
# rho=0 : sous-jacents indépendants → max très élevé → put OTM → P2 faible
# rho=1 : tous identiques → P2 = put vanille classique
# ============================================================

nbr_simu = 100000

# Matrice de corrélation équicorrélée de taille N : (1-rho)*I + rho*11^T
def L_N(rho,N):
    return np.linalg.cholesky((1-rho)*np.eye(N) + np.full((N,N),rho))

# Génère N gaussiennes indépendantes via Box-Muller
def eps_liste(N,nbr_simu):
    eps = np.zeros((nbr_simu,N))
    for i in range(N):
        eps[:,i] = bx_mu(nbr_simu)
    return eps

S0=1
sig=0.3
K=1

N_list = np.array([1,3,10,25])
rho_list = np.array([0,0.5,0.99])

plt.figure()
plt.title("Estimateurs de P2_N en fonction de N")
for rho in rho_list:
    P2_N_list = np.zeros(np.size(N_list))
    for i,N in enumerate(N_list):
        L = L_N(rho,N)
        eps = eps_liste(N,nbr_simu)
        P2_N_list[i] = P2_MC_anti(T,L,S0,sig,eps)[0]
    plt.plot(N_list,P2_N_list,label=f"P2_N pour rho={rho}")
plt.legend()
plt.xlabel("N")
plt.ylabel("Prix de P2_N")
plt.show()

# Q14 : on pour x, x = (x-K)+ - (K-x)+ + K, donc max(Si(T)) - K = (max(Si(T))-K)+ - (K-max(Si(T)))+, en prenant l'espérance
# et en multipliant par exp(-r*T), on a donc : Forward = Call - Put
