import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
#https://docs.scipy.org/doc/scipy/reference/stats.html

# Question 1
dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)

# ===== DISTRIBUTIONS DISCRÈTES =====

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distributions Discrètes', fontsize=16, fontweight='bold')

# 1. Loi de Dirac (spike à une valeur)
ax = axes[0, 0]
x_dirac = np.array([5])
y_dirac = np.array([1])
ax.bar(x_dirac, y_dirac, width=0.1, color='blue')
ax.set_title('Loi de Dirac (δ=5)')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')

# 2. Loi uniforme discrète
ax = axes[0, 1]
x_unif = np.arange(1, 7)
y_unif = stats.randint.pmf(x_unif, 1, 7)
ax.bar(x_unif, y_unif, color='green')
ax.set_title('Loi Uniforme Discrète (1-6)')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')

# 3. Loi binomiale
ax = axes[0, 2]
n, p = 20, 0.5
x_binom = np.arange(0, n+1)
y_binom = stats.binom.pmf(x_binom, n, p)
ax.bar(x_binom, y_binom, color='red')
ax.set_title(f'Loi Binomiale (n={n}, p={p})')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')

# 4. Loi de Poisson
ax = axes[1, 0]
lambda_poisson = 5
x_poisson = np.arange(0, 15)
y_poisson = stats.poisson.pmf(x_poisson, lambda_poisson)
ax.bar(x_poisson, y_poisson, color='purple')
ax.set_title(f'Loi de Poisson (λ={lambda_poisson})')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')

# 5. Loi de Zipf-Mandelbrot
ax = axes[1, 1]
a = 1.5
x_zipf = np.arange(1, 21)
y_zipf = stats.zipf.pmf(x_zipf, a)
ax.bar(x_zipf, y_zipf, color='orange')
ax.set_title(f'Loi de Zipf-Mandelbrot (a={a})')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')

# Laisser le 6ème sous-graphique vide ou ajouter un résumé
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# ===== DISTRIBUTIONS CONTINUES =====

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distributions Continues', fontsize=16, fontweight='bold')

x_cont = np.linspace(-5, 15, 500)

# 1. Loi normale
ax = axes[0, 0]
mu, sigma = 5, 2
y_norm = stats.norm.pdf(x_cont, mu, sigma)
ax.plot(x_cont, y_norm, 'b-', linewidth=2)
ax.fill_between(x_cont, y_norm, alpha=0.3)
ax.set_title(f'Loi Normale (μ={mu}, σ={sigma})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# 2. Loi log-normale
ax = axes[0, 1]
s = 0.5  # shape parameter
y_lognorm = stats.lognorm.pdf(x_cont, s)
ax.plot(x_cont, y_lognorm, 'g-', linewidth=2)
ax.fill_between(x_cont, y_lognorm, alpha=0.3)
ax.set_title(f'Loi Log-Normale (s={s})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# 3. Loi uniforme continue
ax = axes[0, 2]
a_unif, b_unif = 0, 10
x_unif_cont = np.linspace(a_unif-2, b_unif+2, 500)
y_unif_cont = stats.uniform.pdf(x_unif_cont, a_unif, b_unif-a_unif)
ax.plot(x_unif_cont, y_unif_cont, 'r-', linewidth=2)
ax.fill_between(x_unif_cont, y_unif_cont, alpha=0.3)
ax.set_title(f'Loi Uniforme Continue (a={a_unif}, b={b_unif})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# 4. Loi du Chi-deux (χ²)
ax = axes[1, 0]
df = 5  # degrés de liberté
x_chi2 = np.linspace(0, 20, 500)
y_chi2 = stats.chi2.pdf(x_chi2, df)
ax.plot(x_chi2, y_chi2, 'purple', linewidth=2)
ax.fill_between(x_chi2, y_chi2, alpha=0.3)
ax.set_title(f'Loi du Chi-deux (df={df})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# 5. Loi de Pareto
ax = axes[1, 1]
b = 2.5  # shape parameter
x_pareto = np.linspace(1, 10, 500)
y_pareto = stats.pareto.pdf(x_pareto, b)
ax.plot(x_pareto, y_pareto, 'orange', linewidth=2)
ax.fill_between(x_pareto, y_pareto, alpha=0.3)
ax.set_title(f'Loi de Pareto (b={b})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# Note: Poisson est discrète, donc on la place pas ici (elle est déjà en discret)
axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, 'Note: Poisson est une loi discrète', 
                ha='center', va='center', fontsize=12, transform=axes[1, 2].transAxes)

plt.tight_layout()
plt.show()

# Question 2#
# ===== FONCTIONS POUR CALCULER MOYENNE ET ÉCART TYPE =====

def stats_distribution(name, dist, params):
    """
    Calcule la moyenne et l'écart type d'une distribution
    
    Parameters:
    - name: nom de la distribution (str)
    - dist: objet scipy.stats de la distribution
    - params: dictionnaire des paramètres de la distribution
    
    Returns:
    - tuple (moyenne, écart_type)
    """
    mean = dist.mean(**params)
    std = dist.std(**params)
    return mean, std

# ===== DISTRIBUTIONS DISCRÈTES =====

print("=" * 60)
print("STATISTIQUES DES DISTRIBUTIONS DISCRÈTES")
print("=" * 60)

# 1. Loi de Dirac
dirac_mean, dirac_std = 5, 0
print(f"Loi de Dirac (δ=5):")
print(f"  Moyenne: {dirac_mean}")
print(f"  Écart-type: {dirac_std}\n")

# 2. Loi uniforme discrète
params_unif_disc = {'low': 1, 'high': 7}
unif_disc_mean, unif_disc_std = stats_distribution('Uniforme discrète', stats.randint, params_unif_disc)
print(f"Loi Uniforme Discrète (1-6):")
print(f"  Moyenne: {unif_disc_mean:.4f}")
print(f"  Écart-type: {unif_disc_std:.4f}\n")

# 3. Loi binomiale
params_binom = {'n': 20, 'p': 0.5}
binom_mean, binom_std = stats_distribution('Binomiale', stats.binom, params_binom)
print(f"Loi Binomiale (n=20, p=0.5):")
print(f"  Moyenne: {binom_mean:.4f}")
print(f"  Écart-type: {binom_std:.4f}\n")

# 4. Loi de Poisson
params_poisson = {'mu': 5}
poisson_mean, poisson_std = stats_distribution('Poisson', stats.poisson, params_poisson)
print(f"Loi de Poisson (λ=5):")
print(f"  Moyenne: {poisson_mean:.4f}")
print(f"  Écart-type: {poisson_std:.4f}\n")

# 5. Loi de Zipf-Mandelbrot
params_zipf = {'a': 1.5}
zipf_mean, zipf_std = stats_distribution('Zipf', stats.zipf, params_zipf)
print(f"Loi de Zipf-Mandelbrot (a=1.5):")
print(f"  Moyenne: {zipf_mean:.4f}")
print(f"  Écart-type: {zipf_std:.4f}\n")

# ===== DISTRIBUTIONS CONTINUES =====

print("=" * 60)
print("STATISTIQUES DES DISTRIBUTIONS CONTINUES")
print("=" * 60)

# 1. Loi normale
params_norm = {'loc': 5, 'scale': 2}
norm_mean, norm_std = stats_distribution('Normale', stats.norm, params_norm)
print(f"Loi Normale (μ=5, σ=2):")
print(f"  Moyenne: {norm_mean:.4f}")
print(f"  Écart-type: {norm_std:.4f}\n")

# 2. Loi log-normale
params_lognorm = {'s': 0.5}
lognorm_mean, lognorm_std = stats_distribution('Log-Normale', stats.lognorm, params_lognorm)
print(f"Loi Log-Normale (s=0.5):")
print(f"  Moyenne: {lognorm_mean:.4f}")
print(f"  Écart-type: {lognorm_std:.4f}\n")

# 3. Loi uniforme continue
params_uniform = {'loc': 0, 'scale': 10}
uniform_mean, uniform_std = stats_distribution('Uniforme continue', stats.uniform, params_uniform)
print(f"Loi Uniforme Continue (a=0, b=10):")
print(f"  Moyenne: {uniform_mean:.4f}")
print(f"  Écart-type: {uniform_std:.4f}\n")

# 4. Loi du Chi-deux
params_chi2 = {'df': 5}
chi2_mean, chi2_std = stats_distribution('Chi-deux', stats.chi2, params_chi2)
print(f"Loi du Chi-deux (df=5):")
print(f"  Moyenne: {chi2_mean:.4f}")
print(f"  Écart-type: {chi2_std:.4f}\n")

# 5. Loi de Pareto
params_pareto = {'b': 2.5}
pareto_mean, pareto_std = stats_distribution('Pareto', stats.pareto, params_pareto)
print(f"Loi de Pareto (b=2.5):")
print(f"  Moyenne: {pareto_mean:.4f}")
print(f"  Écart-type: {pareto_std:.4f}\n")

# ===== TABLEAU RÉCAPITULATIF =====

print("=" * 60)
print("TABLEAU RÉCAPITULATIF")
print("=" * 60)

# Créer un DataFrame pour un affichage plus lisible
data_stats = {
    'Distribution': [
        'Dirac', 'Uniforme discrète', 'Binomiale', 'Poisson', 'Zipf',
        'Normale', 'Log-Normale', 'Uniforme continue', 'Chi-deux', 'Pareto'
    ],
    'Moyenne': [
        dirac_mean, unif_disc_mean, binom_mean, poisson_mean, zipf_mean,
        norm_mean, lognorm_mean, uniform_mean, chi2_mean, pareto_mean
    ],
    'Écart-type': [
        dirac_std, unif_disc_std, binom_std, poisson_std, zipf_std,
        norm_std, lognorm_std, uniform_std, chi2_std, pareto_std
    ]
}

df_stats = pd.DataFrame(data_stats)
print(df_stats.to_string(index=False))
print("\n")