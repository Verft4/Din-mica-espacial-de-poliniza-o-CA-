"""Modelo em duas escalas: a célula é uma MANCHA de recursos (Passo 3 do plano de viabilização).

Escala da paisagem: cada célula (``lado_m`` metros de lado) é uma mancha com qualidade ``q`` (campo
lognormal espacialmente correlacionado). A colmeia fica no centro. Cada viagem escolhe uma mancha com
probabilidade proporcional a ``q * exp(-(d / lambda)**beta)`` (preferência por distância; ``beta = 1``
exponencial, ``beta = 2`` gaussiana) e a distância da viagem é a da mancha escolhida, com posição
sorteada dentro da célula. NÃO há passeio difusivo: o alcance vem da escolha de mancha.

A dispersão entre sessões (colmeia x dia) **emerge** da heterogeneidade da paisagem: cada sessão usa
uma realização do campo de qualidade. A interação flor a flor (autômato) fica para dentro da mancha e
não é tratada aqui; ``visitas_por_mancha`` é o ponto de ligação com o passo de deposição.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

COMPRIMENTOS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)      # comprimentos de correlação do banco (células)


@dataclass(frozen=True)
class ParametrosManchas:
    lambda_m: float = 800.0       # escala da preferência por distância (m)
    beta: float = 2.0             # forma: 1 = exponencial, 2 = gaussiana
    sigma_q: float = 1.0          # heterogeneidade (desvio do log da qualidade)
    comp_corr: float = 2.0        # comprimento de correlação (células; deve estar em COMPRIMENTOS)

    def __post_init__(self):
        if self.lambda_m <= 0 or self.beta <= 0:
            raise ValueError("lambda_m e beta devem ser > 0")
        if self.sigma_q < 0:
            raise ValueError("sigma_q deve ser >= 0")
        if self.comp_corr not in COMPRIMENTOS:
            raise ValueError(f"comp_corr deve ser um de {COMPRIMENTOS}")


class Paisagem:
    """Grade quadrada de manchas com banco de campos gaussianos suavizados (variância unitária)."""

    def __init__(self, lado_celulas: int = 141, lado_m: float = 100.0, n_campos: int = 80, seed=0):
        if lado_celulas % 2 == 0 or lado_celulas < 5:
            raise ValueError("lado_celulas deve ser ímpar e >= 5 (colmeia no centro)")
        if lado_m <= 0 or n_campos < 1:
            raise ValueError("lado_m e n_campos devem ser positivos")
        self.L, self.lado_m = lado_celulas, float(lado_m)
        rng = np.random.default_rng(seed)
        c = lado_celulas // 2
        ii, jj = np.indices((lado_celulas, lado_celulas))
        self.centro = (c, c)
        self.dx, self.dy = (ii - c).astype(float), (jj - c).astype(float)    # deslocamentos (células)
        self.d_m = np.hypot(self.dx, self.dy) * self.lado_m                  # distância ao centro da mancha (m)
        self.banco: dict[float, np.ndarray] = {}
        for ell in COMPRIMENTOS:
            z = np.stack([gaussian_filter(rng.standard_normal((self.L, self.L)), ell, mode="wrap")
                          for _ in range(n_campos)])
            z /= z.std()                                                     # variância unitária no banco
            self.banco[ell] = z.astype(np.float32)

    def campo(self, rng, comp_corr: float) -> np.ndarray:
        """Realização do campo z: sorteia um campo do banco e o desloca aleatoriamente (a colmeia muda de lugar)."""
        z = self.banco[comp_corr]
        k = int(rng.integers(len(z)))
        return np.roll(z[k], (int(rng.integers(self.L)), int(rng.integers(self.L))), axis=(0, 1)).astype(float)

    def nucleo(self, par: ParametrosManchas) -> np.ndarray:
        return np.exp(-(self.d_m / par.lambda_m) ** par.beta)

    def pesos(self, z: np.ndarray, par: ParametrosManchas, nucleo: np.ndarray | None = None) -> np.ndarray:
        """Peso de escolha de cada mancha: q * núcleo, com q = exp(sigma*z - sigma^2/2) (média 1)."""
        k = self.nucleo(par) if nucleo is None else nucleo
        return np.exp(par.sigma_q * z - par.sigma_q ** 2 / 2.0) * k

    def sortear_viagens(self, rng, pesos: np.ndarray, n: int):
        """``n`` viagens: (distâncias em metros, índices planos das manchas escolhidas)."""
        cum = np.cumsum(pesos.ravel())
        if cum[-1] <= 0:
            raise ValueError("todos os pesos são zero")
        idx = np.minimum(np.searchsorted(cum, rng.random(n) * cum[-1]), cum.size - 1)
        i, j = np.unravel_index(idx, pesos.shape)
        u = rng.random((2, n)) - 0.5                                          # posição dentro da célula
        d = np.hypot(self.dx[i, j] + u[0], self.dy[i, j] + u[1]) * self.lado_m
        return d, idx

    def sessoes(self, rng, par: ParametrosManchas, n_sessoes: int, n_dancas: int):
        """Sessões (colmeia x dia): cada uma usa uma realização da paisagem. Retorna (medianas, médias)."""
        k = self.nucleo(par)
        md, me = np.empty(n_sessoes), np.empty(n_sessoes)
        for s in range(n_sessoes):
            d, _ = self.sortear_viagens(rng, self.pesos(self.campo(rng, par.comp_corr), par, k), n_dancas)
            md[s], me[s] = np.median(d), d.mean()
        return md, me

    def visitas_por_mancha(self, rng, par: ParametrosManchas, n_viagens: int) -> np.ndarray:
        """Contagem de viagens por mancha (grade L x L) em uma paisagem sorteada — ligação com a deposição."""
        w = self.pesos(self.campo(rng, par.comp_corr), par)
        _, idx = self.sortear_viagens(rng, w, n_viagens)
        return np.bincount(idx, minlength=w.size).reshape(w.shape)


def meia_distancia_visitacao(par: ParametrosManchas) -> float:
    """Distância (m) em que a visitação POR MANCHA cai à metade do centro: lambda * (ln 2)^(1/beta)."""
    return float(par.lambda_m * np.log(2.0) ** (1.0 / par.beta))


def abc_manchas(obs: dict, paisagem: Paisagem, n_dancas: int = 15, n_sorteios: int = 8000, n_sessoes: int = 50,
                aceitar: float = 0.02, seed=0) -> dict:
    """ABC dos parâmetros (lambda, beta, sigma_q, comp_corr) casando os quatro padrões de sessão de ``obs``."""
    from .dancas import estatisticas

    rng = np.random.default_rng(seed)
    LAM = np.exp(rng.uniform(np.log(150), np.log(3000), n_sorteios))
    BET = rng.uniform(0.7, 3.0, n_sorteios)
    SIG = rng.uniform(0.0, 2.5, n_sorteios)
    ELL = rng.choice(COMPRIMENTOS, n_sorteios)
    sim = np.empty((n_sorteios, 4))
    for k in range(n_sorteios):
        md, me = paisagem.sessoes(rng, ParametrosManchas(LAM[k], BET[k], SIG[k], float(ELL[k])), n_sessoes, n_dancas)
        e = estatisticas(md, me)
        sim[k] = (e["mediana_das_medianas"], e["mediana_das_medias"], e["razao_media_mediana"], e["dispersao_relativa"])
    alvo = np.array([obs["mediana_das_medianas"], obs["mediana_das_medias"], obs["razao_media_mediana"],
                     obs["dispersao_relativa"]])
    esc = np.array([0.1 * alvo[0], 0.1 * alvo[1], 0.05, 0.1 * alvo[3]])
    dist = np.sqrt((((sim - alvo) / esc) ** 2).sum(axis=1))
    ok = np.argsort(dist)[: max(20, int(aceitar * n_sorteios))]
    return {"lambda_m": LAM[ok], "beta": BET[ok], "sigma_q": SIG[ok], "comp_corr": ELL[ok], "dist": dist[ok],
            "sim": sim[ok]}


def parametros_aceitos(post: dict, k: int) -> ParametrosManchas:
    return ParametrosManchas(float(post["lambda_m"][k]), float(post["beta"][k]), float(post["sigma_q"][k]),
                             float(post["comp_corr"][k]))
