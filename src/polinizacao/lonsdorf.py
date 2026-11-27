"""Modelo analítico de visitação tipo Lonsdorf et al. (2009) para UM ninho (Limitação 2).

A visitação relativa de uma célula de flor *i* é proporcional à sua atratividade ``F_i`` vezes um núcleo
exponencial na distância ao ninho, normalizada (soma 1 sobre todas as flores):

    V_i = F_i * exp(-d_i / alpha) / sum_k F_k * exp(-d_k / alpha)

É uma versão **simplificada** com o mesmo núcleo de decaimento do InVEST; **não** é o código do InVEST
(sem estimativa de oferta de recursos de nidificação, sem várias guildas, sem sazonalidade).
O parâmetro ``alpha`` é a distância típica de forrageio (em células aqui; em metros na literatura).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def nucleo(d, alpha: float):
    """Núcleo exponencial exp(-d/alpha)."""
    if alpha <= 0:
        raise ValueError("alpha deve ser > 0")
    return np.exp(-np.asarray(d, dtype=float) / alpha)


def visitacao_relativa(atratividade, distancias, alpha: float) -> np.ndarray:
    """Participação de cada célula no forrageio (soma 1; zero onde ``atratividade`` é zero)."""
    f = np.asarray(atratividade, dtype=float)
    w = f * nucleo(distancias, alpha)
    total = w.sum()
    if total <= 0:
        raise ValueError("não há flores atrativas")
    return w / total


def intensidade_relativa(atratividade, distancias, alpha: float) -> np.ndarray:
    """Visitação por flor relativa à média (média 1 sobre as flores): N_flores * V_i."""
    f = np.asarray(atratividade, dtype=float)
    return visitacao_relativa(f, distancias, alpha) * np.count_nonzero(f)


def perfil_por_distancia(valores, distancias, mascara, largura_bin: float = 2.0):
    """Média de ``valores`` por faixa de distância, só sobre as células de ``mascara``.

    Retorna (centros, media, n); NaN nas faixas sem células.
    """
    if largura_bin <= 0:
        raise ValueError("largura_bin deve ser > 0")
    v = np.asarray(valores, dtype=float)
    d = np.asarray(distancias, dtype=float)
    m = np.asarray(mascara, dtype=bool)
    faixas = np.floor(d / largura_bin).astype(int)
    nf = int(faixas.max()) + 1
    n = np.bincount(faixas[m], minlength=nf)
    soma = np.bincount(faixas[m], weights=v[m], minlength=nf)
    with np.errstate(invalid="ignore", divide="ignore"):
        media = np.where(n > 0, soma / n, np.nan)
    return (np.arange(nf) + 0.5) * largura_bin, media, n


def meia_distancia_exponencial(alpha: float) -> float:
    """Distância em que um núcleo exponencial cai à metade: alpha * ln 2."""
    return float(alpha * np.log(2.0))


def eficiencia_prevista(intensidade, kappa: float):
    """Probabilidade de polinização 100*(1 - exp(-kappa * intensidade)) (%)."""
    return 100.0 * (1.0 - np.exp(-kappa * np.asarray(intensidade, dtype=float)))


def ajustar_kappa(intensidade, eficiencia_obs, pesos=None) -> float:
    """kappa que minimiza o erro quadrático entre a eficiência prevista e a observada (1 parâmetro livre)."""
    x = np.asarray(intensidade, dtype=float)
    y = np.asarray(eficiencia_obs, dtype=float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    w = np.ones(ok.sum()) if pesos is None else np.asarray(pesos, dtype=float)[ok]
    if ok.sum() < 2:
        return float("nan")
    r = minimize_scalar(lambda k: float(np.sum(w * (eficiencia_prevista(x[ok], k) - y[ok]) ** 2)),
                        bounds=(1e-6, 1e3), method="bounded")
    return float(r.x)
