"""Estatística sobre os resultados das simulações."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def media_dp(x, eixo: int = 0):
    """Média e desvio-padrão amostral (ddof=1) ignorando NaN, mais o n válido."""
    x = np.asarray(x, dtype=float)
    n = np.sum(~np.isnan(x), axis=eixo)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # faixas sem nenhum dado válido
        media = np.nanmean(x, axis=eixo)
        dp = np.where(n > 1, np.nanstd(x, axis=eixo, ddof=1), np.nan)
    return media, dp, n


def ic95(x, eixo: int = 0):
    """Intervalo de confiança de 95% (t de Student) para a média.

    Retorna (media, semi_largura, n). A semi-largura é NaN para n < 2.
    """
    media, dp, n = media_dp(x, eixo)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_crit = np.where(n > 1, stats.t.ppf(0.975, np.maximum(n - 1, 1)), np.nan)
        semi = t_crit * dp / np.sqrt(np.maximum(n, 1))
    return media, semi, n


def _exponencial(d, a, k):
    return a * np.exp(-k * d)


def _gaussiana(d, a, k):
    return a * np.exp(-((k * d) ** 2))


_MODELOS = {"exponencial": _exponencial, "gaussiana": _gaussiana}


def ajustar_decaimento(distancias, eficiencia, modelo: str = "exponencial"):
    """Ajusta a curva de decaimento da eficiência com a distância.

    ``modelo``:
      * ``"exponencial"``: E(d) = A * exp(-lambda * d)        (forma usada no InVEST)
      * ``"gaussiana"``:   E(d) = A * exp(-(lambda * d)**2)   (platô perto da colmeia)

    A amplitude A é limitada a [0, 100] (eficiência é uma porcentagem). Usa só
    as faixas com eficiência definida. Retorna ``A``, ``lambda``, ``r2``
    (sobre os dados) e ``escala`` (= 1/lambda, em células); NaN se o ajuste falhar.
    """
    f = _MODELOS[modelo]
    d = np.asarray(distancias, dtype=float)
    e = np.asarray(eficiencia, dtype=float)
    ok = ~np.isnan(e) & ~np.isnan(d)
    d, e = d[ok], e[ok]
    falha = {"modelo": modelo, "A": np.nan, "lambda": np.nan, "r2": np.nan, "escala": np.nan}
    if d.size < 3 or np.all(e <= 0):
        return falha
    try:
        (a, k), _ = curve_fit(
            f, d, e,
            p0=(min(max(float(e.max()), 1.0), 100.0), 0.1),
            bounds=([0.0, 1e-6], [100.0, 10.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return falha
    ss_res = float(np.sum((e - f(d, a, k)) ** 2))
    ss_tot = float(np.sum((e - e.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"modelo": modelo, "A": float(a), "lambda": float(k), "r2": float(r2),
            "escala": float(1.0 / k)}


def ajustar_exponencial(distancias, eficiencia):
    """Atalho para ``ajustar_decaimento(..., modelo="exponencial")``."""
    return ajustar_decaimento(distancias, eficiencia, "exponencial")


def raio_eficaz(distancias, eficiencia, limiar: float = 5.0) -> float:
    """Primeira distância, após o pico, em que a eficiência média cai abaixo do limiar (%).

    Retorna NaN se a eficiência nunca cair abaixo do limiar.
    """
    d = np.asarray(distancias, dtype=float)
    e = np.asarray(eficiencia, dtype=float)
    ok = ~np.isnan(e)
    d, e = d[ok], e[ok]
    if e.size == 0:
        return float("nan")
    pico = int(np.argmax(e))
    abaixo = np.where(e[pico:] < limiar)[0]
    if abaixo.size == 0:
        return float("nan")
    return float(d[pico + abaixo[0]])


def sd_binomial_esperado(eficiencia, n_receptoras):
    """Desvio-padrão (em pontos percentuais) esperado só por amostragem binomial.

    Para cada faixa de distância, se cada receptora fosse polinizada de forma
    independente com a probabilidade média ``p`` da faixa, a eficiência de uma
    simulação com ``n`` receptoras teria variância p(1-p)/n. Comparar o desvio
    observado entre simulações com este valor separa o ruído de amostragem
    (poucas flores por faixa) da variabilidade causada pela configuração da
    paisagem e pela dinâmica das abelhas.

    ``eficiencia`` em % e ``n_receptoras`` têm forma (n_sim, n_faixas).
    """
    e = np.asarray(eficiencia, dtype=float) / 100.0
    n = np.asarray(n_receptoras, dtype=float)
    n = np.where(n > 0, n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = np.nanmean(e, axis=0)
        var = p * (1.0 - p) / n
        return 100.0 * np.sqrt(np.nanmean(var, axis=0))


def meia_distancia(distancias, valores) -> float:
    """Menor distância, a partir do pico, em que o perfil cai à metade do seu valor máximo.

    Usa interpolação linear entre as duas faixas que cruzam o nível (NaN ignorados).
    Retorna NaN se o perfil nunca cai à metade.
    """
    d = np.asarray(distancias, dtype=float)
    v = np.asarray(valores, dtype=float)
    ok = ~np.isnan(v) & ~np.isnan(d)
    d, v = d[ok], v[ok]
    if v.size < 2 or v.max() <= 0:
        return float("nan")
    pico = int(np.argmax(v))
    alvo = v[pico] / 2.0
    for k in range(pico + 1, v.size):
        if v[k] <= alvo:
            v0, v1, d0, d1 = v[k - 1], v[k], d[k - 1], d[k]
            return float(d0 + (v0 - alvo) / (v0 - v1) * (d1 - d0)) if v0 != v1 else float(d1)
    return float("nan")
