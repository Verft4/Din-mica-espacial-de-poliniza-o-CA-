"""Calibração e validação por estatísticas publicadas, e identificabilidade (Limitação 2).

Não há dados brutos de campo neste projeto. A calibração usa **valores publicados** para fixar as duas
escalas identificáveis do modelo (metros por célula e segundos por passo); todo o resto (parâmetros de
deposição, estoque, retenção…) continua **não calibrado**, e a análise de identificabilidade mostra o
que os dados disponíveis poderiam ou não restringir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.stats import qmc

from .analise import ajustar_decaimento, ic95, meia_distancia

# ---- Âncoras da literatura (ver docs/plano_limitacoes.md) ------------------------------------------------
# Lonsdorf et al. (2009), Tabela 2: distância típica de forrageio alpha (m): ponto e intervalo.
ALPHA_M = {
    "Apis mellifera": (663.0, 562.0, 776.0),
    "Melipona fasciata": (578.0, 578.0, 578.0),
    "Trigona spp.": (70.0, 55.0, 87.0),          # ponto = ponto médio do intervalo 55-87 (escolha nossa)
    "Tetragonisca angustula": (22.0, 22.0, 22.0),
    "Plebeia spp.": (31.0, 28.0, 34.0),          # ponto = ponto médio do intervalo 28-34 (escolha nossa)
}
# Ricketts et al. (2008): meia-distância da visitação, em metros.
MEIA_DISTANCIA_RICKETTS_M = 600.0
# Duração de viagem de forrageio de A. mellifera (min): média e desvio (Johnson, 2025, tese Virginia Tech).
VIAGEM_MIN = {"néctar": (16.7, 8.8), "pólen": (24.5, 11.8)}


def cv_observado(tipo: str) -> float:
    media, dp = VIAGEM_MIN[tipo]
    return dp / media


def escala_espacial(alpha_cel: float, alpha_m: float) -> float:
    """Metros por célula: alpha em metros dividido por alpha em células."""
    if alpha_cel <= 0:
        raise ValueError("alpha_cel deve ser > 0")
    return alpha_m / alpha_cel


def escala_temporal(passos_viagem: float, duracao_s: float) -> float:
    """Segundos por passo: duração observada da viagem dividida pelos passos simulados."""
    if passos_viagem <= 0:
        raise ValueError("passos_viagem deve ser > 0")
    return duracao_s / passos_viagem


@dataclass
class PerfilAjustado:
    alpha_exp: float       # escala (células) do ajuste exponencial
    r2_exp: float
    r2_gauss: float
    d50: float             # meia-distância empírica (células)


def ajustar_perfil(centros, visitacao) -> PerfilAjustado:
    """Ajusta exponencial e gaussiana ao perfil de visitação (normalizado pelo máximo) e mede d50."""
    v = np.asarray(visitacao, dtype=float)
    mx = np.nanmax(v) if np.any(~np.isnan(v)) else np.nan
    if not np.isfinite(mx) or mx <= 0:
        return PerfilAjustado(*(float("nan"),) * 4)
    vn = v / mx
    e = ajustar_decaimento(centros, vn, "exponencial")
    g = ajustar_decaimento(centros, vn, "gaussiana")
    return PerfilAjustado(e["escala"], e["r2"], g["r2"], meia_distancia(centros, vn))


def escala_com_incerteza(alpha_cel_replicas, alpha_m_ponto: float, alpha_m_lo: float, alpha_m_hi: float):
    """Metros por célula: ponto, IC95 entre réplicas (do alpha_cel) e faixa pelo intervalo publicado.

    Retorna ``(ponto, (ic_lo, ic_hi), (faixa_lo, faixa_hi))``; a faixa combina os extremos do intervalo
    publicado com o IC95 das réplicas (conservadora: extremos de ambos).
    """
    a = np.asarray(alpha_cel_replicas, dtype=float)
    media, semi, _ = ic95(a)
    media, semi = float(media), float(semi)
    ponto = alpha_m_ponto / media
    a_lo, a_hi = media - semi, media + semi
    ic = (alpha_m_ponto / a_hi, alpha_m_ponto / a_lo)
    faixa = (alpha_m_lo / a_hi, alpha_m_hi / a_lo)
    return ponto, ic, faixa


# ------------------------------------------------------------------ identificabilidade
def hipercubo_latino(limites: dict[str, tuple[float, float]], n: int, seed=None) -> tuple[list[str], np.ndarray]:
    """Amostra por hipercubo latino em caixas [lo, hi]. Retorna (nomes, matriz n x k)."""
    nomes = list(limites)
    lo = np.array([limites[k][0] for k in nomes], dtype=float)
    hi = np.array([limites[k][1] for k in nomes], dtype=float)
    u = qmc.LatinHypercube(d=len(nomes), seed=seed).random(n)
    return nomes, qmc.scale(u, lo, hi)


def prcc(X, y):
    """Coeficiente de correlação parcial de postos (PRCC) de cada coluna de X com y, com p-valores.

    Cada coluna é controlada pelas demais (resíduos de regressão linear nos postos).
    Retorna (prcc, p). Colunas constantes dão NaN.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    R = np.column_stack([stats.rankdata(X[:, j]) for j in range(k)])
    ry = stats.rankdata(y)
    out = np.full(k, np.nan)
    p = np.full(k, np.nan)
    for j in range(k):
        outras = np.delete(R, j, axis=1)
        A = np.column_stack([np.ones(n), outras])
        rx = R[:, j] - A @ np.linalg.lstsq(A, R[:, j], rcond=None)[0]
        ryy = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        if np.ptp(R[:, j]) == 0 or np.ptp(ry) == 0:      # coluna (ou resposta) constante
            continue
        if rx.std() < 1e-12 or ryy.std() < 1e-12:
            continue
        r = float(np.corrcoef(rx, ryy)[0, 1])
        gl = n - k - 1
        if gl > 0 and abs(r) < 1:
            t = r * np.sqrt(gl / (1 - r * r))
            p[j] = 2 * stats.t.sf(abs(t), gl)
        out[j] = r
    return out, p
