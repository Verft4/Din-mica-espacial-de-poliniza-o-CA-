"""Funções de desenho. Nenhuma chama ``plt.show()``: quem chama decide salvar ou exibir."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .modelo import NOMES_ESTADOS, ModeloPolinizacao
from .analise import ajustar_decaimento

# Cores por código de estado (0..6), as mesmas da Figura 1 do artigo.
CORES_ESTADOS = ["black", "#cccccc", "#ffee00", "#ff9d00", "#8b2d2d", "#00e5ff", "#ff0000"]
_CMAP = mcolors.ListedColormap(CORES_ESTADOS)
_NORM = mcolors.BoundaryNorm(np.arange(-0.5, 7.5, 1.0), _CMAP.N)


def desenhar_paisagem(modelo: ModeloPolinizacao, ax=None, titulo: str | None = None):
    """Mapa da grade com as abelhas sobrepostas e legenda dos estados."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(modelo.mapa_com_abelhas(), cmap=_CMAP, norm=_NORM, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titulo or f"Simulação do modelo CA (passo {modelo.passo_atual})")
    pedacos = [plt.Rectangle((0, 0), 1, 1, facecolor=CORES_ESTADOS[k], edgecolor="0.4")
               for k in range(7)]
    ax.legend(pedacos, [NOMES_ESTADOS[k] for k in range(7)], loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    return ax


def desenhar_eficiencia(resultado, ax=None, ajuste: bool = True):
    """Eficiência por distância: média entre simulações com IC95% e ajuste exponencial."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    media, dp, semi, n = resultado.perfil_medio()
    d = resultado.centros
    ok = ~np.isnan(media)
    ax.bar(d[ok], media[ok], width=0.85 * resultado.largura_bin, color="#ffbb44",
           edgecolor="black", alpha=0.85, label="Média das simulações")
    ax.errorbar(d[ok], media[ok], yerr=np.nan_to_num(semi[ok]), fmt="none", ecolor="black",
                capsize=3, label="IC 95% da média")
    if ajuste:
        xs = np.linspace(0, d[ok].max(), 200)
        ae = ajustar_decaimento(d, media, "exponencial")
        ag = ajustar_decaimento(d, media, "gaussiana")
        if not np.isnan(ae["lambda"]):
            ax.plot(xs, ae["A"] * np.exp(-ae["lambda"] * xs), "r--", lw=2,
                    label=rf"Exponencial: $R^2$={ae['r2']:.2f}")
        if not np.isnan(ag["lambda"]):
            ax.plot(xs, ag["A"] * np.exp(-(ag["lambda"] * xs) ** 2), color="navy", lw=2,
                    label=rf"Gaussiana: $R^2$={ag['r2']:.2f}")
    ax.set_xlabel("Distância da colmeia (células)")
    ax.set_ylabel("Receptoras polinizadas (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def desenhar_serie_temporal(resultado, ax=None):
    """Flores polinizadas ao longo do tempo: média ± desvio-padrão e réplicas individuais."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))
    t = np.arange(1, resultado.passos + 1)
    h = resultado.hist_polinizadas
    media, dp = h.mean(axis=0), h.std(axis=0, ddof=1) if h.shape[0] > 1 else np.zeros(h.shape[1])
    for linha in h[: min(10, h.shape[0])]:
        ax.plot(t, linha, color="0.7", lw=0.7)
    ax.plot(t, media, color="darkgreen", lw=2, label="Média")
    ax.fill_between(t, media - dp, media + dp, color="lightgreen", alpha=0.4,
                    label=r"$\pm$ 1 desvio-padrão")
    ax.set_xlabel("Tempo (passos)")
    ax.set_ylabel("Flores polinizadas (acumulado)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax
