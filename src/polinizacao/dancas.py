"""Distâncias de forrageio de danças decodificadas (Dryad 10.5061/dryad.c2fqz618f) — calibração do núcleo.

O arquivo público tem **uma linha por sessão de observação** (colmeia x dia): mediana e média das
distâncias "traduzidas" (metros) das danças da sessão. NÃO tem as danças individuais (≈ 2 827 danças
em 183 sessões de 20 colmeias, ≈ 15 danças por sessão). Por isso o ajuste do núcleo de distância é
feito **no nível de sessão**, por simulação de sessões (ABC), com o número de danças por sessão
como suposição explícita (``n_dancas``).

Núcleo de distância de uma viagem (Leadbeater et al. 2024): mistura de batedoras (exponencial) e
recrutadas (Rayleigh):  X ~ p * Exp(media = mu) + (1 - p) * Rayleigh(sigma).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sessoes:
    colmeia: np.ndarray      # código da colmeia (20 colmeias)
    uso: np.ndarray          # "Agricultural" | "Urban"
    mediana_m: np.ndarray    # mediana da distância das danças da sessão (m)
    media_m: np.ndarray      # média da distância das danças da sessão (m)

    def __len__(self) -> int:
        return len(self.colmeia)

    def subconjunto(self, mascara) -> "Sessoes":
        m = np.asarray(mascara, dtype=bool)
        return Sessoes(self.colmeia[m], self.uso[m], self.mediana_m[m], self.media_m[m])


def carregar_sessoes(caminho) -> Sessoes:
    """Lê o CSV do Dryad (colunas ``site``, ``Land_use``, ``median/mean.translated.distance``)."""
    col, uso, med, mea = [], [], [], []
    with open(Path(caminho), newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            try:
                a, b = float(linha["median.translated.distance"]), float(linha["mean.translated.distance"])
            except (ValueError, KeyError):
                continue
            if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
                continue
            col.append(linha["site"])
            uso.append(linha["Land_use"])
            med.append(a)
            mea.append(b)
    if not col:
        raise ValueError("nenhuma sessão válida no arquivo")
    return Sessoes(np.array(col), np.array(uso), np.array(med), np.array(mea))


def dobras_por_colmeia(s: Sessoes, seed=0) -> tuple[np.ndarray, np.ndarray]:
    """Duas dobras disjuntas **por colmeia** (nunca separa sessões da mesma colmeia), estratificadas por uso do solo."""
    rng = np.random.default_rng(seed)
    dobra = {}
    for u in sorted(set(s.uso)):
        cols = sorted({c for c, uu in zip(s.colmeia, s.uso) if uu == u})
        rng.shuffle(cols)
        for k, c in enumerate(cols):
            dobra[c] = k % 2
    d = np.array([dobra[c] for c in s.colmeia])
    return d == 0, d == 1


def estatisticas(mediana_m, media_m) -> dict:
    """Padrões usados na calibração: mediana das medianas, mediana das médias, razão média/mediana e dispersão relativa."""
    med, mea = np.asarray(mediana_m, float), np.asarray(media_m, float)
    q1, q3 = np.percentile(med, [25, 75])
    return {"mediana_das_medianas": float(np.median(med)), "mediana_das_medias": float(np.median(mea)),
            "razao_media_mediana": float(np.median(mea / med)), "dispersao_relativa": float((q3 - q1) / np.median(med))}


def amostrar_mistura(rng, p_batedora, media_exp, sigma_ray, tamanho):
    """Distâncias de viagem: exponencial (média ``media_exp``) com prob. ``p_batedora``; Rayleigh (``sigma_ray``) c.c.

    ``media_exp`` e ``sigma_ray`` podem ser arrays que se propagam com ``tamanho``.
    """
    if not 0.0 <= p_batedora <= 1.0:
        raise ValueError("p_batedora deve estar em [0, 1]")
    bat = rng.random(tamanho) < p_batedora
    e = rng.exponential(1.0, tamanho) * media_exp
    r = rng.rayleigh(1.0, tamanho) * sigma_ray
    return np.where(bat, e, r)


def simular_sessoes(rng, p, media_exp, sigma_ray, heterogeneidade, n_sessoes, n_dancas):
    """Simula sessões: cada sessão tem escala própria (lognormal, média 1) e ``n_dancas`` viagens da mistura.

    Retorna (medianas, medias) por sessão.
    """
    f = np.exp(heterogeneidade * rng.standard_normal(n_sessoes) - heterogeneidade ** 2 / 2.0)
    x = amostrar_mistura(rng, p, media_exp * f[:, None], sigma_ray * f[:, None], (n_sessoes, n_dancas))
    return np.median(x, axis=1), x.mean(axis=1)


def abc_mistura(obs: dict, n_dancas: int = 15, n_sorteios: int = 20000, n_sessoes: int = 80,
                aceitar: float = 0.02, seed=0) -> dict:
    """ABC: infere (p, media_exp, sigma_ray, heterogeneidade) casando os padrões ``obs`` (ver :func:`estatisticas`).

    A distância é a soma dos quadrados dos desvios padronizados (10% para as escalas, 0,05 para a razão).
    Retorna os sorteios aceitos e as estatísticas simuladas.
    """
    rng = np.random.default_rng(seed)
    P = rng.random(n_sorteios)
    MU = np.exp(rng.uniform(np.log(200), np.log(3000), n_sorteios))
    SG = np.exp(rng.uniform(np.log(150), np.log(2500), n_sorteios))
    TAU = rng.uniform(0.0, 0.9, n_sorteios)
    sim = np.empty((n_sorteios, 4))
    for k in range(n_sorteios):
        md, me = simular_sessoes(rng, P[k], MU[k], SG[k], TAU[k], n_sessoes, n_dancas)
        q1, q3 = np.percentile(md, [25, 75])
        sim[k] = (np.median(md), np.median(me), np.median(me / md), (q3 - q1) / np.median(md))
    alvo = np.array([obs["mediana_das_medianas"], obs["mediana_das_medias"], obs["razao_media_mediana"],
                     obs["dispersao_relativa"]])
    esc = np.array([0.1 * alvo[0], 0.1 * alvo[1], 0.05, 0.1 * alvo[3]])
    dist = np.sqrt((((sim - alvo) / esc) ** 2).sum(axis=1))
    ok = np.argsort(dist)[: max(20, int(aceitar * n_sorteios))]
    return {"p": P[ok], "media_exp": MU[ok], "sigma_ray": SG[ok], "tau": TAU[ok], "dist": dist[ok],
            "sim": sim[ok], "n_dancas": n_dancas}


def estatisticas_da_viagem(post: dict, seed=0, n=40000) -> dict:
    """Mediana, média e CV da distância de UMA viagem, uma por conjunto de parâmetros aceito (com incerteza)."""
    rng = np.random.default_rng(seed)
    med, mea, cv = [], [], []
    for p, mu, sg in zip(post["p"], post["media_exp"], post["sigma_ray"]):
        x = amostrar_mistura(rng, p, mu, sg, n)
        med.append(np.median(x)); mea.append(x.mean()); cv.append(x.std() / x.mean())
    return {"mediana": np.array(med), "media": np.array(mea), "cv": np.array(cv)}


def prever_sessoes_do_modelo(dist_viagens_celulas, m_por_celula: float, n_dancas: int, n_sessoes: int, seed=0):
    """Sessões previstas pelo modelo: ``n_dancas`` distâncias máximas de viagem (células) x metros por célula.

    ``dist_viagens_celulas`` pode ser um vetor (viagens reunidas) ou uma **lista de vetores, um por réplica**.
    No segundo caso cada sessão sorteia as viagens dentro de UMA réplica (uma paisagem, como uma colmeia), o que
    preserva a heterogeneidade entre sessões; reunir tudo a apagaria por construção.
    """
    rng = np.random.default_rng(seed)
    if isinstance(dist_viagens_celulas, (list, tuple)):
        reps = [np.asarray(r, float) for r in dist_viagens_celulas if len(r) > 0]
        if not reps:
            raise ValueError("nenhuma viagem registrada")
        x = np.empty((n_sessoes, n_dancas))
        for i in range(n_sessoes):
            x[i] = rng.choice(reps[rng.integers(len(reps))], size=n_dancas, replace=True)
        x *= m_por_celula
    else:
        d = np.asarray(dist_viagens_celulas, float) * m_por_celula
        x = rng.choice(d, size=(n_sessoes, n_dancas), replace=True)
    return np.median(x, axis=1), x.mean(axis=1)
