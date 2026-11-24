"""Experimentos de Monte Carlo e varreduras de parâmetros."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .analise import ajustar_decaimento, ic95, media_dp, raio_eficaz
from .modelo import ModeloPolinizacao, Parametros


@dataclass
class ResultadoMC:
    """Resultados de N simulações independentes com os mesmos parâmetros."""

    params: Parametros
    passos: int
    seed: int
    largura_bin: float
    hist_polinizadas: np.ndarray      # (n_sim, passos) flores polinizadas acumuladas
    hist_visitas: np.ndarray          # (n_sim, passos) visitas a flores por passo
    centros: np.ndarray               # (n_faixas,) centro de cada faixa de distância
    eficiencia: np.ndarray            # (n_sim, n_faixas) % por simulação (NaN sem receptoras)
    n_receptoras: np.ndarray          # (n_sim, n_faixas)
    taxa_final: np.ndarray            # (n_sim,) fração de receptoras polinizadas
    raio95: np.ndarray                # (n_sim,) raio que contém 95% das polinizadas
    tempo_total_s: float = 0.0
    extras: dict = field(default_factory=dict)

    @property
    def n_sim(self) -> int:
        return self.hist_polinizadas.shape[0]

    def perfil_medio(self):
        """Eficiência média, desvio-padrão e IC95% entre simulações, por faixa."""
        media, dp, n = media_dp(self.eficiencia)
        _, semi, _ = ic95(self.eficiencia)
        return media, dp, semi, n

    def resumo(self) -> dict:
        """Estatísticas-chave para relatórios (valores em tipos nativos do Python)."""
        media, dp, semi, _ = self.perfil_medio()
        aj_exp = ajustar_decaimento(self.centros, media, "exponencial")
        aj_gau = ajustar_decaimento(self.centros, media, "gaussiana")
        taxa_m, taxa_dp, _ = media_dp(100.0 * self.taxa_final)
        r95_m, r95_dp, r95_n = media_dp(self.raio95)
        return {
            "n_sim": self.n_sim,
            "passos": self.passos,
            "taxa_final_media_pct": float(taxa_m),
            "taxa_final_dp_pct": float(taxa_dp),
            "polinizadas_final_media": float(np.mean(self.hist_polinizadas[:, -1])),
            "polinizadas_final_dp": float(np.std(self.hist_polinizadas[:, -1], ddof=1)),
            "raio95_media": float(r95_m),
            "raio95_dp": float(r95_dp),
            "raio_eficaz_5pct": raio_eficaz(self.centros, media, 5.0),
            "ajuste_exponencial": aj_exp,
            "ajuste_gaussiano": aj_gau,
            "tempo_total_s": float(self.tempo_total_s),
            "tempo_por_simulacao_s": float(self.tempo_total_s / self.n_sim),
        }


def executar_monte_carlo(
    params: Parametros | None = None,
    n_simulacoes: int = 50,
    passos: int = 120,
    seed: int = 12345,
    largura_bin: float = 2.0,
    progresso=None,
) -> ResultadoMC:
    """Roda ``n_simulacoes`` réplicas independentes e reprodutíveis.

    Cada réplica usa uma semente filha de ``SeedSequence(seed)``; o mesmo
    ``seed`` sempre reproduz exatamente os mesmos resultados.
    """
    if n_simulacoes < 1 or passos < 1:
        raise ValueError("n_simulacoes e passos devem ser >= 1")
    params = params if params is not None else Parametros()
    filhas = np.random.SeedSequence(seed).spawn(n_simulacoes)

    hist_pol, hist_vis, efic, tots, taxas, raios = [], [], [], [], [], []
    centros = None
    inicio = time.perf_counter()
    for k, filha in enumerate(filhas):
        modelo = ModeloPolinizacao(params, seed=filha).executar(passos)
        centros, e, t = modelo.eficiencia_por_distancia(largura_bin)
        hist_pol.append(modelo.hist_polinizadas)
        hist_vis.append(modelo.hist_visitas)
        efic.append(e)
        tots.append(t)
        taxas.append(modelo.taxa_polinizacao)
        raios.append(modelo.raio_percentil(0.95))
        if progresso is not None:
            progresso(k + 1, n_simulacoes)
    tempo = time.perf_counter() - inicio

    # As grades têm o mesmo tamanho, logo o mesmo número de faixas.
    return ResultadoMC(
        params=params,
        passos=passos,
        seed=seed,
        largura_bin=largura_bin,
        hist_polinizadas=np.array(hist_pol, dtype=float),
        hist_visitas=np.array(hist_vis, dtype=float),
        centros=centros,
        eficiencia=np.array(efic, dtype=float),
        n_receptoras=np.array(tots, dtype=int),
        taxa_final=np.array(taxas, dtype=float),
        raio95=np.array(raios, dtype=float),
        tempo_total_s=tempo,
    )


def varrer_parametro(
    base: Parametros,
    nome: str,
    valores,
    n_simulacoes: int = 30,
    passos: int = 120,
    seed: int = 2025,
    largura_bin: float = 2.0,
) -> dict:
    """Análise de sensibilidade: roda o Monte Carlo para cada valor de ``nome``.

    Todos os valores usam a mesma ``seed`` (números aleatórios comuns), o que
    reduz o ruído nas comparações entre cenários.
    """
    if not hasattr(base, nome):
        raise AttributeError(f"Parametros não tem o campo '{nome}'")
    return {
        v: executar_monte_carlo(base.com(**{nome: v}), n_simulacoes, passos, seed, largura_bin)
        for v in valores
    }
