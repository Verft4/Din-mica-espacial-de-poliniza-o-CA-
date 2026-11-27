#!/usr/bin/env python3
"""Roda um experimento de Monte Carlo e salva as figuras (série temporal e eficiência x distância).

Exemplo:
    python scripts/executar_monte_carlo.py --n 50 --passos 120 --seed 12345
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from polinizacao import Parametros, executar_monte_carlo  # noqa: E402
from polinizacao.visualizacao import desenhar_eficiencia, desenhar_serie_temporal  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="número de simulações")
    ap.add_argument("--passos", type=int, default=120)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--densidade", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=12.0)
    ap.add_argument("--obstaculos", type=float, default=0.05)
    ap.add_argument("--saida", default=str(RAIZ / "figuras" / "monte_carlo.png"))
    a = ap.parse_args()

    p = Parametros(densidade_floral=a.densidade, alpha_atracao=a.alpha, frac_obstaculos=a.obstaculos)
    res = executar_monte_carlo(p, a.n, a.passos, a.seed,
                               progresso=lambda k, n: print(f"\r  simulação {k}/{n}", end=""))
    print()
    s = res.resumo()
    print(f"receptoras polinizadas: {s['taxa_final_media_pct']:.1f} ± {s['taxa_final_dp_pct']:.1f} %")
    print(f"raio que contém 95% das polinizadas: {s['raio95_media']:.1f} ± {s['raio95_dp']:.1f} células")
    print(f"raio onde a eficiência cai abaixo de 5%: {s['raio_eficaz_5pct']}")
    print(f"ajuste gaussiano R2 = {s['ajuste_gaussiano']['r2']:.2f} | exponencial R2 = {s['ajuste_exponencial']['r2']:.2f}")
    print(f"tempo: {s['tempo_por_simulacao_s'] * 1e3:.0f} ms por simulação")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    desenhar_serie_temporal(res, a1)
    a1.set_title(f"Evolução temporal (N = {a.n})")
    desenhar_eficiencia(res, a2)
    a2.set_title("Eficiência por distância")
    fig.tight_layout()
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.saida, dpi=200, bbox_inches="tight")
    print("figura salva em", a.saida)


if __name__ == "__main__":
    main()
