#!/usr/bin/env python3
"""Roda UMA simulação e salva o mapa final.

Exemplos:
    python scripts/executar_simulacao.py --passos 120 --seed 1
    python scripts/executar_simulacao.py --alpha 4 --densidade 0.25 --saida figuras/minha_sim.png
    python scripts/executar_simulacao.py --mostrar          # abre uma janela em vez de só salvar
"""
import argparse
import sys
from pathlib import Path

import matplotlib

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--passos", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tamanho", type=int, default=60, help="lado da grade (células)")
    ap.add_argument("--densidade", type=float, default=0.15, help="densidade floral")
    ap.add_argument("--alpha", type=float, default=12.0, help="atração floral")
    ap.add_argument("--obstaculos", type=float, default=0.05, help="fração de obstáculos")
    ap.add_argument("--saida", default=str(RAIZ / "figuras" / "simulacao.png"))
    ap.add_argument("--mostrar", action="store_true", help="abrir janela interativa")
    a = ap.parse_args()

    if not a.mostrar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from polinizacao import ModeloPolinizacao, Parametros
    from polinizacao.visualizacao import desenhar_paisagem

    p = Parametros(linhas=a.tamanho, colunas=a.tamanho, densidade_floral=a.densidade,
                   alpha_atracao=a.alpha, frac_obstaculos=a.obstaculos)
    m = ModeloPolinizacao(p, seed=a.seed).executar(a.passos)
    print(f"abelhas: {m.n_abelhas} | receptoras polinizadas: {m.n_polinizadas}/{m.total_receptoras}"
          f" ({100 * m.taxa_polinizacao:.1f}%) | raio95: {m.raio_percentil():.1f} células")

    fig, ax = plt.subplots(figsize=(7.2, 6))
    desenhar_paisagem(m, ax)
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.saida, dpi=200, bbox_inches="tight")
    print("figura salva em", a.saida)
    if a.mostrar:
        plt.show()


if __name__ == "__main__":
    main()
