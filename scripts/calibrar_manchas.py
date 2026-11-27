#!/usr/bin/env python3
"""Passo 3 do plano: modelo em duas escalas (célula = mancha) calibrado e validado com as danças.

Critério de aceite (definido ANTES de rodar): as 4 estatísticas de sessão usadas no ajuste devem cair dentro do
intervalo preditivo de 90% nos DOIS sentidos da validação entre colmeias (8 verificações), e as 2 estatísticas
de cauda que NÃO entram no ajuste (P90/P10 das medianas; fração de sessões com mediana > 1500 m) devem
também cair dentro (4 verificações).

Uso:  python scripts/calibrar_manchas.py [--rapido]
Saídas: figuras/fig12_manchas.png · resultados/resultados_manchas.json · artigo/numeros_manchas.tex · artigo/tab_manchas.tex
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from polinizacao import dancas as dc  # noqa: E402
from polinizacao import manchas as mc  # noqa: E402

SEED = 4242                      # mesma divisão de colmeias de calibrar_dancas.py
MACROS: dict[str, str] = {}
NOMES = ("mediana_das_medianas", "mediana_das_medias", "razao_media_mediana", "dispersao_relativa",
         "p90_p10", "frac_longas")
ROT = {"mediana_das_medianas": "Mediana das medianas (m)", "mediana_das_medias": "Mediana das médias (m)",
       "razao_media_mediana": "Razão média/mediana", "dispersao_relativa": "Dispersão relativa (IQR/mediana)",
       "p90_p10": "P90/P10 das medianas (fora do ajuste)", "frac_longas": "Fração de sessões > 1500 m (fora do ajuste)"}
CASAS = {"mediana_das_medianas": 0, "mediana_das_medias": 0, "razao_media_mediana": 2, "dispersao_relativa": 2,
         "p90_p10": 1, "frac_longas": 2}


def pt(x, casas=0):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{casas}f}".replace(".", "{,}")


def ic(x, casas=0, q=(5, 95)):
    lo, md, hi = np.percentile(x, [q[0], 50, q[1]])
    return rf"{pt(md, casas)} ({pt(lo, casas)}--{pt(hi, casas)})"


def mac(nome, valor):
    assert nome.isalpha(), nome
    MACROS[nome] = str(valor)


def limpar(o):
    if isinstance(o, dict):
        return {str(k): limpar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpar(v) for v in o]
    if isinstance(o, np.ndarray):
        return limpar(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def estats6(md, me):
    e = dc.estatisticas(md, me)
    e["p90_p10"] = float(np.percentile(md, 90) / np.percentile(md, 10))
    e["frac_longas"] = float(np.mean(md > 1500.0))
    return e


def predicao_manchas(pais, post, n_sessoes, n_dancas, seed, maximo=150):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(post["lambda_m"]), size=min(maximo, len(post["lambda_m"])), replace=False)
    out = []
    for k in idx:
        md, me = pais.sessoes(rng, mc.parametros_aceitos(post, int(k)), n_sessoes, n_dancas)
        e = estats6(md, me)
        out.append([e[n] for n in NOMES])
    return np.array(out)


def predicao_mistura(post, n_sessoes, n_dancas, seed, maximo=150):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(post["p"]), size=min(maximo, len(post["p"])), replace=False)
    out = []
    for k in idx:
        md, me = dc.simular_sessoes(rng, post["p"][k], post["media_exp"][k], post["sigma_ray"][k], post["tau"][k],
                                    n_sessoes, n_dancas)
        e = estats6(md, me)
        out.append([e[n] for n in NOMES])
    return np.array(out)


def resumo(pred, alvo):
    r = {}
    for i, n in enumerate(NOMES):
        lo, md, hi = np.percentile(pred[:, i], [5, 50, 95])
        r[n] = {"obs": alvo[n], "pred_5": float(lo), "pred_mediana": float(md), "pred_95": float(hi),
                "dentro": bool(lo <= alvo[n] <= hi)}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RAIZ / "dados" / "dancas_sessoes_dryad.csv"))
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    NS, NSESS, NCAMPOS = (400, 30, 8) if args.rapido else (6000, 50, 60)
    ND = 15
    t0 = time.perf_counter()

    S = dc.carregar_sessoes(args.csv)
    ma, mb = dc.dobras_por_colmeia(S, seed=SEED)
    SA, SB = S.subconjunto(ma), S.subconjunto(mb)
    obs = estats6(S.mediana_m, S.media_m)
    obsA, obsB = estats6(SA.mediana_m, SA.media_m), estats6(SB.mediana_m, SB.media_m)
    pais = mc.Paisagem(lado_celulas=141, lado_m=100.0, n_campos=NCAMPOS, seed=SEED)
    print(f"{len(S)} sessões; paisagem {pais.L}x{pais.L} manchas de {pais.lado_m:.0f} m")

    def ajustar(o, semente):
        return mc.abc_manchas(o, pais, n_dancas=ND, n_sorteios=NS, n_sessoes=NSESS, seed=semente)

    print("ABC manchas: todas, A, B ...")
    post = ajustar(obs, SEED + 1)
    postA, postB = ajustar(obsA, SEED + 2), ajustar(obsB, SEED + 3)
    print("ABC mistura (referência): A, B ...")
    mixA = dc.abc_mistura(obsA, n_dancas=ND, n_sorteios=NS, seed=SEED + 4)
    mixB = dc.abc_mistura(obsB, n_dancas=ND, n_sorteios=NS, seed=SEED + 5)

    # validação entre colmeias
    val = {}
    for rot, pm_, px_, alvo, nS in (("A->B", postA, mixA, obsB, len(SB)), ("B->A", postB, mixB, obsA, len(SA))):
        val[rot] = {"manchas": resumo(predicao_manchas(pais, pm_, nS, ND, SEED + 6), alvo),
                    "mistura": resumo(predicao_mistura(px_, nS, ND, SEED + 7), alvo)}
    def contar(modelo, nomes):
        return sum(val[r][modelo][n]["dentro"] for r in val for n in nomes)
    n4 = contar("manchas", NOMES[:4]), contar("mistura", NOMES[:4])
    n2 = contar("manchas", NOMES[4:]), contar("mistura", NOMES[4:])
    aceite = n4[0] == 8 and n2[0] == 4

    # parâmetros e quantidades derivadas
    meias = np.array([mc.meia_distancia_visitacao(mc.parametros_aceitos(post, k)) for k in range(len(post["beta"]))])
    p_gauss = float(np.mean(post["beta"] > 1.5))

    # figura
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.5))
    x = np.sort(S.mediana_m)
    a1.step(x, np.arange(1, x.size + 1) / x.size, color="k", lw=2)
    rng = np.random.default_rng(SEED + 8)
    for k in rng.choice(len(post["beta"]), size=min(60, len(post["beta"])), replace=False):
        md, _ = pais.sessoes(rng, mc.parametros_aceitos(post, int(k)), 183, ND)
        xs = np.sort(md)
        a1.step(xs, np.arange(1, xs.size + 1) / xs.size, color="#4c9f70", alpha=0.12, lw=1)
    a1.set_xscale("log")
    a1.set_xlabel("Mediana da distância de dança por sessão (m)")
    a1.set_ylabel("Fração acumulada")
    a1.set_title("Modelo de manchas: sessões simuladas × observadas")
    a1.legend(handles=[plt.Line2D([], [], color="k", lw=2), plt.Line2D([], [], color="#4c9f70", alpha=0.7)],
              labels=["Observado (183 sessões)", "Manchas (ABC)"], loc="lower right")
    cores = {"manchas": "#4c9f70", "mistura": "#1b6ca8"}
    for off, (modelo, rot) in zip((-0.12, 0.12), (("manchas", "Manchas (duas escalas)"), ("mistura", "Núcleo fenomenológico"))):
        for r_, mk in (("A->B", "o"), ("B->A", "s")):
            for i, n in enumerate(NOMES):
                v = val[r_][modelo][n]
                o = v["obs"] if v["obs"] != 0 else 1e-9
                yy = i + off + (0.06 if r_ == "B->A" else -0.06)
                a2.plot([v["pred_5"] / o, v["pred_95"] / o], [yy, yy], color=cores[modelo], lw=2, alpha=0.85)
                a2.plot(v["pred_mediana"] / o, yy, mk, color=cores[modelo], ms=4,
                        label=rot if (i == 0 and r_ == "A->B") else None)
    a2.axvline(1.0, color="k", ls=":")
    a2.set_yticks(range(len(NOMES)), [ROT[n].replace(" (fora do ajuste)", "*") for n in NOMES], fontsize=8)
    a2.invert_yaxis()
    a2.set_xscale("log")
    a2.set_xlim(0.12, 14)
    a2.set_xticks([0.25, 0.5, 1, 2, 4, 8], ["0,25", "0,5", "1", "2", "4", "8"])
    a2.minorticks_off()
    a2.set_xlabel("Previsto / observado na metade retida (intervalo de 90%)")
    a2.set_title("Validação entre colmeias (* = fora do ajuste)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RAIZ / "figuras" / "fig12_manchas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # macros e tabela
    mac("ManDraws", NS); mac("ManSessoesPorSorteio", NSESS)
    mac("ManLambda", ic(post["lambda_m"])); mac("ManBeta", ic(post["beta"], 1)); mac("ManSigma", ic(post["sigma_q"], 1))
    mac("ManEll", ic(post["comp_corr"] * pais.lado_m)); mac("ManMeia", ic(meias)); mac("ManPGauss", pt(p_gauss, 2))
    mac("ManDentroQuatro", n4[0]); mac("ManDentroQuatroMix", n4[1])
    mac("ManDentroCauda", n2[0]); mac("ManDentroCaudaMix", n2[1])
    mac("ManAceite", "atendido" if aceite else "não atendido")
    for r_, tag in (("A->B", "AB"), ("B->A", "BA")):
        for n, c in zip(NOMES, ("Med", "Mea", "Raz", "Disp", "Cauda", "Longas")):
            mac(f"ManPred{tag}{c}", ic(np.array([val[r_]["manchas"][n]["pred_5"], val[r_]["manchas"][n]["pred_mediana"],
                                                  val[r_]["manchas"][n]["pred_95"]]), CASAS[n]))
    linhas = [r"\begin{tabular}{lcccccc}", r"\toprule",
              r"Estatística & Obs.\ B & Manchas A$\to$B & Mistura A$\to$B & Obs.\ A & Manchas B$\to$A & Mistura B$\to$A\\",
              r"\midrule"]
    def cel(v, c):
        txt = rf"{pt(v['pred_5'], c)}--{pt(v['pred_95'], c)}"
        return txt if v["dentro"] else rf"\textbf{{{txt}}}"
    for n in NOMES:
        c = CASAS[n]
        ab, ba = val["A->B"], val["B->A"]
        linhas.append(rf"{ROT[n]} & {pt(ab['manchas'][n]['obs'], c)} & {cel(ab['manchas'][n], c)} & "
                      rf"{cel(ab['mistura'][n], c)} & {pt(ba['manchas'][n]['obs'], c)} & {cel(ba['manchas'][n], c)} & "
                      rf"{cel(ba['mistura'][n], c)}\\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    (RAIZ / "artigo" / "tab_manchas.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    (RAIZ / "artigo" / "numeros_manchas.tex").write_text(
        "% Gerado por scripts/calibrar_manchas.py - não editar à mão\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in MACROS.items()), encoding="utf-8")
    saida = {"observado": {"todas": obs, "A": obsA, "B": obsB}, "validacao": val, "dentro_quatro": n4, "dentro_cauda": n2,
             "aceite": aceite, "posterior": {k: post[k] for k in ("lambda_m", "beta", "sigma_q", "comp_corr")},
             "meia_distancia_visitacao_m": meias, "p_beta_gt_1_5": p_gauss, "tempo_s": time.perf_counter() - t0}
    (RAIZ / "resultados" / "resultados_manchas.json").write_text(json.dumps(limpar(saida), indent=1, ensure_ascii=False),
                                                                 encoding="utf-8")
    print(f"dentro (4 ajustadas): manchas {n4[0]}/8, mistura {n4[1]}/8; cauda: manchas {n2[0]}/4, mistura {n2[1]}/4; "
          f"aceite {'atendido' if aceite else 'NÃO atendido'}; {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
