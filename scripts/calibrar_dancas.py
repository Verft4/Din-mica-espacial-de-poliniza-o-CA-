#!/usr/bin/env python3
"""Calibra o núcleo de distância e a escala espacial com danças decodificadas (Dryad 10.5061/dryad.c2fqz618f).

O CSV público é por SESSÃO (183 sessões, 20 colmeias), não por dança. Ver ``polinizacao/dancas.py``.

Uso:  python scripts/calibrar_dancas.py [--csv dados/dancas_sessoes_dryad.csv] [--rapido]
Saídas: figuras/fig11_dancas.png · resultados/resultados_dancas.json · artigo/numeros_dancas.tex · artigo/tab_dancas.tex
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from polinizacao import ModeloPolinizacao, Parametros  # noqa: E402
from polinizacao import dancas as dc  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
SEED = 4242
MACROS: dict[str, str] = {}
NOMES = ("mediana_das_medianas", "mediana_das_medias", "razao_media_mediana", "dispersao_relativa")
ROTULOS = {"mediana_das_medianas": "Mediana das medianas (m)", "mediana_das_medias": "Mediana das médias (m)",
           "razao_media_mediana": "Razão média/mediana", "dispersao_relativa": "Dispersão relativa (IQR/mediana)"}


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
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def predicao(post, n_sessoes, n_dancas, seed, maximo=200):
    """Estatísticas de sessões simuladas com os parâmetros aceitos (distribuição preditiva a posteriori)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(post["p"]), size=min(maximo, len(post["p"])), replace=False)
    out = []
    for k in idx:
        md, me = dc.simular_sessoes(rng, post["p"][k], post["media_exp"][k], post["sigma_ray"][k], post["tau"][k],
                                    n_sessoes, n_dancas)
        e = dc.estatisticas(md, me)
        out.append([e[n] for n in NOMES])
    return np.array(out)


def viagens_do_modelo(par, n, T, seed):
    """Distância máxima de cada viagem (células), agrupada por réplica."""
    ms = [ModeloPolinizacao(par, seed=s).executar(T) for s in np.random.SeedSequence(seed).spawn(n)]
    return [np.array(m.dist_max_viagens, float) for m in ms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(RAIZ / "dados" / "dancas_sessoes_dryad.csv"))
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    NS, n_rep, T = (1500, 6, 240) if args.rapido else (12000, 30, 960)
    ND = 15                                         # danças por sessão: 2827 / 183 ≈ 15,4 (suposição)

    S = dc.carregar_sessoes(args.csv)
    ncol = len(set(S.colmeia))
    obs = dc.estatisticas(S.mediana_m, S.media_m)
    ma, mb = dc.dobras_por_colmeia(S, seed=SEED)
    SA, SB = S.subconjunto(ma), S.subconjunto(mb)
    obsA, obsB = dc.estatisticas(SA.mediana_m, SA.media_m), dc.estatisticas(SB.mediana_m, SB.media_m)
    print(f"{len(S)} sessões, {ncol} colmeias; dobra A = {len(set(SA.colmeia))} colmeias, B = {len(set(SB.colmeia))}")

    # ---------- 1) ABC do núcleo (todas as sessões e por dobra) ----------
    print("ABC (todas as sessões) ...")
    post = dc.abc_mistura(obs, n_dancas=ND, n_sorteios=NS, seed=SEED + 1)
    viag = dc.estatisticas_da_viagem(post, seed=SEED + 2, n=20000)
    print("ABC (dobras A e B) ...")
    postA = dc.abc_mistura(obsA, n_dancas=ND, n_sorteios=NS, seed=SEED + 3)
    postB = dc.abc_mistura(obsB, n_dancas=ND, n_sorteios=NS, seed=SEED + 4)
    sens = {}
    for nd in (8, 30):
        print(f"ABC (sensibilidade n_dancas = {nd}) ...")
        po = dc.abc_mistura(obs, n_dancas=nd, n_sorteios=NS, seed=SEED + 5)
        sens[nd] = {"p": po["p"], "viagem": dc.estatisticas_da_viagem(po, seed=SEED + 6, n=20000)}

    # ---------- 2) validação entre colmeias: calibra em A, prevê B (e vice-versa) ----------
    cobertura = {}
    for rot, po, alvo, nS in (("A->B", postA, obsB, len(SB)), ("B->A", postB, obsA, len(SA))):
        pr = predicao(po, nS, ND, SEED + 7)
        cobertura[rot] = {n: {"obs": alvo[n], "pred_mediana": float(np.median(pr[:, i])),
                              "pred_5": float(np.percentile(pr[:, i], 5)), "pred_95": float(np.percentile(pr[:, i], 95)),
                              "dentro": bool(np.percentile(pr[:, i], 5) <= alvo[n] <= np.percentile(pr[:, i], 95))}
                          for i, n in enumerate(NOMES)}

    # ---------- 3) escala espacial do modelo ----------
    print(f"Modelo: {n_rep} réplicas, T = {T} ...")
    configs = {"orçamento fixo": Parametros(retorno_colmeia=True, t_viagem_max=60),
               "orçamento Gama (k = 3,6)": Parametros(retorno_colmeia=True, t_viagem_max=60, t_viagem_forma=3.6)}
    modelo = {}
    for nome, par in configs.items():
        reps = viagens_do_modelo(par, n_rep, T, SEED + 8)
        d = np.concatenate(reps)
        md_s, me_s = dc.prever_sessoes_do_modelo(reps, 1.0, ND, 4000, seed=SEED + 9)
        e_cel = dc.estatisticas(md_s, me_s)
        res = {"n_viagens": int(d.size), "mediana_cel": float(np.median(d)), "media_cel": float(d.mean()),
               "cv": float(d.std() / d.mean()), "sessoes_cel": e_cel}
        for rot, obs_cal, obs_val in (("A->B", obsA, obsB), ("B->A", obsB, obsA)):
            m = obs_cal["mediana_das_medianas"] / e_cel["mediana_das_medianas"]        # metros por célula
            md, me = dc.prever_sessoes_do_modelo(reps, m, ND, 4000, seed=SEED + 10)
            pred = dc.estatisticas(md, me)
            res[rot] = {"m_por_celula": float(m), "previsto": pred, "observado": obs_val,
                        "erro_rel": {n: float((pred[n] - obs_val[n]) / obs_val[n]) for n in NOMES},
                        "erro_rel_base": {n: float((obs_cal[n] - obs_val[n]) / obs_val[n]) for n in NOMES}}
        res["m_por_celula_todas"] = float(obs["mediana_das_medianas"] / e_cel["mediana_das_medianas"])
        modelo[nome] = res

    # sensibilidade ao tempo simulado (a escala em metros depende de T?)
    sens_T = {}
    for TT in ((120, 240) if args.rapido else (240, 480, 960)):
        reps_t = viagens_do_modelo(configs["orçamento fixo"], max(4, n_rep // 2), TT, SEED + 11)
        md_s, _ = dc.prever_sessoes_do_modelo(reps_t, 1.0, ND, 4000, seed=SEED + 9)
        sens_T[TT] = float(obs["mediana_das_medianas"] / np.median(md_s))

    # ---------- figura ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.sort(S.mediana_m)
    a1.step(x, np.arange(1, x.size + 1) / x.size, color="k", lw=2, label="Observado (183 sessões)")
    rng = np.random.default_rng(SEED)
    for k in rng.choice(len(post["p"]), size=min(60, len(post["p"])), replace=False):
        md, _ = dc.simular_sessoes(rng, post["p"][k], post["media_exp"][k], post["sigma_ray"][k], post["tau"][k], 183, ND)
        xs = np.sort(md)
        a1.step(xs, np.arange(1, xs.size + 1) / xs.size, color="#1b6ca8", alpha=0.12, lw=1,
                label="Núcleo ajustado (ABC)" if k == 0 else None)
    a1.set_xscale("log")
    a1.set_xlabel("Mediana da distância de dança por sessão (m)")
    a1.set_ylabel("Fração acumulada")
    a1.set_title("Núcleo ajustado reproduz as sessões")
    a1.legend(handles=[plt.Line2D([], [], color="k", lw=2), plt.Line2D([], [], color="#1b6ca8", alpha=0.6)],
              labels=["Observado (183 sessões)", "Núcleo ajustado (ABC)"], loc="lower right")
    qs = np.linspace(0.02, 0.98, 49)
    rng2 = np.random.default_rng(SEED + 12)
    kk = int(np.argmin(post["dist"]))
    ref = dc.amostrar_mistura(rng2, post["p"][kk], post["media_exp"][kk], post["sigma_ray"][kk], 100000)
    refq = np.quantile(ref, qs)
    lim = float(refq.max())
    a2.plot([0, lim], [0, lim], color="gray", ls=":")
    for (nome, res), cor in zip(modelo.items(), ("#d95f02", "#4c9f70")):
        d = np.concatenate(viagens_do_modelo(configs[nome], max(4, n_rep // 2), T, SEED + 13)) * res["m_por_celula_todas"]
        a2.plot(refq, np.quantile(d, qs), "o-", ms=3, color=cor, label=nome)
    a2.set_xlabel("Quantis da distância da viagem — núcleo ajustado (m)")
    a2.set_ylabel("Quantis do modelo (m)")
    a2.set_title("Forma da distribuição: modelo × núcleo")
    a2.legend()
    fig.tight_layout()
    fig.savefig(RAIZ / "figuras" / "fig11_dancas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- macros e tabela ----------
    mac("DanSessoes", len(S)); mac("DanColmeias", ncol); mac("DanND", ND); mac("DanDraws", NS)
    mac("DanMedMed", pt(obs["mediana_das_medianas"])); mac("DanMedMea", pt(obs["mediana_das_medias"]))
    mac("DanRazao", pt(obs["razao_media_mediana"], 2)); mac("DanDisp", pt(obs["dispersao_relativa"], 2))
    mac("DanPBat", ic(post["p"], 2)); mac("DanViagemMed", ic(viag["mediana"])); mac("DanViagemMedia", ic(viag["media"]))
    mac("DanViagemCV", ic(viag["cv"], 2))
    mac("DanPBatOito", ic(sens[8]["p"], 2)); mac("DanPBatTrinta", ic(sens[30]["p"], 2))
    mac("DanViagemMedOito", ic(sens[8]["viagem"]["mediana"])); mac("DanViagemMedTrinta", ic(sens[30]["viagem"]["mediana"]))
    nfora = sum(1 for r in cobertura.values() for v in r.values() if not v["dentro"])
    mac("DanForaIC", nfora); mac("DanTotalIC", 2 * len(NOMES))
    for rot, nm in (("A->B", "AB"), ("B->A", "BA")):
        for n, c in zip(NOMES, ("Med", "Mea", "Raz", "Disp")):
            mac(f"DanCob{nm}{c}", "dentro" if cobertura[rot][n]["dentro"] else "fora")
    for nome, nm in (("orçamento fixo", "Fixo"), ("orçamento Gama (k = 3,6)", "Gama")):
        r = modelo[nome]
        mac(f"DanMcel{nm}", pt(r["m_por_celula_todas"])); mac(f"DanMcelA{nm}", pt(r["A->B"]["m_por_celula"]))
        mac(f"DanMcelB{nm}", pt(r["B->A"]["m_por_celula"]))
        mac(f"DanModCV{nm}", pt(r["cv"], 2)); mac(f"DanModMed{nm}", pt(r["mediana_cel"], 1))
        mac(f"DanModRazao{nm}", pt(r["sessoes_cel"]["razao_media_mediana"], 2))
        mac(f"DanModDisp{nm}", pt(r["sessoes_cel"]["dispersao_relativa"], 2))
        for rot, tag in (("A->B", "AB"), ("B->A", "BA")):
            for n, c in zip(NOMES, ("Med", "Mea", "Raz", "Disp")):
                mac(f"DanErr{nm}{tag}{c}", pt(100 * r[rot]["erro_rel"][n]))
    for TT, v in sens_T.items():
        mac(f"DanMcelT{'abc'[list(sens_T).index(TT)]}", pt(v)); mac(f"DanTT{'abc'[list(sens_T).index(TT)]}", TT)

    linhas = [r"\begin{tabular}{lcccc}", r"\toprule",
              r"Estatística de sessão & Observado (B) & Previsto, núcleo ajustado em A (90\%) & Modelo, fixo & Modelo, Gama\\",
              r"\midrule"]
    for n in NOMES:
        c = cobertura["A->B"][n]
        casas = 0 if "(m)" in ROTULOS[n] else 2
        linhas.append(rf"{ROTULOS[n]} & {pt(c['obs'], casas)} & {pt(c['pred_5'], casas)}--{pt(c['pred_95'], casas)} & "
                      rf"{pt(modelo['orçamento fixo']['A->B']['previsto'][n], casas)} & "
                      rf"{pt(modelo['orçamento Gama (k = 3,6)']['A->B']['previsto'][n], casas)}\\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    (RAIZ / "artigo" / "tab_dancas.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    (RAIZ / "artigo" / "numeros_dancas.tex").write_text(
        "% Gerado por scripts/calibrar_dancas.py - não editar à mão\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in MACROS.items()), encoding="utf-8")
    saida = {"observado": {"todas": obs, "A": obsA, "B": obsB}, "n_sessoes": len(S), "n_colmeias": ncol,
             "abc": {"p": post["p"], "media_exp": post["media_exp"], "sigma_ray": post["sigma_ray"], "tau": post["tau"]},
             "viagem": viag, "cobertura": cobertura, "modelo": modelo, "escala_vs_T": sens_T,
             "sensibilidade_n_dancas": {str(k): {"p": v["p"], "viagem": v["viagem"]} for k, v in sens.items()}}
    (RAIZ / "resultados" / "resultados_dancas.json").write_text(json.dumps(limpar(saida), indent=1, ensure_ascii=False),
                                                                encoding="utf-8")
    print("pronto")


if __name__ == "__main__":
    main()
