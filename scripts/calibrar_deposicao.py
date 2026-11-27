#!/usr/bin/env python3
"""Passo 4: deposição por visita relativa a *Apis* vs tamanho do corpo (Földesi et al. 2020; Cecala et al. 2020).

Critério de aceite fixado ANTES do ajuste em ``docs/plano_limitacoes.md`` (Passo 4). O script o avalia e relata o
resultado, qualquer que seja. O limiar de polinização NÃO é calibrado (sem dados de pegamento de fruto/semente).

Uso:  python scripts/calibrar_deposicao.py [--rapido]
Saídas: figuras/fig13_deposicao.png · resultados/resultados_deposicao.json · artigo/numeros_deposicao.tex
        artigo/tab_deposicao.tex · artigo/tab_deposicao_ca.tex
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
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from polinizacao import ModeloPolinizacao, Parametros  # noqa: E402
from polinizacao import deposicao as dp  # noqa: E402

SEED = 4242
MACROS: dict[str, str] = {}


def pt(x, casas=2):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{casas}f}".replace(".", "{,}")


def faixa(lo, hi, casas=2):
    return rf"{pt(lo, casas)} a {pt(hi, casas)}"


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
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# --------------------------------------------------------------------------- análise da meta-regressão
def selecionar(obs, resposta, tamanho, ordens=("Hymenoptera",)):
    y = obs.lnr_delta if resposta == "lnRdelta" else obs.lnr
    v = obs.v_delta if resposta == "lnRdelta" else obs.v_lnr
    x = getattr(obs, tamanho)
    ok = np.isin(obs.ordem, ordens) & np.isfinite(x) & (x > 0) & np.isfinite(y) & np.isfinite(v) & (v > 0)
    return obs.estudo[ok], np.log(x[ok]), y[ok], v[ok], obs.taxon[ok]


def analisar(est, x, y, v, n_boot, seed):
    m2 = dp.ajustar_metarregressao(x, y, v, True)
    m3 = dp.ajustar_metarregressao(x, y, v, False)
    b2 = dp.bootstrap_estudos(est, x, y, v, True, n_boot, seed)
    b3 = dp.bootstrap_estudos(est, x, y, v, False, n_boot, seed + 1)
    lo = dp.validar_loso(est, x, y, v)
    w = lo["w_ref"]
    d01 = dp.diferenca_loso_ic(est, lo["M0"]["erro"], lo["M3"]["erro"], w)   # >0: M3 melhor que M0
    d13 = dp.diferenca_loso_ic(est, lo["M1"]["erro"], lo["M3"]["erro"], w)   # >0: M3 melhor que M1
    d12 = dp.diferenca_loso_ic(est, lo["M1"]["erro"], lo["M2"]["erro"], w)   # >0: M2 melhor que M1
    return {
        "n": int(len(y)), "n_estudos": int(len(np.unique(est))),
        "m2_intercepto": float(m2.beta[0]), "m2_inclinacao": float(m2.beta[1]), "m2_tau2": m2.tau2,
        "m2_inclinacao_ic": [float(np.quantile(b2[:, 1], 0.025)), float(np.quantile(b2[:, 1], 0.975))],
        "m3_inclinacao": float(m3.beta[0]), "m3_tau2": m3.tau2,
        "m3_inclinacao_ic": [float(np.quantile(b3[:, 0], 0.025)), float(np.quantile(b3[:, 0], 0.975))],
        "mse": {m: lo[m]["mse_ponderado"] for m in dp.MODELOS},
        "rmse": {m: lo[m]["rmse_simples"] for m in dp.MODELOS},
        "cobertura": {m: lo[m]["cobertura95"] for m in dp.MODELOS},
        "dif_m0_menos_m3": d01, "dif_m1_menos_m3": d13, "dif_m1_menos_m2": d12,
        "_m2": m2, "_m3": m3, "_loso": lo, "_b3": b3,
    }


def criterio(a):
    """Critério de aceite pré-fixado (Passo 4, item 5)."""
    c_a = a["dif_m0_menos_m3"][1] > 0 and a["dif_m1_menos_m3"][1] > 0  # M3 melhor que M0 e M1 com IC acima de 0
    c_a_ponto = a["mse"]["M3"] < a["mse"]["M0"] and a["mse"]["M3"] < a["mse"]["M1"]
    c_b = a["m3_inclinacao_ic"][0] > 0 or a["m3_inclinacao_ic"][1] < 0
    c_c = 0.90 <= a["cobertura"]["M3"] <= 0.99
    return {"a_ponto": bool(c_a_ponto), "a_ic": bool(c_a), "b": bool(c_b), "c": bool(c_c),
            "aceito": bool(c_a_ponto and c_b and c_c)}


def previsao_loso(obs, estudo_alvo, tamanho, resposta="lnRdelta"):
    """Prediz as observações de um estudo com o modelo ajustado SEM ele (M1 e M3), via ordens Hymenoptera."""
    est, x, y, v, tax = selecionar(obs, resposta, tamanho)
    te = est == estudo_alvo
    if not te.any():
        return []
    a1 = dp._ajuste_intercepto(y[~te], v[~te])
    a3 = dp.ajustar_metarregressao(x[~te], y[~te], v[~te], False)
    saida = []
    for xi, yi, vi, ti in zip(x[te], y[te], v[te], tax[te]):
        m3, var3 = dp.prever(a3, [xi])
        saida.append({"taxon": str(ti), "ln_razao_tamanho": float(xi), "observado": float(yi), "var_obs": float(vi),
                      "pred_m1": float(a1.beta[0]), "pred_m3": float(m3[0]),
                      "dp_pred_m3": float(np.sqrt(var3[0] + a3.tau2))})
    return saida


# --------------------------------------------------------------------------- Cecala et al. 2020
def analisar_cecala(caminho, aj_itd_m1, aj_itd_m3, tau2_m1, cov_m1, tau2_m3, cov_m3, seed):
    c = dp.carregar_cecala(caminho)
    sel = lambda g, t=None, gen=None: np.array(  # noqa: E731
        [(c.grupo[i] == g) and (t is None or c.tratamento[i] == t) and (gen is None or c.genero[i] == gen)
         and np.isfinite(c.graos[i]) for i in range(len(c.graos))])
    out = {}
    for nome, g, gen in (("nativas", "nativa", None), ("Halictus", "nativa", "Halictus")):
        for trat in ("control", "extracted"):
            a = c.graos[sel(g, trat, gen)]
            b = c.graos[sel("Apis", trat)]
            est, lo, hi = dp.log_razao_medias(a, b, seed=seed)
            out[f"{nome}_{trat}"] = {"n_nativa": int(len(a)), "n_apis": int(len(b)), "media_nativa": float(a.mean()),
                                     "media_apis": float(b.mean()), "ln_razao": est, "ic": [lo, hi]}
    # duração e relação grãos-duração
    dur = {}
    for g in ("Apis", "nativa"):
        m = np.array([(c.grupo[i] == g) and np.isfinite(c.graos[i]) and np.isfinite(c.duracao[i]) for i in range(len(c.graos))])
        rho = stats.spearmanr(c.duracao[m], c.graos[m])
        dur[g] = {"n": int(m.sum()), "mediana_duracao": float(np.median(c.duracao[m])), "rho_spearman": float(rho.statistic),
                  "p": float(rho.pvalue)}
    out["duracao"] = dur
    # predição por tamanho para Halictus: intervalo de ITD ancorado nos dois gêneros halictídeos da meta-análise
    pred = {}
    for s in (0.43, 0.67):
        m1, v1 = float(aj_itd_m1), cov_m1
        m3, v3 = dp.prever(aj_itd_m3, [np.log(s)])
        pred[str(s)] = {"m1": m1, "dp_m1": float(np.sqrt(tau2_m1 + v1)), "m3": float(m3[0]),
                        "dp_m3": float(np.sqrt(aj_itd_m3.tau2 + v3[0]))}
    out["predicao"] = pred
    return out


# --------------------------------------------------------------------------- experimento no autômato
def eficiencia(params, n, passos, seed):
    ms = [ModeloPolinizacao(params, seed=s).executar(passos) for s in np.random.SeedSequence(seed).spawn(n)]
    return np.array([100.0 * m.taxa_polinizacao for m in ms])


def experimento_ca(b_valores, tamanhos, limiares, n, passos):
    res = []
    base = Parametros()
    t0 = time.perf_counter()
    for acumula in (False, True):
        for lim in limiares:
            for b in b_valores:
                for s in (tamanhos if b != 0.0 else [1.0]):
                    p = base.com(limiar_polinizacao=lim, deposicao_acumulada=acumula,
                                 tamanho_relativo=s, expoente_deposicao=b)
                    e = eficiencia(p, n, passos, SEED)
                    res.append({"acumula": acumula, "limiar": lim, "b": b, "tamanho": s, "media": float(e.mean()),
                                "ep": float(e.std(ddof=1) / np.sqrt(len(e))), "taxa_efetiva": p.taxa_deposicao *
                                (s ** b) if p.taxa_deposicao * (s ** b) < 1 else 1.0})
    print(f"CA: {len(res)} cenários em {time.perf_counter() - t0:.0f} s")
    return res


# --------------------------------------------------------------------------- figura e LaTeX
def figura(prim, est, x, y, v, ca, b3, caminho):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    a = ax[0]
    tam = 8 + 60 / np.sqrt(v)
    a.scatter(np.exp(x), y, s=tam, c="#3b6ea5", alpha=0.55, edgecolor="none")
    xs = np.linspace(x.min(), x.max(), 100)
    m1 = prim["_m2"]
    a.axhline(0, color="#999", ls=":", lw=1, label="paridade com Apis (M0)")
    a.plot(np.exp(xs), np.full_like(xs, dp._ajuste_intercepto(y, v).beta[0]), color="#555", lw=1.4, label="intercepto (M1)")
    a.plot(np.exp(xs), m1.beta[0] + m1.beta[1] * xs, color="#c1502e", lw=1.6, label="reta (M2)")
    a.plot(np.exp(xs), prim["_m3"].beta[0] * xs, color="#2e8b57", lw=1.6, label="pela origem (M3)")
    a.set_xscale("log")
    a.set_xticks([0.4, 0.6, 0.8, 1.0, 1.5])
    a.set_xticklabels(["0,4", "0,6", "0,8", "1", "1,5"])
    a.minorticks_off()
    a.set_xlabel("tamanho relativo a Apis (ITD)")
    a.set_ylabel("ln(deposição por visita / Apis)")
    a.set_title("(a) Dados e ajustes (73 efeitos, 24 estudos)", fontsize=10)
    a.legend(fontsize=8, frameon=False)
    a = ax[1]
    ms = [prim["mse"][m] for m in dp.MODELOS]
    cores = ["#999", "#555", "#c1502e", "#2e8b57"]
    a.bar(range(4), ms, color=cores)
    a.set_xticks(range(4))
    a.set_xticklabels(["M0\nparidade", "M1\nintercepto", "M2\nreta", "M3\npela origem"], fontsize=9)
    a.set_ylabel("erro quadrático ponderado (estudo omitido)")
    a.set_title("(b) Validação deixando um estudo inteiro de fora", fontsize=10)
    for i, m in enumerate(ms):
        a.text(i, m, f"{m:.3f}".replace(".", ","), ha="center", va="bottom", fontsize=8)
    a = ax[2]
    lims = (4.0,)
    for acumula, estilo in ((False, "-"), (True, "--")):
        for b, cor in zip(sorted({r["b"] for r in ca}), ("#555", "#2e8b57", "#c1502e")):
            pts = [r for r in ca if r["acumula"] == acumula and r["limiar"] in lims and r["b"] == b]
            if not pts:
                continue
            pts.sort(key=lambda r: r["tamanho"])
            if b == 0.0:
                a.axhline(pts[0]["media"], color=cor, ls=estilo, lw=1.3)
            else:
                a.errorbar([r["tamanho"] for r in pts], [r["media"] for r in pts], yerr=[1.96 * r["ep"] for r in pts],
                           color=cor, ls=estilo, marker="o", ms=4, lw=1.3, capsize=2)
    a.plot([], [], color="#555", label="b = 0")
    a.plot([], [], color="#2e8b57", label=f"b = {b3:.2f}".replace(".", ",") + " (M3)")
    a.plot([], [], color="#c1502e", label="b = 1")
    a.plot([], [], color="k", ls="-", label="visita única")
    a.plot([], [], color="k", ls="--", label="dose acumulada")
    a.legend(fontsize=8, frameon=False, ncol=1)
    a.set_xlabel("tamanho relativo a Apis")
    a.set_ylabel("receptoras polinizadas (%)")
    a.set_title("(c) Autômato: eficiência vs tamanho (limiar = 4)", fontsize=10)
    for z in ax:
        z.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(caminho, dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    n_boot = 400 if args.rapido else 4000
    n_ca, passos = (6, 60) if args.rapido else (40, 120)
    t0 = time.perf_counter()
    obs = dp.carregar_foldesi(RAIZ / "dados" / "foldesi2020_metadata.csv")
    _, info = dp.subconjunto_abelhas(obs, "itd")

    # --- análise primária (pré-fixada)
    est, x, y, v, _ = selecionar(obs, "lnRdelta", "itd")
    prim = analisar(est, x, y, v, n_boot, SEED)
    crit = criterio(prim)

    # --- sensibilidade (todas declaradas no plano)
    sens = {}
    for nome, resposta, tam, ordens in (
        ("lnRdelta_BL_himenopteros", "lnRdelta", "bl", ("Hymenoptera",)),
        ("lnR_ITD_himenopteros", "lnR", "itd", ("Hymenoptera",)),
        ("lnR_BL_himenopteros", "lnR", "bl", ("Hymenoptera",)),
        ("lnRdelta_BL_todos_insetos", "lnRdelta", "bl", ("Hymenoptera", "Diptera")),
    ):
        e, xx, yy, vv, _t = selecionar(obs, resposta, tam, ordens)
        a = analisar(e, xx, yy, vv, max(n_boot // 2, 200), SEED + 100)
        a["criterio"] = criterio(a)
        sens[nome] = {k: val for k, val in a.items() if not k.startswith("_")}

    # --- abelhas sem ferrão (verificação ad hoc, estudo omitido)
    sem_ferrao = {
        "Osorio-Beristain_et_al_1997 (Trigona nigra, ITD)": previsao_loso(obs, "Osorio-Beristain_et_al_1997", "itd"),
        "Witter_et_al_2015 (Plebeia e Tetragonisca, BL)": previsao_loso(obs, "Witter_et_al_2015", "bl"),
    }

    # --- Cecala
    m1_itd = dp._ajuste_intercepto(y, v)
    cec = analisar_cecala(RAIZ / "dados" / "cecala2020_visitas.csv", m1_itd.beta[0], prim["_m3"], m1_itd.tau2,
                          float(m1_itd.cov[0, 0]), prim["_m3"].tau2, float(prim["_m3"].cov[0, 0]), SEED)

    # --- autômato: expoentes 0, b_M3 (ponto) e 1 (proporcional)
    b3 = prim["m3_inclinacao"]
    ca = experimento_ca([0.0, round(b3, 2), 1.0], [0.4, 0.7, 1.0, 1.5], (2.0, 4.0, 8.0), n_ca, passos)

    # --- saídas
    (RAIZ / "figuras").mkdir(exist_ok=True)
    figura(prim, est, x, y, v, ca, b3, RAIZ / "figuras" / "fig13_deposicao.png")
    escrever_latex(info, prim, crit, sens, cec, ca, b3, sem_ferrao)
    out = {"info": info, "primaria": {k: val for k, val in prim.items() if not k.startswith("_")}, "criterio": crit,
           "sensibilidade": sens, "sem_ferrao": sem_ferrao, "cecala": cec, "automato": ca,
           "tempo_s": time.perf_counter() - t0}
    out["primaria"].pop("cobertura_por_modelo", None)
    (RAIZ / "resultados" / "resultados_deposicao.json").write_text(
        json.dumps(limpar(out), indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(limpar({"info": info, "criterio": crit,
                             "b3": prim["m3_inclinacao"], "b3_ic": prim["m3_inclinacao_ic"],
                             "b2": prim["m2_inclinacao"], "b2_ic": prim["m2_inclinacao_ic"],
                             "mse": prim["mse"], "cob": prim["cobertura"]}), indent=1))
    print(f"pronto em {time.perf_counter() - t0:.0f} s")


def escrever_latex(info, prim, crit, sens, cec, ca, b3, sem_ferrao):
    mac("DepN", info["n_usado"])
    mac("DepNEstudos", info["n_estudos"])
    mac("DepNHim", info["n_himenoptera"])
    mac("DepNTotal", info["n_total"])
    mac("DepExclAusente", info["excluidos_tamanho_ausente"])
    mac("DepExclNulo", info["excluidos_tamanho_nulo"])
    mac("DepBTres", pt(b3))
    mac("DepBTresIC", faixa(*prim["m3_inclinacao_ic"]))
    mac("DepBDois", pt(prim["m2_inclinacao"]))
    mac("DepBDoisIC", faixa(*prim["m2_inclinacao_ic"]))
    mac("DepInterc", pt(prim["m2_intercepto"]))
    mac("DepTauDois", pt(prim["m2_tau2"]))
    for m, nm in zip(dp.MODELOS, ("Zero", "Um", "Dois", "Tres")):
        mac(f"DepMSE{nm}", pt(prim["mse"][m], 3))
        mac(f"DepCob{nm}", pt(100 * prim["cobertura"][m], 0))
    d = prim["dif_m0_menos_m3"]
    mac("DepDifZeroTres", rf"{pt(d[0], 3)} (IC95\% {faixa(d[1], d[2], 3)})")
    d = prim["dif_m1_menos_m3"]
    mac("DepDifUmTres", rf"{pt(d[0], 3)} (IC95\% {faixa(d[1], d[2], 3)})")
    mac("DepCritA", "atendido" if crit["a_ponto"] else "não atendido")
    mac("DepCritB", "atendido" if crit["b"] else "não atendido")
    mac("DepCritC", "atendido" if crit["c"] else "não atendido")
    mac("DepVeredito", "aceito" if crit["aceito"] else "rejeitado")
    # Cecala
    for chave, nome in (("nativas_control", "CecNatCtl"), ("Halictus_control", "CecHalCtl"),
                        ("nativas_extracted", "CecNatExt"), ("Halictus_extracted", "CecHalExt")):
        r = cec[chave]
        mac(nome, rf"{pt(r['ln_razao'])} (IC95\% {faixa(*r['ic'])})")
        mac(nome + "N", f"{r['n_nativa']}")
        mac(nome + "NApis", f"{r['n_apis']}")
        mac(nome + "Razao", pt(float(np.exp(r["ln_razao"])), 2))
    mac("CecMediaApisCtl", pt(cec["nativas_control"]["media_apis"], 0))
    mac("CecMediaHalCtl", pt(cec["Halictus_control"]["media_nativa"], 0))
    mac("CecPredMUmBaixo", pt(cec["predicao"]["0.43"]["m1"]))
    mac("CecPredMTresBaixo", pt(cec["predicao"]["0.43"]["m3"]))
    mac("CecPredMTresAlto", pt(cec["predicao"]["0.67"]["m3"]))
    mac("CecDurApis", pt(cec["duracao"]["Apis"]["mediana_duracao"], 0))
    mac("CecDurNat", pt(cec["duracao"]["nativa"]["mediana_duracao"], 0))
    mac("CecRhoApis", pt(cec["duracao"]["Apis"]["rho_spearman"]))
    mac("CecRhoNat", pt(cec["duracao"]["nativa"]["rho_spearman"]))
    # tabela de sensibilidade
    linhas = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
              r"Análise & $n$ & $b$ (M3) & IC95\% de $b$ & EQ M0 & EQ M1 & EQ M3 \\", r"\midrule"]
    linhas.append(rf"lnR$\delta$, ITD, abelhas (primária) & {info['n_usado']} & {pt(b3)} & {faixa(*prim['m3_inclinacao_ic'])} & "
                  rf"{pt(prim['mse']['M0'], 3)} & {pt(prim['mse']['M1'], 3)} & {pt(prim['mse']['M3'], 3)} \\")
    nomes = {"lnRdelta_BL_himenopteros": r"lnR$\delta$, BL, abelhas", "lnR_ITD_himenopteros": "lnR, ITD, abelhas",
             "lnR_BL_himenopteros": "lnR, BL, abelhas", "lnRdelta_BL_todos_insetos": r"lnR$\delta$, BL, abelhas e moscas"}
    for k, a in sens.items():
        linhas.append(rf"{nomes[k]} & {a['n']} & {pt(a['m3_inclinacao'])} & {faixa(*a['m3_inclinacao_ic'])} & "
                      rf"{pt(a['mse']['M0'], 3)} & {pt(a['mse']['M1'], 3)} & {pt(a['mse']['M3'], 3)} \\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    (RAIZ / "artigo" / "tab_deposicao.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    mac("DepSensNAceitos", sum(1 for a in sens.values() if a["criterio"]["aceito"]))
    mac("DepSensN", len(sens))
    # tabela do autômato (limiar x dose x expoente) em tamanho 0,4 relativo ao Apis
    linhas = [r"\begin{tabular}{llrrrr}", r"\toprule",
              r"Regime & Limiar & $b=0$ & $b=" + pt(b3) + r"$ ($s=0{,}4$) & $b=1$ ($s=0{,}4$) & $b=1$ ($s=1{,}5$) \\", r"\midrule"]

    def val(acu, lim, b, s):
        for r in ca:
            if r["acumula"] == acu and r["limiar"] == lim and r["b"] == b and r["tamanho"] == s:
                return pt(r["media"], 1)
        return "--"
    bb = round(b3, 2)
    for acu, nome in ((False, "visita única"), (True, "dose acumulada")):
        for lim in (2.0, 4.0, 8.0):
            linhas.append(rf"{nome} & {pt(lim, 0)} & {val(acu, lim, 0.0, 1.0)} & {val(acu, lim, bb, 0.4)} & "
                          rf"{val(acu, lim, 1.0, 0.4)} & {val(acu, lim, 1.0, 1.5)} \\")
    linhas += [r"\bottomrule", r"\end{tabular}"]
    (RAIZ / "artigo" / "tab_deposicao_ca.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    # sem ferrão
    (RAIZ / "artigo" / "numeros_deposicao.tex").write_text(
        "% gerado por scripts/calibrar_deposicao.py\n" +
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in MACROS.items()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
