

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # nunca abre janela: só grava arquivos
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import scipy  # noqa: E402

from polinizacao import (  # noqa: E402
    ModeloPolinizacao, Parametros, __version__, executar_monte_carlo, sd_binomial_esperado,
    varrer_parametro,
)
from polinizacao.visualizacao import (  # noqa: E402
    desenhar_eficiencia, desenhar_paisagem, desenhar_serie_temporal,
)

SEED_BASE = 12345
SEED_SENSIBILIDADE = 2025
ZONAS = [(0, 6), (6, 12), (12, 20), (20, np.inf)]  # bordas alinhadas às faixas de largura 2


# --------------------------------------------------------------------- utilidades
def limpar(o):
    """Converte numpy/NaN para tipos JSON válidos (NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): limpar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpar(v) for v in o]
    if isinstance(o, np.ndarray):
        return limpar(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def pt(x, casas=1):
    """Número com vírgula decimal (padrão pt-BR) para o LaTeX."""
    return f"{x:.{casas}f}".replace(".", "{,}")


def pm(m, s, casas=1):
    return rf"{pt(m, casas)} $\pm$ {pt(s, casas)}"


def zonas(res):
    """Eficiência agregada (todas as réplicas) por zona de distância."""
    pol = np.nan_to_num(res.eficiencia / 100.0 * res.n_receptoras)
    tot = res.n_receptoras
    saida = []
    for lo, hi in ZONAS:
        m = (res.centros >= lo) & (res.centros < hi)
        t = tot[:, m].sum()
        saida.append({"de": lo, "ate": None if np.isinf(hi) else hi,
                      "eficiencia_pct": float(100 * pol[:, m].sum() / t) if t else float("nan"),
                      "receptoras": int(t)})
    return saida


def salvar(fig, nome):
    caminho = RAIZ / "figuras" / nome
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: figuras/{nome}")


# -------------------------------------------------------------------- experimentos
def experimento_base(n_sim, passos):
    print(f"[1/5] Cenário-base: {n_sim} simulações de {passos} passos")
    res = executar_monte_carlo(Parametros(), n_sim, passos, SEED_BASE)
    modelo = ModeloPolinizacao(Parametros(), seed=SEED_BASE).executar(passos)

    # Fig. 1 - instantâneo de uma simulação
    fig, ax = plt.subplots(figsize=(7.2, 6))
    desenhar_paisagem(modelo, ax, f"Uma simulação (semente {SEED_BASE}), passo {passos}")
    salvar(fig, "fig1_paisagem.png")

    # Fig. 2 - dinâmica temporal
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    desenhar_serie_temporal(res, ax)
    ax.set_title(f"Evolução temporal (N = {n_sim})")
    salvar(fig, "fig2_dinamica_temporal.png")

    # Fig. 3 - eficiência x distância, e variabilidade observada vs ruído binomial
    media, dp, semi, n = res.perfil_medio()
    esperado = sd_binomial_esperado(res.eficiencia, res.n_receptoras)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    desenhar_eficiencia(res, a)
    a.set_title("Eficiência de polinização por distância")
    ok = ~np.isnan(dp)
    b.plot(res.centros[ok], dp[ok], "o-", color="darkorange", label="Desvio observado entre simulações")
    b.plot(res.centros[ok], esperado[ok], "s--", color="navy",
           label="Esperado só por amostragem binomial")
    b.set_xlabel("Distância da colmeia (células)")
    b.set_ylabel("Desvio-padrão (pontos percentuais)")
    b.set_title("Variabilidade entre simulações")
    b.grid(alpha=0.3)
    b.legend(fontsize=8)
    salvar(fig, "fig3_eficiencia_distancia.png")

    resumo = res.resumo()
    # variabilidade nas zonas: observado / esperado
    razoes = []
    for lo, hi in ZONAS:
        m = (res.centros >= lo) & (res.centros < hi) & ok & (esperado > 0)
        razoes.append({"de": lo, "ate": None if np.isinf(hi) else hi,
                       "dp_observado": float(np.mean(dp[m])) if m.any() else float("nan"),
                       "dp_binomial": float(np.mean(esperado[m])) if m.any() else float("nan")})
    resumo["zonas"] = zonas(res)
    resumo["variabilidade_zonas"] = razoes
    resumo["receptoras_media"] = float(np.mean(res.n_receptoras.sum(axis=1)))
    resumo["abelhas_final_media"] = float(np.mean([modelo.n_abelhas]))
    return res, resumo


def experimento_sensibilidade(n_sim, passos):
    print(f"[2/5] Sensibilidade: {n_sim} simulações por valor")
    base = Parametros()
    grupos = {
        "densidade_floral": ("Densidade floral", [0.05, 0.10, 0.15, 0.25], base),
        "alpha_atracao": (r"Atração floral $\alpha$", [1.0, 4.0, 12.0, 30.0], base),
        "frac_obstaculos": ("Fração de obstáculos (independentes)", [0.0, 0.05, 0.15, 0.30], base),
        "agregacao_obstaculos": ("Agregação dos obstáculos (15% da grade)", [0.0, 1.5, 3.0],
                                 base.com(frac_obstaculos=0.15)),
    }
    saida, resultados, taxas = {}, {}, {}
    fig, eixos = plt.subplots(2, 2, figsize=(11.5, 8), sharey=True)
    for ax, (nome, (titulo, valores, b)) in zip(eixos.ravel(), grupos.items()):
        varredura = varrer_parametro(b, nome, valores, n_sim, passos, SEED_SENSIBILIDADE)
        linhas = []
        taxas[nome] = {v: 100 * r.taxa_final for v, r in varredura.items()}
        for v, r in varredura.items():
            media, _, _, _ = r.perfil_medio()
            ok = ~np.isnan(media)
            ax.plot(r.centros[ok], media[ok], "o-", ms=3.5, label=f"{v:g}")
            s = r.resumo()
            linhas.append({"valor": v, "taxa_final_media_pct": s["taxa_final_media_pct"],
                           "taxa_final_dp_pct": s["taxa_final_dp_pct"],
                           "raio95_media": s["raio95_media"], "raio95_dp": s["raio95_dp"],
                           "raio_eficaz_5pct": s["raio_eficaz_5pct"],
                           "polinizadas_final_media": s["polinizadas_final_media"]})
        resultados[nome] = linhas
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Distância da colmeia (células)")
        ax.grid(alpha=0.3)
        ax.legend(title="valor", fontsize=8)
    for ax in eixos[:, 0]:
        ax.set_ylabel("Receptoras polinizadas (%)")
    fig.tight_layout()
    salvar(fig, "fig4_sensibilidade.png")
    def welch(nome, a, b):
        x, y = taxas[nome][a], taxas[nome][b]
        t = stats.ttest_ind(x, y, equal_var=False)
        return {"a": a, "b": b, "dif_medias_pp": float(np.mean(y) - np.mean(x)),
                "p": float(t.pvalue)}
    saida["testes"] = {
        "obstaculos_0_vs_30": welch("frac_obstaculos", 0.0, 0.3),
        "agregacao_0_vs_3": welch("agregacao_obstaculos", 0.0, 3.0),
        "alpha_1_vs_12": welch("alpha_atracao", 1.0, 12.0),
        "densidade_005_vs_015": welch("densidade_floral", 0.05, 0.15),
    }
    saida["grupos"] = {k: v[0] for k, v in grupos.items()}
    saida["resultados"] = resultados
    return saida


def experimento_tempo(n_sim, passos_lista):
    print(f"[3/5] Efeito do tempo de simulação: passos = {passos_lista}")
    linhas = []
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for passos in passos_lista:
        r = executar_monte_carlo(Parametros(), n_sim, passos, SEED_SENSIBILIDADE)
        media, _, _, _ = r.perfil_medio()
        ok = ~np.isnan(media)
        a.plot(r.centros[ok], media[ok], "o-", ms=3.5, label=f"{passos} passos")
        s = r.resumo()
        linhas.append({"passos": passos, "taxa_final_media_pct": s["taxa_final_media_pct"],
                       "taxa_final_dp_pct": s["taxa_final_dp_pct"],
                       "raio95_media": s["raio95_media"], "raio95_dp": s["raio95_dp"],
                       "raio_eficaz_5pct": s["raio_eficaz_5pct"]})
    a.set_xlabel("Distância da colmeia (células)")
    a.set_ylabel("Receptoras polinizadas (%)")
    a.set_title("Perfil de eficiência em diferentes tempos")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    b.errorbar([l["passos"] for l in linhas], [l["raio95_media"] for l in linhas],
               yerr=[l["raio95_dp"] for l in linhas], fmt="o-", capsize=3, color="darkgreen")
    b.set_xscale("log")
    b.set_xlabel("Passos de simulação")
    b.set_ylabel("Raio que contém 95% das polinizadas (células)")
    b.set_title("Expansão da área de influência com o tempo")
    b.grid(alpha=0.3, which="both")
    fig.tight_layout()
    salvar(fig, "fig5_efeito_do_tempo.png")
    return linhas


def experimento_desempenho(repeticoes=3):
    print("[4/5] Desempenho computacional")
    def medir(linhas, colunas, abelhas, passos):
        tempos, trabalho = [], 0
        ModeloPolinizacao(Parametros(linhas=linhas, colunas=colunas, max_abelhas=abelhas,
                                     prob_geracao=1.0), seed=99).executar(20)  # aquecimento
        for k in range(repeticoes):
            p = Parametros(linhas=linhas, colunas=colunas, max_abelhas=abelhas, prob_geracao=1.0)
            t0 = time.perf_counter()
            m = ModeloPolinizacao(p, seed=k).executar(passos)
            tempos.append(time.perf_counter() - t0)
            trabalho = int(np.sum(m.hist_abelhas))
        t = float(np.median(tempos))
        return {"linhas": linhas, "colunas": colunas, "abelhas_max": abelhas, "passos": passos,
                "tempo_s": t, "tempo_por_passo_ms": 1e3 * t / passos,
                "passos_abelha": trabalho,
                "tempo_por_passo_abelha_us": 1e6 * t / max(trabalho, 1)}
    por_grade = [medir(n, n, 60, 150) for n in (30, 60, 120, 240)]
    por_abelhas = [medir(60, 60, a, 300) for a in (10, 60, 240, 960)]
    return {"por_tamanho_da_grade": por_grade, "por_numero_de_abelhas": por_abelhas,
            "ambiente": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "sistema": platform.platform(),
                         "polinizacao": __version__}}


# ------------------------------------------------------------------- saída em LaTeX
def escrever_latex(base, sens, tempo, desemp, n_sim, passos):
    z = base["zonas"]
    ae, ag = base["ajuste_exponencial"], base["ajuste_gaussiano"]
    v = base["variabilidade_zonas"]
    macros = {
        "NSim": str(n_sim), "Passos": str(passos), "SementeBase": str(SEED_BASE),
        "TaxaFinal": pt(base["taxa_final_media_pct"]), "TaxaFinalDP": pt(base["taxa_final_dp_pct"]),
        "PolFinal": pt(base["polinizadas_final_media"]), "PolFinalDP": pt(base["polinizadas_final_dp"]),
        "ReceptorasMedia": pt(base["receptoras_media"], 0),
        "RaioNoveCinco": pt(base["raio95_media"]), "RaioNoveCincoDP": pt(base["raio95_dp"]),
        "RaioEficaz": pt(base["raio_eficaz_5pct"], 0),
        "LamExp": pt(ae["lambda"], 3), "RsqExp": pt(ae["r2"], 2),
        "LamGau": pt(ag["lambda"], 3), "RsqGau": pt(ag["r2"], 2),
        "EscalaGau": pt(ag["escala"], 1), "EscalaExp": pt(ae["escala"], 1),
        "ZonaA": pt(z[0]["eficiencia_pct"]), "ZonaB": pt(z[1]["eficiencia_pct"]),
        "ZonaC": pt(z[2]["eficiencia_pct"]), "ZonaD": pt(z[3]["eficiencia_pct"]),
        "UsMin": pt(min(l["tempo_por_passo_abelha_us"] for l in desemp["por_tamanho_da_grade"] + desemp["por_numero_de_abelhas"]), 0),
        "UsMax": pt(max(l["tempo_por_passo_abelha_us"] for l in desemp["por_tamanho_da_grade"] + desemp["por_numero_de_abelhas"]), 0),
        "TempoSimMs": pt(1e3 * base["tempo_por_simulacao_s"], 0),
        "pObst": pt(sens["testes"]["obstaculos_0_vs_30"]["p"], 3),
        "difObst": pt(abs(sens["testes"]["obstaculos_0_vs_30"]["dif_medias_pp"]), 1),
        "pAgreg": pt(sens["testes"]["agregacao_0_vs_3"]["p"], 2),
        "pAlpha": pt(sens["testes"]["alpha_1_vs_12"]["p"], 4),
        "difAlpha": pt(abs(sens["testes"]["alpha_1_vs_12"]["dif_medias_pp"]), 1),
        "pDens": pt(sens["testes"]["densidade_005_vs_015"]["p"], 3),
        "DPobsA": pt(v[0]["dp_observado"], 0), "DPbinA": pt(v[0]["dp_binomial"], 0),
        "DPobsB": pt(v[1]["dp_observado"], 0), "DPbinB": pt(v[1]["dp_binomial"], 0),
        "DPobsC": pt(v[2]["dp_observado"], 0), "DPbinC": pt(v[2]["dp_binomial"], 0),
    }
    linhas = ["% GERADO AUTOMATICAMENTE por scripts/gerar_resultados.py -- não editar à mão."]
    linhas += [rf"\newcommand{{\{k}}}{{{val}}}" for k, val in macros.items()]
    (RAIZ / "artigo" / "numeros.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")

    # tabela de sensibilidade
    rot = {"densidade_floral": "Densidade floral", "alpha_atracao": r"Atração $\alpha$",
           "frac_obstaculos": "Obstáculos (fração)", "agregacao_obstaculos": r"Agregação ($\sigma$)"}
    t = [r"\begin{tabular}{llccc}", r"\toprule",
         r"Parâmetro & Valor & Receptoras polinizadas (\%) & Raio 95\% (células) & Raio 5\% (células)\\",
         r"\midrule"]
    for nome, linhas_g in sens["resultados"].items():
        for k, l in enumerate(linhas_g):
            rotulo = rot[nome] if k == 0 else ""
            r5 = "--" if l["raio_eficaz_5pct"] is None or np.isnan(l["raio_eficaz_5pct"]) \
                else pt(l["raio_eficaz_5pct"], 0)
            t.append(rf"{rotulo} & {pt(l['valor'], 2)} & {pm(l['taxa_final_media_pct'], l['taxa_final_dp_pct'])}"
                     rf" & {pm(l['raio95_media'], l['raio95_dp'])} & {r5}\\")
        t.append(r"\midrule")
    t[-1] = r"\bottomrule"
    t.append(r"\end{tabular}")
    (RAIZ / "artigo" / "tab_sensibilidade.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # tabela de tempo
    t = [r"\begin{tabular}{rccc}", r"\toprule",
         r"Passos & Receptoras polinizadas (\%) & Raio 95\% (células) & Raio 5\% (células)\\", r"\midrule"]
    for l in tempo:
        r5 = "--" if l["raio_eficaz_5pct"] is None or np.isnan(l["raio_eficaz_5pct"]) \
            else pt(l["raio_eficaz_5pct"], 0)
        t.append(rf"{l['passos']} & {pm(l['taxa_final_media_pct'], l['taxa_final_dp_pct'])}"
                 rf" & {pm(l['raio95_media'], l['raio95_dp'])} & {r5}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (RAIZ / "artigo" / "tab_tempo.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # tabela de desempenho
    t = [r"\begin{tabular}{rrrrr}", r"\toprule",
         r"Grade & Abelhas (máx.) & Passos & Tempo por passo (ms) & Tempo por passo-abelha ($\mu$s)\\",
         r"\midrule"]
    for l in desemp["por_tamanho_da_grade"] + desemp["por_numero_de_abelhas"]:
        t.append(rf"${l['linhas']}\times{l['colunas']}$ & {l['abelhas_max']} & {l['passos']}"
                 rf" & {pt(l['tempo_por_passo_ms'], 2)} & {pt(l['tempo_por_passo_abelha_us'], 0)}\\")
        if l is desemp["por_tamanho_da_grade"][-1]:
            t.append(r"\midrule")
    t += [r"\bottomrule", r"\end{tabular}"]
    (RAIZ / "artigo" / "tab_desempenho.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    print("  LaTeX: artigo/numeros.tex, tab_sensibilidade.tex, tab_tempo.tex, tab_desempenho.tex")


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="poucas réplicas (só para testar o pipeline)")
    args = ap.parse_args()
    n_base, n_sens, passos = (6, 4, 60) if args.rapido else (50, 30, 120)
    lista_tempo = [30, 60] if args.rapido else [60, 120, 240, 480]

    (RAIZ / "figuras").mkdir(exist_ok=True)
    (RAIZ / "resultados").mkdir(exist_ok=True)
    (RAIZ / "artigo").mkdir(exist_ok=True)
    t0 = time.perf_counter()

    res, base = experimento_base(n_base, passos)
    sens = experimento_sensibilidade(n_sens, passos)
    tempo = experimento_tempo(n_sens, lista_tempo)
    desemp = experimento_desempenho(1 if args.rapido else 3)

    print("[5/5] Gravando resultados")
    tudo = {"parametros_base": Parametros().__dict__, "n_simulacoes_base": n_base,
            "passos": passos, "seed_base": SEED_BASE, "seed_sensibilidade": SEED_SENSIBILIDADE,
            "base": base, "sensibilidade": sens, "efeito_do_tempo": tempo, "desempenho": desemp,
            "perfil_base": {"centros": res.centros, "media": res.perfil_medio()[0],
                            "dp": res.perfil_medio()[1], "ic95": res.perfil_medio()[2]}}
    (RAIZ / "resultados" / "resultados.json").write_text(
        json.dumps(limpar(tudo), indent=2, ensure_ascii=False), encoding="utf-8")
    escrever_latex(base, sens, tempo, desemp, n_base, passos)
    print(f"Concluído em {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
