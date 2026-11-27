#!/usr/bin/env python3
"""Experimentos das extensões do artigo (Limitações 1-4). Não altera os resultados da v1.

Uso:  python scripts/gerar_resultados_extensoes.py            (completo)
      python scripts/gerar_resultados_extensoes.py --rapido   (poucas réplicas, para teste)

Saídas:
  figuras/fig6..fig10*.png         figuras novas
  resultados/resultados_extensoes.json
  artigo/numeros_ext.tex           macros LaTeX com os números
  artigo/tab_abm.tex, tab_biologia.tex, tab_frag.tex, tab_calib.tex, tab_prcc.tex
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from polinizacao import (  # noqa: E402
    ModeloABMContinuo, ModeloPolinizacao, Parametros, ic95, media_dp,
)
from polinizacao import calibracao as cal  # noqa: E402
from polinizacao import lonsdorf as lo  # noqa: E402
from polinizacao.modelo import FLOR_DOADORA, FLOR_RECEPTORA, POLINIZADA  # noqa: E402
from polinizacao.visualizacao import desenhar_paisagem  # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")  # faixas sem flores
SEED = 777
MACROS: dict[str, str] = {}


# --------------------------------------------------------------------- utilidades
def limpar(o):
    if isinstance(o, dict):
        return {str(k): limpar(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpar(v) for v in o]
    if isinstance(o, np.ndarray):
        return limpar(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def pt(x, casas=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{casas}f}".replace(".", "{,}")


def pm(m, s, casas=1):
    return rf"{pt(m, casas)} $\pm$ {pt(s, casas)}"


def pval(p):
    if p is None or np.isnan(p):
        return "--"
    return r"\ensuremath{<\!0{,}001}" if p < 0.001 else pt(p, 3)


def pcomp(p):
    """p-valor com o operador (p<0,001 ou p=0,328), seguro dentro e fora do modo matemático."""
    if p is None or np.isnan(p):
        return "--"
    return r"\ensuremath{p<0{,}001}" if p < 0.001 else rf"\ensuremath{{p={pt(p, 3)}}}"


def mac(nome, valor):
    assert nome.isalpha(), nome
    MACROS[nome] = str(valor)


def salvar(fig, nome):
    fig.savefig(RAIZ / "figuras" / nome, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: figuras/{nome}")


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=False)
    return float(t.statistic), float(t.pvalue)


def rodar(params, n, passos, seed, cls=ModeloPolinizacao, **kw):
    """n réplicas independentes (sementes filhas); devolve os modelos finais."""
    return [cls(params, seed=s, **kw).executar(passos) for s in np.random.SeedSequence(seed).spawn(n)]


def taxas(ms):
    return np.array([100.0 * m.taxa_polinizacao for m in ms])


def r95s(ms):
    return np.array([m.raio_percentil(0.95) for m in ms])


def perfil(ms):
    c = ms[0].eficiencia_por_distancia()[0]
    return c, np.array([m.eficiencia_por_distancia()[1] for m in ms])


def eficiencia_fora(m, raio_min):
    """Eficiência (%) das receptoras a distância >= raio_min."""
    rec = np.isin(m.paisagem, (FLOR_RECEPTORA, POLINIZADA)) & (m._distancias >= raio_min)
    if rec.sum() == 0:
        return float("nan")
    return 100.0 * float((m.paisagem[rec] == POLINIZADA).sum() / rec.sum())


# ----------------------------------------------------------- Limitação 1: CA x ABM
def experimento_abm(n, passos, n_alpha):
    print(f"[1/4] Limitação 1: CA x ABM contínuo ({n} réplicas)")
    p = Parametros()
    ca = rodar(p, n, passos, SEED)
    ab = rodar(p, n, passos, SEED, cls=ModeloABMContinuo)       # mesmas sementes => mesmas paisagens
    c, pc = perfil(ca)
    _, pa = perfil(ab)
    mc, ma = np.nanmean(pc, 0), np.nanmean(pa, 0)
    ok = ~np.isnan(mc) & ~np.isnan(ma)
    rmse = float(np.sqrt(np.mean((mc[ok] - ma[ok]) ** 2)))
    tc, ta = taxas(ca), taxas(ab)
    rc, ra = r95s(ca), r95s(ab)
    t_taxa, p_taxa = welch(tc, ta)
    t_r95, p_r95 = welch(rc, ra)

    # Efeito de alpha em cada formalismo
    alfas = [1.0, 4.0, 12.0, 30.0]
    efeito = {}
    for a in alfas:
        pa_ = p.com(alpha_atracao=a)
        efeito[a] = {"CA": taxas(rodar(pa_, n_alpha, passos, SEED + 1)),
                     "ABM": taxas(rodar(pa_, n_alpha, passos, SEED + 1, cls=ModeloABMContinuo))}
    # Sensibilidade ao raio de percepção do ABM
    raios = [1.0, 1.5, 2.5]
    sens_raio = {r: taxas(rodar(p, n_alpha, passos, SEED + 2, cls=ModeloABMContinuo, raio_percepcao=r))
                 for r in raios}

    # Custo: microssegundos por abelha-passo (média de 5 execuções, após aquecimento)
    def custo(cls):
        cls(p, seed=1).executar(20)
        tempos, abelhas = 0.0, 0
        for s in range(5):
            t0 = time.perf_counter()
            m = cls(p, seed=100 + s).executar(passos)
            tempos += time.perf_counter() - t0
            abelhas += sum(m.hist_abelhas)
        return 1e6 * tempos / abelhas
    us_ca, us_abm = custo(ModeloPolinizacao), custo(ModeloABMContinuo)

    # Figura 6
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for nome, M, cor, mk in (("CA (Moore)", pc, "#1b6ca8", "o"), ("ABM contínuo", pa, "#d95f02", "s")):
        m_, s_ = np.nanmean(M, 0), ic95(M)[1]
        a1.plot(c, m_, marker=mk, ms=4, color=cor, label=nome)
        a1.fill_between(c, m_ - s_, m_ + s_, color=cor, alpha=0.18)
    a1.set_xlabel("Distância à colmeia (células)")
    a1.set_ylabel("Receptoras polinizadas (%)")
    a1.set_title(f"Perfil de eficiência (N = {n}, IC 95%)")
    a1.legend()
    x = np.arange(len(alfas))
    for k, (nome, cor) in enumerate((("CA", "#1b6ca8"), ("ABM", "#d95f02"))):
        mm = [np.mean(efeito[a][nome]) for a in alfas]
        ee = [ic95(efeito[a][nome])[1] for a in alfas]
        a2.bar(x + (k - 0.5) * 0.38, mm, 0.38, yerr=ee, color=cor, capsize=3,
               label="CA" if k == 0 else "ABM contínuo")
    a2.set_xticks(x, [f"{a:g}" for a in alfas])
    a2.set_xlabel(r"Atração floral $\alpha$")
    a2.set_ylabel("Receptoras polinizadas (%)")
    a2.set_title(f"Efeito de $\\alpha$ em cada formalismo (N = {n_alpha})")
    a2.legend()
    fig.tight_layout()
    salvar(fig, "fig6_ca_vs_abm.png")

    mac("AbmN", n)
    mac("AbmTaxaCA", pm(tc.mean(), tc.std(ddof=1)))
    mac("AbmTaxaABM", pm(ta.mean(), ta.std(ddof=1)))
    mac("AbmTaxaP", pcomp(p_taxa))
    mac("AbmRNoveCA", pm(np.nanmean(rc), np.nanstd(rc, ddof=1)))
    mac("AbmRNoveABM", pm(np.nanmean(ra), np.nanstd(ra, ddof=1)))
    mac("AbmRNoveP", pcomp(p_r95))
    mac("AbmRMSE", pt(rmse))
    mac("AbmUsCA", pt(us_ca))
    mac("AbmUsABM", pt(us_abm))
    mac("AbmRazaoCusto", pt(us_ca / us_abm))
    mac("AbmNAlpha", n_alpha)
    return {
        "taxa_CA": tc, "taxa_ABM": ta, "welch_taxa": (t_taxa, p_taxa), "r95_CA": rc, "r95_ABM": ra,
        "welch_r95": (t_r95, p_r95), "rmse_perfil_pp": rmse, "us_por_abelha_passo": {"CA": us_ca, "ABM": us_abm},
        "efeito_alpha": {str(a): {k: (float(v.mean()), float(v.std(ddof=1))) for k, v in d.items()}
                         for a, d in efeito.items()},
        "raio_percepcao": {str(r): (float(v.mean()), float(v.std(ddof=1))) for r, v in sens_raio.items()},
        "n": n, "n_alpha": n_alpha,
        "alfas": alfas, "raios": raios,
        "_efeito": efeito, "_sens_raio": sens_raio,
    }


# -------------------------------------------------------------- Limitação 3: biologia
def experimento_biologia(n, passos_lista, passos_abl):
    print(f"[2/4] Limitação 3: biologia ({n} réplicas)")
    base = Parametros()
    ret = Parametros(retorno_colmeia=True, t_viagem_max=60)
    saturacao = {"base": {}, "retorno": {}}
    for T in passos_lista:
        for nome, par in (("base", base), ("retorno", ret)):
            ms = rodar(par, n, T, SEED + 10)
            saturacao[nome][T] = {"r95": r95s(ms), "taxa": taxas(ms)}
    # Figura 7a: raio 95% vs tempo
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for nome, cor, rot in (("base", "#1b6ca8", "Modelo-base"), ("retorno", "#d95f02", "Com retorno à colmeia")):
        m_ = [np.nanmean(saturacao[nome][T]["r95"]) for T in passos_lista]
        e_ = [ic95(saturacao[nome][T]["r95"])[1] for T in passos_lista]
        a1.errorbar(passos_lista, m_, yerr=e_, marker="o", color=cor, capsize=3, label=rot)
    a1.set_xscale("log")
    a1.set_xticks(passos_lista, [str(t) for t in passos_lista])
    a1.set_xlabel("Passos simulados")
    a1.set_ylabel("Raio de 95% das polinizadas (células)")
    a1.set_title(f"A área de influência satura? (N = {n})")
    a1.legend()

    # Ablação em T = passos_abl
    itens = [
        ("Modelo-base", base),
        ("Retorno à colmeia", ret),
        ("Retorno + viagens de duração variável", ret.com(t_viagem_forma=4.0)),
        ("Distância máx. de voo = 20", base.com(dist_max_voo=20.0)),
        ("Mortalidade 0,01/passo", base.com(mortalidade=0.01)),
        ("Memória de mancha ($\\beta$ = 2)", base.com(memoria_vies=2.0)),
        ("Pesos distintos (doadora 24, receptora 6)", base.com(peso_doadora=24.0, peso_receptora=6.0)),
        ("Dose acumulada", base.com(deposicao_acumulada=True)),
        ("Tudo ligado", ret.com(t_viagem_forma=4.0, dist_max_voo=20.0, mortalidade=0.01, memoria_vies=2.0,
                                peso_doadora=24.0, peso_receptora=6.0, deposicao_acumulada=True)),
    ]
    abl = []
    ref = None
    for nome, par in itens:
        ms = rodar(par, n, passos_abl, SEED + 11)
        t, r = taxas(ms), r95s(ms)
        if ref is None:
            ref = (t, r)
        pt_ = welch(t, ref[0])[1] if len(abl) else float("nan")
        pr_ = welch(r[~np.isnan(r)], ref[1][~np.isnan(ref[1])])[1] if len(abl) else float("nan")
        abl.append({"nome": nome, "taxa": t, "r95": r, "p_taxa": pt_, "p_r95": pr_,
                    "mortes": float(np.mean([m.n_mortes for m in ms]))})
    y = np.arange(len(abl))[::-1]
    a2.barh(y, [x["taxa"].mean() for x in abl], xerr=[ic95(x["taxa"])[1] for x in abl], color="#4c9f70",
            capsize=3)
    a2.set_yticks(y, [x["nome"].replace("$\\beta$", "β") for x in abl], fontsize=8)
    a2.set_xlabel("Receptoras polinizadas (%)")
    a2.set_title(f"Efeito de cada extensão ({passos_abl} passos)")
    fig.tight_layout()
    salvar(fig, "fig7_biologia.png")

    # Viagens: duração e CV (em T = maior)
    Tmax = passos_lista[-1]
    ms_v = rodar(ret, n, Tmax, SEED + 12)
    dur = np.concatenate([m.duracoes_viagem for m in ms_v])
    cvs = np.array([np.std(m.duracoes_viagem) / np.mean(m.duracoes_viagem) for m in ms_v if m.duracoes_viagem])

    T0, T1 = passos_lista[1], passos_lista[-1]
    rb0, rb1 = np.nanmean(saturacao["base"][T0]["r95"]), np.nanmean(saturacao["base"][T1]["r95"])
    rr0, rr1 = np.nanmean(saturacao["retorno"][T0]["r95"]), np.nanmean(saturacao["retorno"][T1]["r95"])
    Tm = passos_lista[-2]
    mac("BioTmeio", Tm)
    mac("BioBaseRmeio", pt(np.nanmean(saturacao["base"][Tm]["r95"])))
    mac("BioRetRmeio", pt(np.nanmean(saturacao["retorno"][Tm]["r95"])))
    mac("BioAblBase", pt(abl[0]["taxa"].mean()))
    mac("BioAblRet", pt(abl[1]["taxa"].mean()))
    mac("BioAblTudo", pt(abl[-1]["taxa"].mean()))
    mac("BioAblDoseP", pcomp(abl[7]["p_taxa"]))
    mac("BioN", n)
    mac("BioTzero", T0)
    mac("BioTmax", T1)
    mac("BioBaseRzero", pt(rb0))
    mac("BioBaseRmax", pt(rb1))
    mac("BioRetRzero", pt(rr0))
    mac("BioRetRmax", pt(rr1))
    mac("BioTrajeto", pt(dur.mean(), 0))
    mac("BioCVTrajeto", pt(float(np.mean(cvs)), 2))
    mac("BioAblT", passos_abl)
    return {"saturacao": {k: {str(T): v for T, v in d.items()} for k, d in saturacao.items()},
            "ablacao": abl, "duracao_media": float(dur.mean()), "duracao_dp": float(dur.std(ddof=1)),
            "cv_por_replica": float(np.mean(cvs)), "n": n, "passos_lista": passos_lista, "passos_abl": passos_abl}


# ----------------------------------------------------------- Limitação 4: fragmentação
def experimento_fragmentacao(n, passos):
    print(f"[3/4] Limitação 4: fragmentação ({n} réplicas)")
    base = Parametros()
    raio, esp = 10.0, 2.0
    corredores = {}
    for k in (0, 1, 2, 4):
        par = base.com(barreira_raio=raio, barreira_espessura=esp, barreira_aberturas=k,
                       barreira_largura_abertura=3.0)
        ms = rodar(par, n, passos, SEED + 20)
        corredores[k] = {"fora": np.array([eficiencia_fora(m, raio + esp) for m in ms]), "taxa": taxas(ms),
                         "dentro_flores": float(np.mean([m.total_receptoras for m in ms]))}
    sem = rodar(base, n, passos, SEED + 20)
    ref_fora = np.array([eficiencia_fora(m, raio + esp) for m in sem])

    # Blocos x obstáculos independentes com a mesma fração nominal
    comp_blocos = {}
    for nb in (5, 10):
        frac = nb * 36 / 3600.0
        mb = rodar(base.com(frac_obstaculos=0.0, n_blocos=nb, tamanho_bloco=6), n, passos, SEED + 21)
        mi = rodar(base.com(frac_obstaculos=frac), n, passos, SEED + 21)
        comp_blocos[nb] = {"frac_nominal": frac, "blocos_taxa": taxas(mb), "indep_taxa": taxas(mi),
                           "blocos_r95": r95s(mb), "indep_r95": r95s(mi)}
    # Flores agrupadas
    flores = {}
    for sg in (0.0, 1.5, 3.0):
        ms = rodar(base.com(agregacao_flores=sg), n, passos, SEED + 22)
        flores[sg] = {"taxa": taxas(ms), "r95": r95s(ms)}

    # Figura 8: paisagens e resultados
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    amostras = [("Anel fechado", base.com(barreira_raio=raio, barreira_aberturas=0)),
                ("Anel com 2 corredores", base.com(barreira_raio=raio, barreira_aberturas=2)),
                ("Blocos (10 × 6×6)", base.com(frac_obstaculos=0.0, n_blocos=10)),
                ("Flores agrupadas (σ = 3)", base.com(agregacao_flores=3.0))]
    for k, (titulo, par) in enumerate(amostras):
        ax = fig.add_subplot(2, 4, k + 1)
        m = ModeloPolinizacao(par, seed=SEED + 30).executar(passos)
        desenhar_paisagem(m, ax, titulo)
    a5 = fig.add_subplot(2, 3, 4)
    ks = sorted(corredores)
    mm = [np.nanmean(corredores[k]["fora"]) for k in ks]
    ee = [ic95(corredores[k]["fora"])[1] for k in ks]
    a5.errorbar(ks, mm, yerr=ee, marker="o", color="#d95f02", capsize=3, label="Com anel")
    a5.axhline(np.nanmean(ref_fora), color="#1b6ca8", ls="--", label="Sem anel")
    a5.set_xlabel("Corredores no anel")
    a5.set_ylabel("Eficiência fora do anel (%)")
    a5.set_title(f"Efeito de sombra (N = {n})")
    a5.legend()
    a6 = fig.add_subplot(2, 3, 5)
    rot, vals, cores = [], [], []
    for nb, d in comp_blocos.items():
        rot += [f"{d['frac_nominal']*100:.0f}% indep.", f"{d['frac_nominal']*100:.0f}% blocos"]
        vals += [d["indep_taxa"], d["blocos_taxa"]]
        cores += ["#1b6ca8", "#d95f02"]
    a6.bar(range(len(vals)), [v.mean() for v in vals], yerr=[ic95(v)[1] for v in vals], color=cores, capsize=3)
    a6.set_xticks(range(len(vals)), rot, rotation=25, fontsize=8)
    a6.set_ylabel("Receptoras polinizadas (%)")
    a6.set_title("Obstáculos: blocos × independentes")
    a7 = fig.add_subplot(2, 3, 6)
    sg = sorted(flores)
    a7.bar(range(len(sg)), [flores[s]["taxa"].mean() for s in sg], yerr=[ic95(flores[s]["taxa"])[1] for s in sg],
           color="#4c9f70", capsize=3)
    a7.set_xticks(range(len(sg)), ["aleatórias" if s == 0 else f"σ = {s:g}" for s in sg])
    a7.set_ylabel("Receptoras polinizadas (%)")
    a7.set_title("Flores agrupadas (mesma densidade)")
    salvar(fig, "fig8_fragmentacao.png")

    mac("FragN", n)
    mac("FragPBlocoCinco", pcomp(welch(comp_blocos[5]["blocos_taxa"], comp_blocos[5]["indep_taxa"])[1]))
    mac("FragPBlocoDez", pcomp(welch(comp_blocos[10]["blocos_taxa"], comp_blocos[10]["indep_taxa"])[1]))
    for sg_, nome_ in ((0.0, "Aleat"), (1.5, "Fraca"), (3.0, "Forte")):
        mac(f"FragFlor{nome_}", pm(flores[sg_]["taxa"].mean(), flores[sg_]["taxa"].std(ddof=1)))
    mac("FragPFlorFraca", pcomp(welch(flores[1.5]["taxa"], flores[0.0]["taxa"])[1]))
    mac("FragPFlorForte", pcomp(welch(flores[3.0]["taxa"], flores[0.0]["taxa"])[1]))
    mac("FragRaio", f"{raio:g}")
    mac("FragSemAnel", pm(np.nanmean(ref_fora), np.nanstd(ref_fora, ddof=1)))
    mac("FragAnelFechado", pm(np.nanmean(corredores[0]["fora"]), np.nanstd(corredores[0]["fora"], ddof=1)))
    for k, nome in ((1, "Um"), (2, "Dois"), (4, "Quatro")):
        mac(f"FragAnel{nome}", pm(np.nanmean(corredores[k]["fora"]), np.nanstd(corredores[k]["fora"], ddof=1)))
    return {"corredores": {str(k): v for k, v in corredores.items()}, "sem_anel_fora": ref_fora,
            "blocos": {str(k): v for k, v in comp_blocos.items()},
            "flores": {str(k): v for k, v in flores.items()}, "n": n}


# ----------------------------------------------------- Limitação 2: calibração e validação
def experimento_calibracao(n, T, n_lhs):
    print(f"[4/4] Limitação 2: calibração, validação e identificabilidade ({n} réplicas, T = {T})")
    ret = Parametros(retorno_colmeia=True, t_viagem_max=60)
    ms = rodar(ret, n, T, SEED + 40)
    ajustes, vis_perfis, d_centros = [], [], None
    for m in ms:
        c, v, _ = m.visitacao_por_distancia()
        ajustes.append(cal.ajustar_perfil(c, v))
        vis_perfis.append(v)
        d_centros = c
    alpha_cel = np.array([a.alpha_exp for a in ajustes])
    r2e = np.array([a.r2_exp for a in ajustes])
    r2g = np.array([a.r2_gauss for a in ajustes])
    d50 = np.array([a.d50 for a in ajustes])
    dur_media = np.array([np.mean(m.duracoes_viagem) for m in ms])
    dur_cv = np.array([np.std(m.duracoes_viagem) / np.mean(m.duracoes_viagem) for m in ms])

    # 1) Escala espacial por espécie (Lonsdorf, tabela 2)
    espacial = {}
    for nome, (pt_, lo_, hi_) in cal.ALPHA_M.items():
        ponto, ic, faixa = cal.escala_com_incerteza(alpha_cel, pt_, lo_, hi_)
        espacial[nome] = {"alpha_m": (pt_, lo_, hi_), "m_por_celula": ponto, "ic": ic, "faixa": faixa,
                          "largura_grade_m": 60 * ponto}
    # 2) Escala temporal (A. mellifera)
    temporal = {}
    for tipo, (mu, sd) in cal.VIAGEM_MIN.items():
        s_passo = cal.escala_temporal(float(dur_media.mean()), mu * 60.0)
        temporal[tipo] = {"s_por_passo": s_passo, "cv_obs": cal.cv_observado(tipo)}
    m_cel_apis = espacial["Apis mellifera"]["m_por_celula"]
    passo_medio_cel = (4 * 1.0 + 4 * np.sqrt(2.0)) / 8.0
    for tipo in temporal:
        temporal[tipo]["velocidade_m_s"] = passo_medio_cel * m_cel_apis / temporal[tipo]["s_por_passo"]

    # 2b) Viagens de duração variável: orçamento ~ Gama com CV igual ao observado (isto é CALIBRAÇÃO do
    #     formato do orçamento, não validação: o CV observado passa a ser usado)
    gama = {}
    for tipo in ("néctar", "pólen"):
        k = 1.0 / cal.cv_observado(tipo) ** 2
        mg = rodar(ret.com(t_viagem_forma=k), n, 480, SEED + 41)
        cvg = np.array([np.std(m.duracoes_viagem) / np.mean(m.duracoes_viagem) for m in mg])
        gama[tipo] = {"k": k, "cv": cvg, "dur": np.array([np.mean(m.duracoes_viagem) for m in mg])}

    # 3) Validação fora da amostra
    d50_m = d50 * m_cel_apis
    d50_m_ic = ic95(d50_m)
    cv_ic = ic95(dur_cv)
    formas = (float(np.nanmean(r2g)), float(np.nanmean(r2e)))

    # 4) Comparação com o modelo tipo Lonsdorf, sobre as mesmas paisagens
    r2_lonsdorf, kappas = [], []
    for m, a in zip(ms, ajustes):
        flor = np.isin(m.paisagem, (FLOR_DOADORA, FLOR_RECEPTORA, POLINIZADA))
        inten = lo.intensidade_relativa(flor.astype(float), m._distancias, a.alpha_exp)
        c_, r_, _ = lo.perfil_por_distancia(inten, m._distancias, flor)
        c_, e_, _ = m.eficiencia_por_distancia()
        rec = np.isin(m.paisagem, (FLOR_RECEPTORA, POLINIZADA))
        _, r_rec, _ = lo.perfil_por_distancia(inten, m._distancias, rec)
        ok = ~np.isnan(e_) & ~np.isnan(r_rec)
        k = lo.ajustar_kappa(r_rec[ok], e_[ok])
        prev = lo.eficiencia_prevista(r_rec[ok], k)
        ss_res = float(np.sum((e_[ok] - prev) ** 2))
        ss_tot = float(np.sum((e_[ok] - e_[ok].mean()) ** 2))
        r2_lonsdorf.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
        kappas.append(k)
    r2_lonsdorf = np.array(r2_lonsdorf)

    # Figura 9: perfil de visitação com ajustes + Lonsdorf
    V = np.array(vis_perfis)
    vm = np.nanmean(V, 0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    ok = ~np.isnan(vm)
    pf = cal.ajustar_perfil(d_centros, vm)
    xs = np.linspace(0, d_centros[ok].max(), 200)
    from polinizacao.analise import ajustar_decaimento
    e_fit = ajustar_decaimento(d_centros, vm / np.nanmax(vm), "exponencial")
    g_fit = ajustar_decaimento(d_centros, vm / np.nanmax(vm), "gaussiana")
    a1.plot(d_centros[ok], vm[ok] / np.nanmax(vm), "ko", label="Simulação (média)")
    a1.plot(xs, e_fit["A"] * np.exp(-e_fit["lambda"] * xs), "--", color="#d95f02",
            label=f"Exponencial (R² = {e_fit['r2']:.2f})")
    a1.plot(xs, g_fit["A"] * np.exp(-(g_fit["lambda"] * xs) ** 2), "-", color="#1b6ca8",
            label=f"Gaussiana (R² = {g_fit['r2']:.2f})")
    a1.set_xlabel("Distância à colmeia (células)")
    a1.set_ylabel("Visitas por flor (normalizado)")
    a1.set_title(f"Perfil de visitação (retorno à colmeia, T = {T})")
    a1.legend()
    a2.hist(dur_cv, bins=12, color="#9a9a9a")
    for tipo, cor in (("néctar", "#d95f02"), ("pólen", "#1b6ca8")):
        a2.axvline(cal.cv_observado(tipo), color=cor, lw=2, label=f"Observado ({tipo}): {cal.cv_observado(tipo):.2f}")
    a2.set_xlabel("CV da duração da viagem (por réplica)")
    a2.set_ylabel("Réplicas")
    a2.set_title("Validação: variabilidade das viagens")
    a2.legend()
    fig.tight_layout()
    salvar(fig, "fig9_calibracao.png")

    # 5) Identificabilidade: hipercubo latino + PRCC (modelo-base, T = 240)
    limites = {"alpha_atracao": (1.0, 30.0), "densidade_floral": (0.05, 0.30), "taxa_deposicao": (0.2, 1.0),
               "limiar_polinizacao": (1.0, 10.0), "t_retencao": (5, 60), "estoque_max": (10.0, 80.0),
               "eficiencia_coleta": (2.0, 16.0), "cap_max": (10.0, 80.0), "t_recarga": (10, 150)}
    nomes, X = cal.hipercubo_latino(limites, n_lhs, seed=SEED)
    saidas = {"taxa": [], "r95": [], "alpha_cel": [], "d50": []}
    for k, row in enumerate(X):
        kw = {nm: (int(round(v)) if nm in ("t_retencao", "t_recarga") else float(v)) for nm, v in zip(nomes, row)}
        m = ModeloPolinizacao(Parametros(**kw), seed=SEED + 1000 + k).executar(240)
        c, v, _ = m.visitacao_por_distancia()
        pf_ = cal.ajustar_perfil(c, v)
        saidas["taxa"].append(100.0 * m.taxa_polinizacao)
        saidas["r95"].append(m.raio_percentil(0.95))
        saidas["alpha_cel"].append(pf_.alpha_exp)
        saidas["d50"].append(pf_.d50)
    pr = {}
    for s, y in saidas.items():
        y = np.array(y, float)
        bom = ~np.isnan(y)
        r_, p_ = cal.prcc(X[bom], y[bom])
        pr[s] = {"prcc": dict(zip(nomes, r_)), "p": dict(zip(nomes, p_)), "n": int(bom.sum())}

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w = 0.2
    for k, (s, rot, cor) in enumerate((("taxa", "Taxa de polinização", "#1b6ca8"), ("r95", "Raio 95%", "#d95f02"),
                                       ("alpha_cel", r"Escala espacial $\alpha_{cel}$", "#4c9f70"),
                                       ("d50", "Meia-distância", "#9a9a9a"))):
        ax.bar(np.arange(len(nomes)) + (k - 1.5) * w, [pr[s]["prcc"][nm] for nm in nomes], w, label=rot, color=cor)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(nomes)), [nm.replace("_", "\n") for nm in nomes], fontsize=7)
    ax.set_ylabel("PRCC")
    ax.set_title(f"Identificabilidade: hipercubo latino (N = {n_lhs}, T = 240)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    salvar(fig, "fig10_identificabilidade.png")

    ap = espacial["Apis mellifera"]
    mac("CalN", n)
    mac("CalT", T)
    mac("CalAlphaCel", pm(np.nanmean(alpha_cel), np.nanstd(alpha_cel, ddof=1), 2))
    mac("CalRdoisExp", pm(np.nanmean(r2e), np.nanstd(r2e, ddof=1), 2))
    mac("CalRdoisGauss", pm(np.nanmean(r2g), np.nanstd(r2g, ddof=1), 2))
    mac("CalDcinquenta", pm(np.nanmean(d50), np.nanstd(d50, ddof=1), 1))
    mac("CalMetrosCelula", pt(ap["m_por_celula"], 0))
    mac("CalMetrosCelulaLo", pt(ap["faixa"][0], 0))
    mac("CalMetrosCelulaHi", pt(ap["faixa"][1], 0))
    mac("CalGradeKm", pt(ap["largura_grade_m"] / 1000.0, 1))
    mac("CalPassoNectar", pt(temporal["néctar"]["s_por_passo"], 1))
    mac("CalPassoPolen", pt(temporal["pólen"]["s_por_passo"], 1))
    mac("CalVelNectar", pt(temporal["néctar"]["velocidade_m_s"], 0))
    mac("CalVelPolen", pt(temporal["pólen"]["velocidade_m_s"], 0))
    mac("CalDuracaoPassos", pt(float(dur_media.mean()), 0))
    mac("CalDmetros", pt(d50_m_ic[0], 0))
    mac("CalDmetrosSemi", pt(d50_m_ic[1], 0))
    mac("CalCV", pm(float(dur_cv.mean()), float(dur_cv.std(ddof=1)), 2))
    mac("CalKNectar", pt(gama["néctar"]["k"], 1))
    mac("CalKPolen", pt(gama["pólen"]["k"], 1))
    mac("CalCVGamaNectar", pm(gama["néctar"]["cv"].mean(), gama["néctar"]["cv"].std(ddof=1), 2))
    mac("CalCVGamaPolen", pm(gama["pólen"]["cv"].mean(), gama["pólen"]["cv"].std(ddof=1), 2))
    mac("CalVelRef", "7{,}5")
    mac("CalCVNectar", pt(cal.cv_observado("néctar"), 2))
    mac("CalCVPolen", pt(cal.cv_observado("pólen"), 2))
    mac("CalLonsdorfRdois", pm(np.nanmean(r2_lonsdorf), np.nanstd(r2_lonsdorf, ddof=1), 2))
    mac("LhsN", n_lhs)
    return {"alpha_cel": alpha_cel, "r2_exp": r2e, "r2_gauss": r2g, "d50": d50, "duracao_media_passos": dur_media,
            "cv": dur_cv, "espacial": espacial, "temporal": temporal, "d50_m": d50_m, "d50_m_ic": d50_m_ic,
            "cv_ic": cv_ic, "gama": gama, "r2_lonsdorf": r2_lonsdorf, "kappa": np.array(kappas), "prcc": pr,
            "limites": {k: v for k, v in limites.items()}, "n": n, "T": T, "n_lhs": n_lhs}


# ----------------------------------------------------------------------------- LaTeX
def escrever_latex(abm, bio, frag, calib):
    art = RAIZ / "artigo"
    (art / "numeros_ext.tex").write_text(
        "% Gerado por scripts/gerar_resultados_extensoes.py - não editar à mão\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in MACROS.items()), encoding="utf-8")

    # Tabela ABM
    t = [r"\begin{tabular}{lcc}", r"\toprule", r"Medida & CA (Moore) & ABM contínuo\\", r"\midrule",
         rf"Receptoras polinizadas (\%) & {pm(abm['taxa_CA'].mean(), abm['taxa_CA'].std(ddof=1))} & "
         rf"{pm(abm['taxa_ABM'].mean(), abm['taxa_ABM'].std(ddof=1))}\\",
         rf"Raio 95\% (células) & {pm(np.nanmean(abm['r95_CA']), np.nanstd(abm['r95_CA'], ddof=1))} & "
         rf"{pm(np.nanmean(abm['r95_ABM']), np.nanstd(abm['r95_ABM'], ddof=1))}\\",
         rf"Custo ($\mu$s por abelha-passo) & {pt(abm['us_por_abelha_passo']['CA'])} & "
         rf"{pt(abm['us_por_abelha_passo']['ABM'])}\\", r"\midrule"]
    for a in abm["alfas"]:
        d = abm["_efeito"][a]
        t.append(rf"Taxa (\%) com $\alpha$ = {pt(a, 0)} & {pm(d['CA'].mean(), d['CA'].std(ddof=1))} & "
                 rf"{pm(d['ABM'].mean(), d['ABM'].std(ddof=1))}\\")
    t += [r"\midrule"]
    for r_ in abm["raios"]:
        v = abm["_sens_raio"][r_]
        t.append(rf"Taxa (\%), raio de percepção {pt(r_, 1)} & -- & {pm(v.mean(), v.std(ddof=1))}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (art / "tab_abm.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # Tabela biologia
    t = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Configuração & Polinizadas (\%) & Raio 95\% (células) & $p$ (taxa vs. base)\\", r"\midrule"]
    for x in bio["ablacao"]:
        t.append(rf"{x['nome']} & {pm(x['taxa'].mean(), x['taxa'].std(ddof=1))} & "
                 rf"{pm(np.nanmean(x['r95']), np.nanstd(x['r95'], ddof=1))} & {pval(x['p_taxa'])}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (art / "tab_biologia.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # Tabela fragmentação
    t = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Cenário & Polinizadas (\%) & Raio 95\% (células) & $p$ (vs. controle)\\", r"\midrule"]
    for nb, d in frag["blocos"].items():
        fr = d["frac_nominal"] * 100
        t.append(rf"Obstáculos independentes ({fr:.0f}\%) & {pm(d['indep_taxa'].mean(), d['indep_taxa'].std(ddof=1))}"
                 rf" & {pm(np.nanmean(d['indep_r95']), np.nanstd(d['indep_r95'], ddof=1))} & --\\")
        t.append(rf"Blocos {nb} $\times$ 6$\times$6 ({fr:.0f}\%) & "
                 rf"{pm(d['blocos_taxa'].mean(), d['blocos_taxa'].std(ddof=1))} & "
                 rf"{pm(np.nanmean(d['blocos_r95']), np.nanstd(d['blocos_r95'], ddof=1))} & "
                 rf"{pval(welch(d['blocos_taxa'], d['indep_taxa'])[1])}\\")
    t.append(r"\midrule")
    ctrl = frag["flores"]["0.0"]
    for s, d in frag["flores"].items():
        nome = "Flores aleatórias" if float(s) == 0 else rf"Flores agrupadas ($\sigma$ = {pt(float(s), 1)})"
        p_ = "--" if float(s) == 0 else pval(welch(d["taxa"], ctrl["taxa"])[1])
        t.append(rf"{nome} & {pm(d['taxa'].mean(), d['taxa'].std(ddof=1))} & "
                 rf"{pm(np.nanmean(d['r95']), np.nanstd(d['r95'], ddof=1))} & {p_}\\")
    t.append(r"\midrule")
    ref = frag["sem_anel_fora"]
    t.append(rf"Eficiência fora do anel, sem anel (\%) & {pm(np.nanmean(ref), np.nanstd(ref, ddof=1))} & -- & --\\")
    for k, d in frag["corredores"].items():
        f = d["fora"]
        t.append(rf"Eficiência fora do anel, {k} corredor(es) (\%) & {pm(np.nanmean(f), np.nanstd(f, ddof=1))} & "
                 rf"-- & {pval(welch(f[~np.isnan(f)], ref[~np.isnan(ref)])[1])}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (art / "tab_frag.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # Tabela de calibração
    t = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Espécie & $\alpha$ (m) & m/célula (faixa) & Grade 60 células (km)\\", r"\midrule"]
    for nome, d in calib["espacial"].items():
        a = d["alpha_m"]
        intervalo = pt(a[0], 0) if a[1] == a[2] else rf"{pt(a[0], 0)} ({pt(a[1], 0)}--{pt(a[2], 0)})"
        t.append(rf"\textit{{{nome}}} & {intervalo} & {pt(d['m_por_celula'], 1)} "
                 rf"({pt(d['faixa'][0], 1)}--{pt(d['faixa'][1], 1)}) & {pt(d['largura_grade_m'] / 1000.0, 2)}\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (art / "tab_calib.tex").write_text("\n".join(t) + "\n", encoding="utf-8")

    # Tabela PRCC
    nomes = list(calib["limites"])
    t = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Parâmetro (faixa) & Taxa & Raio 95\% & $\alpha_{cel}$ & Meia-dist.\\", r"\midrule"]
    for nm in nomes:
        lo_, hi_ = calib["limites"][nm]
        faixa = f"{lo_:g}--{hi_:g}".replace(".", "{,}")
        cel = [rf"{nm.replace('_', ' ')} ({faixa})"]
        for s in ("taxa", "r95", "alpha_cel", "d50"):
            r_, p_ = calib["prcc"][s]["prcc"][nm], calib["prcc"][s]["p"][nm]
            txt = "--" if np.isnan(r_) else pt(r_, 2)
            if not np.isnan(p_) and p_ < 0.01:
                txt = rf"\textbf{{{txt}}}"
            cel.append(txt)
        t.append(" & ".join(cel) + r"\\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (art / "tab_prcc.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    if args.rapido:
        n, n_alpha, n_lhs, passos_lista, T_cal = 6, 5, 24, [60, 120, 240], 240
    else:
        n, n_alpha, n_lhs, passos_lista, T_cal = 50, 30, 200, [60, 120, 240, 480, 960], 960
    t0 = time.perf_counter()
    abm = experimento_abm(n, 120, n_alpha)
    bio = experimento_biologia(n if not args.rapido else 6, passos_lista, passos_lista[2])
    frag = experimento_fragmentacao(n, 120)
    calib = experimento_calibracao(30 if not args.rapido else 6, T_cal, n_lhs)
    escrever_latex(abm, bio, frag, calib)
    tudo = {"abm": {k: v for k, v in abm.items() if not k.startswith("_")}, "biologia": bio, "fragmentacao": frag,
            "calibracao": calib, "tempo_total_s": time.perf_counter() - t0}
    (RAIZ / "resultados" / "resultados_extensoes.json").write_text(
        json.dumps(limpar(tudo), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"pronto em {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
