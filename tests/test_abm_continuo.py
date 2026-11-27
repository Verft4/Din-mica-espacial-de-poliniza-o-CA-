"""Testes do ABM contínuo (Limitação 1)."""

import numpy as np
import pytest

from polinizacao import ModeloABMContinuo, ModeloPolinizacao, Parametros, meia_distancia
from polinizacao.abm_continuo import PASSO_DIFUSAO
from polinizacao.modelo import VAZIO


def test_passo_de_difusao_casa_com_a_vizinhanca_de_moore():
    desloc = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    msd_moore = np.mean([a * a + b * b for a, b in desloc])
    assert msd_moore == pytest.approx(1.5)
    assert PASSO_DIFUSAO ** 2 == pytest.approx(msd_moore)


def test_mesma_paisagem_que_o_ca_para_a_mesma_semente():
    p = Parametros(agregacao_obstaculos=1.5, agregacao_flores=1.5)
    a, b = ModeloPolinizacao(p, seed=11), ModeloABMContinuo(p, seed=11)
    assert np.array_equal(a.paisagem, b.paisagem)


def test_reprodutivel_e_sementes_diferentes_divergem():
    p = Parametros()
    a = ModeloABMContinuo(p, seed=5).executar(60)
    b = ModeloABMContinuo(p, seed=5).executar(60)
    c = ModeloABMContinuo(p, seed=6).executar(60)
    assert a.hist_polinizadas == b.hist_polinizadas
    assert a.hist_polinizadas != c.hist_polinizadas


def test_invariantes_do_abm():
    m = ModeloABMContinuo(Parametros(), seed=2)
    anterior = 0
    for _ in range(150):
        m.passo()
        n = m.n_abelhas
        assert m.n_polinizadas >= anterior
        anterior = m.n_polinizadas
        assert np.all(m.ab_carga[:n] >= 0) and np.all(m.ab_carga[:n] <= m.p.cap_max + 1e-9)
        assert np.all((m.px[:n] > -0.5) & (m.px[:n] < m.p.linhas - 0.5))
        assert np.all((m.py[:n] > -0.5) & (m.py[:n] < m.p.colunas - 0.5))
        est = m.paisagem[m.ab_i[:n], m.ab_j[:n]]
        assert not np.any(est == VAZIO)                     # nunca dentro de obstáculo
    assert m.n_abelhas <= m.p.max_abelhas


def test_abelha_nunca_fica_em_obstaculo_com_muitos_obstaculos():
    m = ModeloABMContinuo(Parametros(frac_obstaculos=0.3, densidade_floral=0.0), seed=4)
    for _ in range(200):
        m.passo()
        n = m.n_abelhas
        est = m.paisagem[m.ab_i[:n], m.ab_j[:n]]
        if n:
            assert not np.any(est == VAZIO)


def test_difusao_livre_tem_msd_de_1_5_por_passo():
    """Sem flores nem obstáculos e sem bordas por perto, o deslocamento quadrático é ~1,5/passo."""
    p = Parametros(linhas=401, colunas=401, densidade_floral=0.0, frac_obstaculos=0.0,
                   max_abelhas=1, prob_geracao=1.0)
    m = ModeloABMContinuo(p, seed=0)
    m.n_abelhas = 1
    m.px[0], m.py[0] = m.centro
    vals = []
    for _ in range(4000):
        x0, y0 = m.px[0], m.py[0]
        m._mover(0)
        vals.append((m.px[0] - x0) ** 2 + (m.py[0] - y0) ** 2)
    assert np.mean(vals) == pytest.approx(1.5, rel=0.05)


def test_alpha_alto_concentra_visitas_em_flores():
    base = dict(densidade_floral=0.2)
    f = []
    for alpha in (1.0, 20.0):
        m = ModeloABMContinuo(Parametros(alpha_atracao=alpha, **base), seed=8).executar(120)
        f.append(sum(m.hist_visitas) / max(sum(m.hist_abelhas), 1))
    assert f[1] > 1.5 * f[0]


def test_extensoes_de_biologia_sao_recusadas():
    for kw in (dict(retorno_colmeia=True), dict(mortalidade=0.1), dict(dist_max_voo=10.0),
               dict(memoria_vies=1.0), dict(deposicao_acumulada=True), dict(toroidal=True)):
        with pytest.raises(ValueError):
            ModeloABMContinuo(Parametros(**kw), seed=1)
    with pytest.raises(ValueError):
        ModeloABMContinuo(Parametros(), seed=1, raio_percepcao=0)


def test_abm_aceita_paisagem_fragmentada():
    m = ModeloABMContinuo(Parametros(barreira_raio=8, barreira_aberturas=1), seed=3).executar(80)
    assert m.n_abelhas > 0


def test_meia_distancia():
    d = np.arange(0, 10, 1.0) + 0.5
    v = 100 * np.exp(-d / 3.0)
    assert meia_distancia(d, v) == pytest.approx(3.0 * np.log(2) + 0.5, abs=0.05)  # pico em d=0,5
    assert np.isnan(meia_distancia(d, np.full_like(d, 5.0)))
    assert np.isnan(meia_distancia([1.0], [1.0]))
    assert np.isnan(meia_distancia(d, np.zeros_like(d)))
