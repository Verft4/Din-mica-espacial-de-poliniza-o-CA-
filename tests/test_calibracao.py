"""Testes do modelo tipo Lonsdorf e da calibração/identificabilidade (Limitação 2)."""

import numpy as np
import pytest

from polinizacao import ModeloPolinizacao, Parametros
from polinizacao import calibracao as cal
from polinizacao import lonsdorf as lo


# ------------------------------------------------------------------------------ Lonsdorf
def test_nucleo_exponencial():
    assert lo.nucleo(0.0, 5.0) == 1.0
    assert lo.nucleo(5.0, 5.0) == pytest.approx(np.exp(-1))
    with pytest.raises(ValueError):
        lo.nucleo(1.0, 0.0)


def test_visitacao_relativa_soma_um_e_decresce_com_a_distancia():
    d = np.arange(1, 21, dtype=float)
    v = lo.visitacao_relativa(np.ones(20), d, alpha=4.0)
    assert v.sum() == pytest.approx(1.0)
    assert np.all(np.diff(v) < 0)
    razao = v[1:] / v[:-1]
    assert np.allclose(razao, np.exp(-1 / 4.0))


def test_visitacao_respeita_atratividade_e_zeros():
    d = np.array([1.0, 1.0, 1.0])
    v = lo.visitacao_relativa(np.array([2.0, 1.0, 0.0]), d, 3.0)
    assert v[2] == 0 and v[0] == pytest.approx(2 * v[1])
    with pytest.raises(ValueError):
        lo.visitacao_relativa(np.zeros(3), d, 3.0)


def test_intensidade_relativa_tem_media_um_nas_flores():
    d = np.linspace(1, 30, 40)
    f = (np.arange(40) % 2 == 0).astype(float)
    r = lo.intensidade_relativa(f, d, 6.0)
    assert r[f > 0].mean() == pytest.approx(1.0)


def test_perfil_por_distancia():
    d = np.array([0.5, 1.0, 2.5, 3.0])
    c, m, n = lo.perfil_por_distancia(np.array([1.0, 3.0, 5.0, 7.0]), d, np.ones(4, bool), 2.0)
    assert list(n) == [2, 2] and list(m) == [2.0, 6.0] and list(c) == [1.0, 3.0]
    _, m2, n2 = lo.perfil_por_distancia(np.ones(4), d, np.array([1, 1, 0, 0], bool), 2.0)
    assert np.isnan(m2[1]) and n2[1] == 0
    with pytest.raises(ValueError):
        lo.perfil_por_distancia(d, d, np.ones(4, bool), 0)


def test_meia_distancia_exponencial_e_ajuste_de_kappa():
    assert lo.meia_distancia_exponencial(10.0) == pytest.approx(10 * np.log(2))
    x = np.linspace(0.05, 3, 25)
    y = lo.eficiencia_prevista(x, 0.8)
    assert lo.ajustar_kappa(x, y) == pytest.approx(0.8, rel=1e-3)
    assert np.isnan(lo.ajustar_kappa([1.0], [1.0]))


# ------------------------------------------------------------------------------ calibração
def test_ancoras_da_literatura():
    assert cal.ALPHA_M["Apis mellifera"] == (663.0, 562.0, 776.0)
    assert cal.ALPHA_M["Tetragonisca angustula"][0] == 22.0
    assert cal.MEIA_DISTANCIA_RICKETTS_M == 600.0
    assert cal.cv_observado("néctar") == pytest.approx(8.8 / 16.7)
    assert cal.cv_observado("pólen") == pytest.approx(11.8 / 24.5)


def test_escalas():
    assert cal.escala_espacial(3.0, 600.0) == 200.0
    assert cal.escala_temporal(100.0, 1000.0) == 10.0
    for f, args in ((cal.escala_espacial, (0.0, 1.0)), (cal.escala_temporal, (0.0, 1.0))):
        with pytest.raises(ValueError):
            f(*args)


def test_ajustar_perfil_reconhece_exponencial_e_gaussiana():
    d = np.arange(1, 30, 2.0)
    pe = cal.ajustar_perfil(d, 0.05 * np.exp(-d / 6.0))
    assert pe.alpha_exp == pytest.approx(6.0, rel=0.02) and pe.r2_exp > 0.999
    pg = cal.ajustar_perfil(d, 0.05 * np.exp(-(d / 8.0) ** 2))
    assert pg.r2_gauss > pg.r2_exp and pg.r2_gauss > 0.999
    assert np.isnan(cal.ajustar_perfil(d, np.zeros_like(d)).alpha_exp)


def test_escala_com_incerteza():
    ponto, ic, faixa = cal.escala_com_incerteza([3.0, 3.2, 2.8, 3.1, 2.9], 663.0, 562.0, 776.0)
    assert ponto == pytest.approx(663.0 / 3.0)
    assert ic[0] < ponto < ic[1]
    assert faixa[0] < ic[0] and faixa[1] > ic[1]


def test_hipercubo_latino_estratificado():
    nomes, X = cal.hipercubo_latino({"a": (0.0, 1.0), "b": (10.0, 20.0)}, n=50, seed=1)
    assert nomes == ["a", "b"] and X.shape == (50, 2)
    for j, (lo_, hi) in enumerate([(0, 1), (10, 20)]):
        estrato = np.floor((X[:, j] - lo_) / (hi - lo_) * 50).astype(int)
        assert sorted(estrato) == list(range(50))      # um ponto por estrato
    assert np.array_equal(X, cal.hipercubo_latino({"a": (0.0, 1.0), "b": (10.0, 20.0)}, 50, seed=1)[1])


def test_prcc_recupera_sinais_e_irrelevantes():
    rng = np.random.default_rng(0)
    X = rng.random((400, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 0.05 * rng.normal(size=400)      # colunas 2 e 3 irrelevantes
    r, p = cal.prcc(X, y)
    assert r[0] > 0.9 and r[1] < -0.9
    assert abs(r[2]) < 0.15 and abs(r[3]) < 0.15
    assert p[0] < 1e-6 and p[2] > 0.01 or abs(r[2]) < 0.15


def test_prcc_monotonico_nao_linear_e_constante():
    rng = np.random.default_rng(1)
    X = rng.random((300, 2))
    y = np.exp(5 * X[:, 0])                                         # monotônico: PRCC ~ 1
    assert cal.prcc(X, y)[0][0] > 0.99
    X[:, 1] = 1.0
    assert np.isnan(cal.prcc(X, y)[0][1])


def test_visitacao_do_ca_tem_perfil_decrescente():
    m = ModeloPolinizacao(Parametros(), seed=3).executar(120)
    c, v, n = m.visitacao_por_distancia()
    ok = ~np.isnan(v)
    assert v[ok][0] > v[ok][-1]
    pf = cal.ajustar_perfil(c, v)
    assert pf.r2_gauss > 0.5 and np.isfinite(pf.d50)
