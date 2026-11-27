"""Testes do modelo em duas escalas (célula = mancha)."""

import numpy as np
import pytest

from polinizacao import dancas as dc
from polinizacao import manchas as mc


@pytest.fixture(scope="module")
def pais():
    return mc.Paisagem(lado_celulas=101, lado_m=100.0, n_campos=6, seed=1)


def test_parametros_validos_e_invalidos():
    mc.ParametrosManchas()
    for kw in (dict(lambda_m=0), dict(beta=-1), dict(sigma_q=-0.1), dict(comp_corr=3.0)):
        with pytest.raises(ValueError):
            mc.ParametrosManchas(**kw)


def test_paisagem_validacao_e_geometria(pais):
    with pytest.raises(ValueError):
        mc.Paisagem(lado_celulas=100)
    with pytest.raises(ValueError):
        mc.Paisagem(lado_celulas=101, lado_m=0)
    assert pais.centro == (50, 50) and pais.d_m[50, 50] == 0.0
    assert pais.d_m[50, 51] == pytest.approx(100.0) and pais.d_m[51, 51] == pytest.approx(100 * np.sqrt(2))


def test_banco_tem_variancia_unitaria_e_correlacao_crescente(pais):
    def corr(z):
        return np.mean([np.corrcoef(c[:, :-1].ravel(), c[:, 1:].ravel())[0, 1] for c in z])
    assert all(pais.banco[e].std() == pytest.approx(1.0, rel=0.02) for e in mc.COMPRIMENTOS)
    cs = [corr(pais.banco[e]) for e in mc.COMPRIMENTOS]
    assert all(b > a for a, b in zip(cs, cs[1:]))


def test_pesos_decrescem_com_a_distancia_sem_heterogeneidade(pais):
    par = mc.ParametrosManchas(lambda_m=500, beta=1.5, sigma_q=0.0)
    w = pais.pesos(np.zeros_like(pais.d_m), par)
    assert w[50, 50] == 1.0 and w[50, 60] < w[50, 55] < w[50, 51]
    assert np.allclose(w, w.T)                                    # simetria radial


def test_qualidade_tem_media_um(pais):
    rng = np.random.default_rng(0)
    z = np.concatenate([pais.campo(rng, 2.0).ravel() for _ in range(5)])
    for s in (0.5, 1.5):
        assert np.exp(s * z - s * s / 2).mean() == pytest.approx(1.0, rel=0.1)


def test_gaussiana_gera_distancias_tipo_rayleigh(pais):
    par = mc.ParametrosManchas(lambda_m=700, beta=2.0, sigma_q=0.0, comp_corr=1.0)
    d, _ = pais.sortear_viagens(np.random.default_rng(1), pais.pesos(np.zeros_like(pais.d_m), par), 100000)
    assert d.mean() / np.median(d) == pytest.approx(1.064, abs=0.03)       # Rayleigh
    par1 = mc.ParametrosManchas(lambda_m=700, beta=1.0, sigma_q=0.0, comp_corr=1.0)
    d1, _ = pais.sortear_viagens(np.random.default_rng(1), pais.pesos(np.zeros_like(pais.d_m), par1), 100000)
    assert d1.mean() / np.median(d1) == pytest.approx(2 / 1.678, abs=0.05)  # gama(2): média 2L, mediana 1,678L
    assert d1.mean() > d.mean()


def test_sortear_viagens_limites_e_reprodutibilidade(pais):
    par = mc.ParametrosManchas()
    w = pais.pesos(pais.campo(np.random.default_rng(3), par.comp_corr), par)
    a, ia = pais.sortear_viagens(np.random.default_rng(5), w, 500)
    b, ib = pais.sortear_viagens(np.random.default_rng(5), w, 500)
    assert np.array_equal(a, b) and np.array_equal(ia, ib)
    assert a.min() >= 0 and a.max() <= pais.d_m.max() + 100 * np.sqrt(2)
    with pytest.raises(ValueError):
        pais.sortear_viagens(np.random.default_rng(0), np.zeros_like(w), 3)


def test_heterogeneidade_aumenta_a_dispersao_entre_sessoes(pais):
    kw = dict(lambda_m=800, beta=2.0, comp_corr=4.0)
    rng = np.random.default_rng(2)
    md0, me0 = pais.sessoes(rng, mc.ParametrosManchas(sigma_q=0.0, **kw), 150, 15)
    md1, me1 = pais.sessoes(rng, mc.ParametrosManchas(sigma_q=2.0, **kw), 150, 15)
    d0, d1 = dc.estatisticas(md0, me0)["dispersao_relativa"], dc.estatisticas(md1, me1)["dispersao_relativa"]
    assert d1 > 1.5 * d0


def test_visitas_por_mancha_conservam_o_total(pais):
    v = mc.Paisagem.visitas_por_mancha(pais, np.random.default_rng(4), mc.ParametrosManchas(), 300)
    assert v.shape == pais.d_m.shape and v.sum() == 300 and v.min() >= 0


def test_meia_distancia_da_visitacao():
    assert mc.meia_distancia_visitacao(mc.ParametrosManchas(lambda_m=1000, beta=1.0)) == pytest.approx(693.1, rel=1e-3)
    assert mc.meia_distancia_visitacao(mc.ParametrosManchas(lambda_m=1000, beta=2.0)) == pytest.approx(832.6, rel=1e-3)


def test_abc_manchas_recupera_o_sentido_de_lambda(pais):
    rng = np.random.default_rng(7)
    def obs(lam):
        md, me = pais.sessoes(rng, mc.ParametrosManchas(lam, 2.0, 1.0, 2.0), 120, 15)
        return dc.estatisticas(md, me)
    a = mc.abc_manchas(obs(300), pais, n_sorteios=600, n_sessoes=30, seed=1)
    b = mc.abc_manchas(obs(1500), pais, n_sorteios=600, n_sessoes=30, seed=1)
    assert np.median(b["lambda_m"]) > 1.5 * np.median(a["lambda_m"])
    assert mc.parametros_aceitos(a, 0).lambda_m == a["lambda_m"][0]
