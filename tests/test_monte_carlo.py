"""Testes do Monte Carlo e das varreduras de parâmetros."""

import numpy as np
import pytest

from polinizacao import Parametros, executar_monte_carlo, varrer_parametro


def test_fumaca_roda_e_tem_formas_corretas():
    r = executar_monte_carlo(n_simulacoes=3, passos=20, seed=1)
    assert r.n_sim == 3
    assert r.hist_polinizadas.shape == (3, 20)
    assert r.eficiencia.shape[0] == 3 and r.eficiencia.shape[1] == r.centros.size
    assert r.taxa_final.shape == (3,) and r.raio95.shape == (3,)
    assert r.tempo_total_s > 0


def test_reprodutivel_com_mesma_semente():
    a = executar_monte_carlo(n_simulacoes=4, passos=30, seed=9)
    b = executar_monte_carlo(n_simulacoes=4, passos=30, seed=9)
    c = executar_monte_carlo(n_simulacoes=4, passos=30, seed=10)
    assert np.array_equal(a.hist_polinizadas, b.hist_polinizadas)
    assert np.array_equal(a.eficiencia, b.eficiencia, equal_nan=True)
    assert not np.array_equal(a.hist_polinizadas, c.hist_polinizadas)


def test_replicas_sao_independentes():
    r = executar_monte_carlo(n_simulacoes=6, passos=60, seed=3)
    finais = r.hist_polinizadas[:, -1]
    assert len(set(finais.tolist())) > 1


def test_historico_acumulado_nao_decrescente():
    r = executar_monte_carlo(n_simulacoes=4, passos=50, seed=2)
    assert np.all(np.diff(r.hist_polinizadas, axis=1) >= 0)


def test_prefixo_igual_para_mais_passos():
    curto = executar_monte_carlo(n_simulacoes=3, passos=40, seed=5)
    longo = executar_monte_carlo(n_simulacoes=3, passos=80, seed=5)
    assert np.array_equal(curto.hist_polinizadas, longo.hist_polinizadas[:, :40])


def test_eficiencia_entre_0_e_100():
    r = executar_monte_carlo(n_simulacoes=5, passos=80, seed=6)
    v = r.eficiencia[~np.isnan(r.eficiencia)]
    assert np.all(v >= 0) and np.all(v <= 100)
    assert np.all((r.taxa_final >= 0) & (r.taxa_final <= 1))


def test_resumo_tem_campos_finitos():
    r = executar_monte_carlo(n_simulacoes=8, passos=100, seed=7)
    s = r.resumo()
    for chave in ("taxa_final_media_pct", "taxa_final_dp_pct", "raio95_media", "raio95_dp",
                  "tempo_por_simulacao_s"):
        assert np.isfinite(s[chave]), chave
    assert s["ajuste_exponencial"]["modelo"] == "exponencial"
    assert s["ajuste_gaussiano"]["modelo"] == "gaussiana"


def test_ic95_estreita_com_mais_simulacoes():
    pequeno = executar_monte_carlo(n_simulacoes=5, passos=80, seed=11)
    grande = executar_monte_carlo(n_simulacoes=40, passos=80, seed=11)
    semi_pequeno = pequeno.perfil_medio()[2]
    semi_grande = grande.perfil_medio()[2]
    assert np.nanmean(semi_grande) < np.nanmean(semi_pequeno)


def test_menos_abelhas_menos_polinizacao():
    poucas = executar_monte_carlo(Parametros(max_abelhas=5), n_simulacoes=8, passos=80, seed=1)
    muitas = executar_monte_carlo(Parametros(max_abelhas=60), n_simulacoes=8, passos=80, seed=1)
    assert poucas.taxa_final.mean() < muitas.taxa_final.mean()


def test_varrer_parametro():
    out = varrer_parametro(Parametros(), "alpha_atracao", [1.0, 12.0], n_simulacoes=2, passos=15)
    assert set(out) == {1.0, 12.0}
    assert out[1.0].params.alpha_atracao == 1.0 and out[12.0].params.alpha_atracao == 12.0


def test_varrer_parametro_nome_invalido():
    with pytest.raises(AttributeError):
        varrer_parametro(Parametros(), "nao_existe", [1], n_simulacoes=1, passos=5)


@pytest.mark.parametrize("n,p", [(0, 10), (3, 0)])
def test_argumentos_invalidos(n, p):
    with pytest.raises(ValueError):
        executar_monte_carlo(n_simulacoes=n, passos=p)
