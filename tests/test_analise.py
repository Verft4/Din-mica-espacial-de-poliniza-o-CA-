"""Testes das funções estatísticas."""

import numpy as np
import pytest

from polinizacao import ajustar_decaimento, ajustar_exponencial, ic95, media_dp, raio_eficaz


def test_media_dp_ignora_nan():
    x = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, 2.0]])
    m, dp, n = media_dp(x)
    assert m[0] == pytest.approx(3.0) and dp[0] == pytest.approx(2.0) and n[0] == 3
    assert m[1] == pytest.approx(2.0) and np.isnan(dp[1]) and n[1] == 1


def test_coluna_toda_nan_nao_gera_aviso():
    import warnings
    x = np.array([[np.nan], [np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m, dp, n = media_dp(x)
    assert np.isnan(m[0]) and n[0] == 0


def test_ic95_valor_conhecido():
    media, semi, n = ic95(np.array([[1.0], [2.0], [3.0]]))
    # t(0.975, 2) = 4.3027; dp = 1; semi = 4.3027 / sqrt(3)
    assert media[0] == pytest.approx(2.0)
    assert semi[0] == pytest.approx(4.302653 / np.sqrt(3), rel=1e-4)
    assert n[0] == 3


def test_ic95_n_menor_que_2_e_nan():
    _, semi, _ = ic95(np.array([[1.0]]))
    assert np.isnan(semi[0])


def test_ajuste_exponencial_recupera_parametros():
    d = np.arange(1, 40, 2.0)
    e = 80.0 * np.exp(-0.15 * d)
    aj = ajustar_exponencial(d, e)
    assert aj["A"] == pytest.approx(80.0, rel=1e-3)
    assert aj["lambda"] == pytest.approx(0.15, rel=1e-3)
    assert aj["r2"] > 0.9999
    assert aj["escala"] == pytest.approx(1 / 0.15, rel=1e-3)


def test_ajuste_gaussiano_recupera_parametros():
    d = np.arange(1, 40, 2.0)
    e = 95.0 * np.exp(-((0.07 * d) ** 2))
    aj = ajustar_decaimento(d, e, "gaussiana")
    assert aj["A"] == pytest.approx(95.0, rel=1e-3)
    assert aj["lambda"] == pytest.approx(0.07, rel=1e-3)
    assert aj["r2"] > 0.9999


def test_gaussiana_ajusta_melhor_dados_gaussianos():
    d = np.arange(1, 40, 2.0)
    e = 100.0 * np.exp(-((0.07 * d) ** 2))
    assert ajustar_decaimento(d, e, "gaussiana")["r2"] > ajustar_decaimento(d, e, "exponencial")["r2"]


def test_ajuste_ignora_nan():
    d = np.arange(1, 30, 2.0)
    e = 70.0 * np.exp(-0.1 * d)
    e[3] = np.nan
    assert ajustar_exponencial(d, e)["lambda"] == pytest.approx(0.1, rel=1e-3)


def test_ajuste_sem_dados_suficientes_devolve_nan():
    assert np.isnan(ajustar_exponencial([1, 2], [50, 40])["lambda"])
    assert np.isnan(ajustar_exponencial([1, 2, 3, 4], [0, 0, 0, 0])["lambda"])


def test_amplitude_limitada_a_100():
    d = np.arange(1, 30, 2.0)
    aj = ajustar_exponencial(d, 150.0 * np.exp(-0.1 * d))  # dado fora do possível
    assert aj["A"] <= 100.0 + 1e-9


def test_raio_eficaz():
    d = np.array([1, 3, 5, 7, 9, 11.0])
    e = np.array([90, 70, 30, 4, 2, 0.0])
    assert raio_eficaz(d, e, 5.0) == 7.0
    assert np.isnan(raio_eficaz(d, np.array([90, 80, 70, 60, 50, 40.0]), 5.0))
    assert np.isnan(raio_eficaz([], []))


def test_raio_eficaz_ignora_faixas_sem_dados():
    d = np.array([1, 3, 5, 7.0])
    e = np.array([np.nan, 60.0, np.nan, 2.0])
    assert raio_eficaz(d, e, 5.0) == 7.0


def test_sd_binomial_esperado_valor_conhecido():
    from polinizacao import sd_binomial_esperado
    e = np.array([[50.0], [50.0]])
    n = np.array([[100], [100]])
    assert sd_binomial_esperado(e, n)[0] == pytest.approx(100 * np.sqrt(0.25 / 100))


def test_sd_binomial_bate_com_simulacao_binomial():
    from polinizacao import sd_binomial_esperado
    rng = np.random.default_rng(0)
    n = np.full((4000, 1), 25)
    e = 100.0 * rng.binomial(25, 0.3, size=(4000, 1)) / 25
    obs = np.std(e, axis=0, ddof=1)[0]
    assert obs == pytest.approx(sd_binomial_esperado(e, n)[0], rel=0.05)


def test_sd_binomial_ignora_faixas_sem_receptoras():
    from polinizacao import sd_binomial_esperado
    e = np.array([[np.nan, 40.0], [np.nan, 60.0]])
    n = np.array([[0, 10], [0, 10]])
    out = sd_binomial_esperado(e, n)
    assert np.isnan(out[0]) and np.isfinite(out[1])
