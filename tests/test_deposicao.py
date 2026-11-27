"""Testes do módulo de deposição (meta-regressão, validação por estudo, Cecala) e do parâmetro de tamanho no CA."""

from pathlib import Path

import numpy as np
import pytest

from polinizacao import ModeloPolinizacao, Parametros
from polinizacao import deposicao as dp
from polinizacao.modelo import FLOR_RECEPTORA

DADOS = Path(__file__).resolve().parents[1] / "dados"


def _sintetico(b=0.8, tau=0.0, k=30, m=3, seed=1):
    rng = np.random.default_rng(seed)
    est, x, y, v = [], [], [], []
    for s in range(k):
        u = rng.normal(0, tau) if tau > 0 else 0.0
        for _ in range(m):
            xi = rng.uniform(np.log(0.4), np.log(1.6))
            vi = rng.uniform(0.02, 0.1)
            est.append(f"e{s}")
            x.append(xi)
            v.append(vi)
            y.append(b * xi + u + rng.normal(0, np.sqrt(vi)))
    return np.array(est), np.array(x), np.array(y), np.array(v)


def test_recupera_inclinacao_conhecida():
    est, x, y, v = _sintetico(b=0.8)
    aj = dp.ajustar_metarregressao(x, y, v, intercepto=False)
    assert abs(aj.beta[0] - 0.8) < 0.1
    aj2 = dp.ajustar_metarregressao(x, y, v, intercepto=True)
    assert abs(aj2.beta[1] - 0.8) < 0.15 and abs(aj2.beta[0]) < 0.1


def test_tau2_recupera_heterogeneidade():
    est, x, y, v = _sintetico(b=0.0, tau=0.5, k=80)
    aj = dp.ajustar_metarregressao(x, y, v, True)
    assert 0.15 < aj.tau2 < 0.4   # verdadeiro 0.25


def test_tau2_zero_sem_heterogeneidade():
    est, x, y, v = _sintetico(b=0.5, tau=0.0, k=60)
    assert dp.ajustar_metarregressao(x, y, v, True).tau2 < 0.02


def test_loso_prefere_modelo_verdadeiro():
    est, x, y, v = _sintetico(b=1.0, k=40)
    lo = dp.validar_loso(est, x, y, v)
    assert lo["M3"]["mse_ponderado"] < lo["M1"]["mse_ponderado"] < lo["M0"]["mse_ponderado"] * 1.01 or \
        lo["M3"]["mse_ponderado"] < lo["M0"]["mse_ponderado"]
    assert lo["M3"]["mse_ponderado"] < 0.5 * lo["M1"]["mse_ponderado"]


def test_loso_nao_favorece_inclinacao_sob_nulo():
    """Sem efeito de tamanho, a reta não deve ganhar do intercepto na validação."""
    est, x, y, v = _sintetico(b=0.0, k=40, seed=5)
    lo = dp.validar_loso(est, x, y, v)
    assert lo["M3"]["mse_ponderado"] > 0.8 * lo["M1"]["mse_ponderado"]


def test_loso_omite_estudo_inteiro():
    """Um estudo omitido não pode influenciar a própria previsão: corromper seus dados não muda o ajuste dos demais."""
    est, x, y, v = _sintetico(b=0.5, k=12)
    a = dp.validar_loso(est, x, y, v)
    y2 = y.copy()
    y2[est == "e0"] += 50.0
    b = dp.validar_loso(est, x, y2, v)
    # erro de e0 muda (o dado mudou), mas a previsão de e0 usa só os outros estudos: erro varia exatamente pelo deslocamento
    d = b["M3"]["erro"][est == "e0"] - a["M3"]["erro"][est == "e0"]
    assert np.allclose(d, 50.0)


def test_cobertura_calibrada_sob_modelo_verdadeiro():
    est, x, y, v = _sintetico(b=0.7, tau=0.3, k=60, seed=9)
    lo = dp.validar_loso(est, x, y, v)
    assert 0.88 <= lo["M3"]["cobertura95"] <= 1.0


def test_bootstrap_cobre_o_valor_verdadeiro():
    est, x, y, v = _sintetico(b=0.6, k=40)
    bs = dp.bootstrap_estudos(est, x, y, v, False, 300, 1)
    lo, hi = np.quantile(bs[:, 0], [0.025, 0.975])
    assert lo < 0.6 < hi


def test_diferenca_loso_ic_sinal():
    est, x, y, v = _sintetico(b=1.0, k=40)
    lo = dp.validar_loso(est, x, y, v)
    d, a, b = dp.diferenca_loso_ic(est, lo["M1"]["erro"], lo["M3"]["erro"], lo["w_ref"], n_boot=500)
    assert d > 0 and a > 0 and a < d < b


def test_carregar_foldesi_dados_reais():
    obs = dp.carregar_foldesi(DADOS / "foldesi2020_metadata.csv")
    assert len(obs) == 111
    assert len(np.unique(obs.estudo)) == 27   # a página do Dryad cita 127 observações e 28 estudos; o arquivo tem 111 e 27
    sub, info = dp.subconjunto_abelhas(obs, "itd")
    assert info["n_usado"] == len(sub) == 73
    assert info["excluidos_tamanho_nulo"] == 1       # ITD = 0 (Chalepogenus caeruleus): inviável em log
    assert np.all(sub.itd > 0) and np.all(sub.ordem == "Hymenoptera")


def test_carregar_cecala_dados_reais():
    c = dp.carregar_cecala(DADOS / "cecala2020_visitas.csv")
    assert (c.grupo == "Apis").sum() == 173 and (c.grupo == "nativa").sum() == 130


def test_log_razao_medias_ic_contem_estimativa():
    rng = np.random.default_rng(0)
    a = rng.poisson(30, 80).astype(float)
    b = rng.poisson(60, 80).astype(float)
    est, lo, hi = dp.log_razao_medias(a, b, n_boot=500)
    assert lo < est < hi and abs(est - np.log(0.5)) < 0.15


def test_multiplicador():
    assert dp.multiplicador(0.5, 1.0) == pytest.approx(0.5)
    assert dp.multiplicador(0.5, 0.0) == 1.0
    assert dp.multiplicador(2.0, 1.0) == pytest.approx(2.0)


# --------------------------------------------------------------------------- parâmetro no autômato
def _deposito_unico(p, carga=20.0):
    m = ModeloPolinizacao(p, seed=3)
    x, y = np.argwhere(m.paisagem == FLOR_RECEPTORA)[0]
    m.ab_carga[0] = carga
    antes = m.deposicao_total
    m._interagir(0, int(x), int(y))
    return m.deposicao_total - antes


def test_taxa_efetiva_segue_lei_de_potencia():
    base = Parametros(max_abelhas=1)
    ref = _deposito_unico(base)
    assert ref == pytest.approx(20.0 * 0.6)
    p = base.com(tamanho_relativo=0.5, expoente_deposicao=1.0)
    assert _deposito_unico(p) == pytest.approx(20.0 * 0.6 * 0.5)
    p = base.com(tamanho_relativo=0.5, expoente_deposicao=2.0)
    assert _deposito_unico(p) == pytest.approx(20.0 * 0.6 * 0.25)


def test_taxa_efetiva_satura_em_um():
    p = Parametros(max_abelhas=1, tamanho_relativo=3.0, expoente_deposicao=1.0)
    assert _deposito_unico(p) == pytest.approx(20.0)   # nunca deposita mais do que carrega


def test_expoente_zero_nao_muda_nada():
    a = _deposito_unico(Parametros(max_abelhas=1))
    b = _deposito_unico(Parametros(max_abelhas=1, tamanho_relativo=0.3, expoente_deposicao=0.0))
    assert a == b


def test_validacao_tamanho_e_abm():
    with pytest.raises(ValueError):
        Parametros(tamanho_relativo=0.0)
    from polinizacao import ModeloABMContinuo
    with pytest.raises(ValueError):
        ModeloABMContinuo(Parametros(expoente_deposicao=1.0), seed=1)


def test_defeito_reproduz_base():
    a = ModeloPolinizacao(Parametros(), seed=7).executar(40)
    b = ModeloPolinizacao(Parametros(tamanho_relativo=0.2, expoente_deposicao=0.0), seed=7).executar(40)
    assert a.n_polinizadas == b.n_polinizadas and a.deposicao_total == b.deposicao_total


def test_abelha_menor_poliniza_menos_com_expoente_positivo():
    ef = []
    for s in (0.3, 1.0):
        p = Parametros(tamanho_relativo=s, expoente_deposicao=1.5)
        ef.append(np.mean([ModeloPolinizacao(p, seed=k).executar(60).n_polinizadas for k in range(8)]))
    assert ef[0] < ef[1]
