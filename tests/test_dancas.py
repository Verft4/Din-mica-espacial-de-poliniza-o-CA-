"""Testes do módulo de danças decodificadas (ajuste do núcleo de distância no nível de sessão)."""

import numpy as np
import pytest

from polinizacao import dancas as dc

CSV = (
    "actual.date,site,Land_use,median.translated.distance,mean.translated.distance\n"
    "01/01/2017,AAA,Agricultural,500,600\n02/01/2017,AAA,Agricultural,700,800\n"
    "03/01/2017,BBB,Agricultural,400,450\n04/01/2017,BBB,Agricultural,NA,450\n"
    "05/01/2017,CCC,Urban,300,350\n06/01/2017,DDD,Urban,350,380\n07/01/2017,CCC,Urban,0,10\n"
)


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text(CSV, encoding="utf-8")
    return p


def test_carregar_sessoes_descarta_invalidas(csv_path):
    s = dc.carregar_sessoes(csv_path)
    assert len(s) == 5                                   # NA e zero descartados
    assert set(s.colmeia) == {"AAA", "BBB", "CCC", "DDD"}
    with pytest.raises(ValueError):
        (csv_path.parent / "v.csv").write_text("site,Land_use,median.translated.distance,mean.translated.distance\n")
        dc.carregar_sessoes(csv_path.parent / "v.csv")


def test_dobras_separam_por_colmeia_e_sao_disjuntas(csv_path):
    s = dc.carregar_sessoes(csv_path)
    a, b = dc.dobras_por_colmeia(s, seed=1)
    assert not np.any(a & b) and np.all(a | b)
    assert not set(s.colmeia[a]) & set(s.colmeia[b])      # nenhuma colmeia nas duas dobras
    assert np.array_equal(dc.dobras_por_colmeia(s, seed=1)[0], a)


def test_estatisticas():
    e = dc.estatisticas([100, 200, 300, 400], [110, 220, 330, 440])
    assert e["mediana_das_medianas"] == 250 and e["mediana_das_medias"] == 275
    assert e["razao_media_mediana"] == pytest.approx(1.1)


def test_mistura_extremos_e_validacao():
    rng = np.random.default_rng(0)
    x = dc.amostrar_mistura(rng, 1.0, 500.0, 300.0, 200000)          # só exponencial
    assert x.mean() == pytest.approx(500, rel=0.02) and np.median(x) == pytest.approx(500 * np.log(2), rel=0.03)
    y = dc.amostrar_mistura(rng, 0.0, 500.0, 300.0, 200000)          # só Rayleigh
    assert y.mean() == pytest.approx(300 * np.sqrt(np.pi / 2), rel=0.02)
    assert y.mean() / np.median(y) < x.mean() / np.median(x)         # Rayleigh menos assimétrica
    with pytest.raises(ValueError):
        dc.amostrar_mistura(rng, 1.5, 1, 1, 3)


def test_simular_sessoes_formato_e_razao_maior_para_exponencial():
    rng = np.random.default_rng(1)
    md, me = dc.simular_sessoes(rng, 1.0, 500, 300, 0.0, 100, 15)
    assert md.shape == me.shape == (100,)
    r_exp = np.median(me / md)
    md2, me2 = dc.simular_sessoes(rng, 0.0, 500, 300, 0.0, 100, 15)
    assert r_exp > np.median(me2 / md2)


def test_abc_distingue_formatos_e_e_reprodutivel():
    rng = np.random.default_rng(2)
    def obs(p):
        md, me = dc.simular_sessoes(rng, p, 700, 600, 0.3, 183, 15)
        return dc.estatisticas(md, me)
    a = dc.abc_mistura(obs(1.0), n_sorteios=2500, n_sessoes=60, seed=3)
    b = dc.abc_mistura(obs(0.0), n_sorteios=2500, n_sessoes=60, seed=3)
    assert np.mean(a["p"]) > np.mean(b["p"]) + 0.15
    c = dc.abc_mistura(obs(0.0), n_sorteios=2500, n_sessoes=60, seed=3)
    assert len(b["p"]) == len(c["p"]) and np.all(np.diff(a["dist"]) >= -1e-12)


def test_estatisticas_da_viagem():
    post = {"p": np.array([1.0, 0.0]), "media_exp": np.array([500.0, 500.0]), "sigma_ray": np.array([300.0, 300.0])}
    e = dc.estatisticas_da_viagem(post, n=100000)
    assert e["media"][0] == pytest.approx(500, rel=0.03)
    assert e["cv"][0] == pytest.approx(1.0, rel=0.05)                # CV da exponencial = 1
    assert e["cv"][1] == pytest.approx(np.sqrt(4 / np.pi - 1), rel=0.05)  # CV da Rayleigh


def test_prever_sessoes_do_modelo_escala_com_metros_por_celula():
    d = np.array([4.0, 6.0, 8.0])
    md1, _ = dc.prever_sessoes_do_modelo(d, 100.0, 15, 50, seed=0)
    md2, _ = dc.prever_sessoes_do_modelo(d, 200.0, 15, 50, seed=0)
    assert np.allclose(md2, 2 * md1)


def test_sessoes_por_replica_preservam_heterogeneidade_entre_sessoes():
    reps = [np.full(50, 4.0), np.full(50, 12.0)]                  # duas "paisagens" muito diferentes
    md_rep, me_rep = dc.prever_sessoes_do_modelo(reps, 100.0, 15, 400, seed=1)
    _, me_pool = dc.prever_sessoes_do_modelo(np.concatenate(reps), 100.0, 15, 400, seed=1)
    assert set(np.round(md_rep)) == {400.0, 1200.0}                # cada sessão vem de uma só réplica
    assert me_rep.std() > 3 * me_pool.std()                         # reunir as viagens apaga a heterogeneidade
    with pytest.raises(ValueError):
        dc.prever_sessoes_do_modelo([np.array([])], 1.0, 5, 5)
