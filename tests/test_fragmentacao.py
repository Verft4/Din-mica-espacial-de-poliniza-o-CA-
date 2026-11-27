"""Testes das extensões de fragmentação (limitação 4)."""

import numpy as np
import pytest
from scipy.ndimage import label

from polinizacao import (
    COLMEIA, FLOR_DOADORA, FLOR_RECEPTORA, POLINIZADA, VAZIO, ModeloPolinizacao, Parametros,
)

CONECTIVIDADE_8 = np.ones((3, 3), dtype=int)


def regiao_alcancavel(m):
    """Células alcançáveis a partir da colmeia por movimentos de Moore sobre células não-obstáculo."""
    rotulos, _ = label(m.paisagem != VAZIO, structure=CONECTIVIDADE_8)
    return rotulos == rotulos[m.centro]


def pares_adjacentes(mascara):
    return (np.mean(mascara[:, :-1] & mascara[:, 1:]) + np.mean(mascara[:-1, :] & mascara[1:, :])) / 2


# --------------------------------------------------------------------- anel-barreira
@pytest.mark.parametrize("raio", [4, 9, 15, 25])
@pytest.mark.parametrize("espessura", [2.0, 3.0])
def test_anel_fechado_isola_a_colmeia(raio, espessura):
    m = ModeloPolinizacao(Parametros(barreira_raio=raio, barreira_espessura=espessura,
                                     frac_obstaculos=0.0), seed=1)
    alcance = regiao_alcancavel(m)
    assert alcance.sum() > 1
    assert m._distancias[alcance].max() < raio


def test_anel_fino_vaza_na_diagonal():
    """Documenta por que a espessura padrão é 2: com 1 célula a vizinhança de Moore atravessa."""
    m = ModeloPolinizacao(Parametros(barreira_raio=10, barreira_espessura=1.0, frac_obstaculos=0.0), seed=1)
    assert m._distancias[regiao_alcancavel(m)].max() > 10


def test_aberturas_permitem_passar():
    m = ModeloPolinizacao(Parametros(barreira_raio=10, barreira_aberturas=1, barreira_largura_abertura=3.0,
                                     frac_obstaculos=0.0), seed=1)
    assert m._distancias[regiao_alcancavel(m)].max() > 20


def test_mais_aberturas_menos_celulas_bloqueadas():
    contagens = [int(np.sum(ModeloPolinizacao(
        Parametros(barreira_raio=12, barreira_aberturas=k, frac_obstaculos=0.0), seed=1).paisagem == VAZIO))
        for k in (0, 1, 2, 4, 8)]
    assert all(a > b for a, b in zip(contagens, contagens[1:]))


def test_celulas_do_anel_sao_exatamente_o_obstaculo():
    p = Parametros(barreira_raio=12, barreira_espessura=2.0, frac_obstaculos=0.0)
    m = ModeloPolinizacao(p, seed=3)
    r = m._distancias
    esperado = (r >= 12) & (r < 14)
    assert np.array_equal(m.paisagem == VAZIO, esperado)
    assert m.paisagem[m.centro] == COLMEIA


def test_anel_fechado_impede_polinizacao_fora():
    p = Parametros(barreira_raio=8, frac_obstaculos=0.0, max_abelhas=60, prob_geracao=1.0,
                   densidade_floral=0.3)
    m = ModeloPolinizacao(p, seed=2)
    for _ in range(300):
        m.passo()
        n = m.n_abelhas
        assert m._distancias[m.ab_i[:n], m.ab_j[:n]].max() < 8
    fora = m._distancias >= 10
    assert int(np.sum(m.paisagem[fora] == POLINIZADA)) == 0
    assert m.n_polinizadas > 0                       # dentro, a polinização acontece


def test_anel_com_corredor_permite_polinizacao_fora():
    p = Parametros(barreira_raio=8, barreira_aberturas=2, barreira_largura_abertura=4.0,
                   frac_obstaculos=0.0, max_abelhas=60, prob_geracao=1.0, densidade_floral=0.3)
    m = ModeloPolinizacao(p, seed=2).executar(400)
    assert int(np.sum(m.paisagem[m._distancias >= 10] == POLINIZADA)) > 0


# ------------------------------------------------------------------------- blocos
def test_blocos_criam_obstaculos_extensos_e_poupam_a_colmeia():
    m = ModeloPolinizacao(Parametros(n_blocos=5, tamanho_bloco=6, frac_obstaculos=0.0), seed=1)
    n_vazio = int(np.sum(m.paisagem == VAZIO))
    assert 20 <= n_vazio <= 5 * 36
    cx, cy = m.centro
    assert np.all(m.paisagem[cx - 1:cx + 2, cy - 1:cy + 2][(np.arange(9) != 4).reshape(3, 3)] != VAZIO)
    assert m.paisagem[m.centro] == COLMEIA


def test_blocos_sao_mais_agrupados_que_obstaculos_independentes():
    blocos = ModeloPolinizacao(Parametros(n_blocos=6, tamanho_bloco=7, frac_obstaculos=0.0), seed=2)
    frac = (blocos.paisagem == VAZIO).mean()
    indep = ModeloPolinizacao(Parametros(frac_obstaculos=float(frac)), seed=2)
    assert pares_adjacentes(blocos.paisagem == VAZIO) > 3 * pares_adjacentes(indep.paisagem == VAZIO)


def test_bloco_maior_que_a_grade_e_recortado():
    m = ModeloPolinizacao(Parametros(linhas=20, colunas=20, n_blocos=1, tamanho_bloco=99,
                                     frac_obstaculos=0.0), seed=1)
    assert m.paisagem[m.centro] == COLMEIA


# ------------------------------------------------------------------ flores agrupadas
def _mascara_flores(m):
    return (m.paisagem == FLOR_DOADORA) | (m.paisagem == FLOR_RECEPTORA)


def test_flores_agrupadas_mantem_a_densidade_exata():
    p = Parametros(agregacao_flores=2.0, densidade_floral=0.15, frac_obstaculos=0.05)
    m = ModeloPolinizacao(p, seed=1)
    livres = int(np.sum(m.paisagem != VAZIO))
    # a colmeia (célula livre) pode ter caído sobre uma flor; tolerância de 2 células
    assert abs(int(_mascara_flores(m).sum()) - round(0.15 * livres)) <= 2


def test_flores_agrupadas_formam_manchas():
    ind = ModeloPolinizacao(Parametros(agregacao_flores=0.0), seed=3)
    agr = ModeloPolinizacao(Parametros(agregacao_flores=2.5), seed=3)
    assert pares_adjacentes(_mascara_flores(agr)) > 2 * pares_adjacentes(_mascara_flores(ind))


def test_flores_agrupadas_nunca_em_obstaculos_e_mistura_de_tipos():
    m = ModeloPolinizacao(Parametros(agregacao_flores=2.0, frac_obstaculos=0.2), seed=4)
    assert not np.any(_mascara_flores(m) & (m.paisagem == VAZIO))
    d, r = int(np.sum(m.paisagem == FLOR_DOADORA)), int(np.sum(m.paisagem == FLOR_RECEPTORA))
    assert 0.35 < d / (d + r) < 0.65


def test_flores_agrupadas_reprodutivel_e_executa():
    p = Parametros(agregacao_flores=2.0, n_blocos=3, barreira_raio=10, barreira_aberturas=2)
    a = ModeloPolinizacao(p, seed=5).executar(60)
    b = ModeloPolinizacao(p, seed=5).executar(60)
    assert a.hist_polinizadas == b.hist_polinizadas


@pytest.mark.parametrize("kw", [
    dict(barreira_raio=2.0), dict(barreira_raio=-1.0), dict(barreira_espessura=0.0),
    dict(barreira_largura_abertura=0.0), dict(barreira_aberturas=-1), dict(n_blocos=-1),
    dict(tamanho_bloco=0), dict(agregacao_flores=-1.0),
])
def test_parametros_de_fragmentacao_invalidos(kw):
    with pytest.raises(ValueError):
        Parametros(**kw)


def test_bloco_do_tamanho_da_grade_poupa_so_o_entorno_da_colmeia():
    """Determinístico: o bloco cobre tudo; só a colmeia e seus 8 vizinhos continuam livres."""
    m = ModeloPolinizacao(Parametros(n_blocos=1, tamanho_bloco=60, frac_obstaculos=0.0), seed=1)
    assert int(np.sum(m.paisagem == VAZIO)) == 3600 - 9
    cx, cy = m.centro
    assert not np.any(m.paisagem[cx - 1:cx + 2, cy - 1:cy + 2] == VAZIO)
