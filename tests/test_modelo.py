"""Testes do modelo: invariantes, regras de transição e métricas."""

import numpy as np
import pytest
from scipy import stats

from polinizacao import (
    COLMEIA, FLOR_DOADORA, FLOR_RECEPTORA, NEUTRO, POLINIZADA, VAZIO,
    ModeloPolinizacao, Parametros,
)


def modelo_vazio(**kw):
    """Paisagem 100% neutra (sem flores nem obstáculos), 60x60, colmeia no centro."""
    base = dict(densidade_floral=0.0, frac_obstaculos=0.0)
    base.update(kw)
    return ModeloPolinizacao(Parametros(**base), seed=0)


# ----------------------------------------------------------- reprodutibilidade
def test_mesma_semente_mesmo_resultado():
    a = ModeloPolinizacao(seed=42).executar(60)
    b = ModeloPolinizacao(seed=42).executar(60)
    assert np.array_equal(a.paisagem, b.paisagem)
    assert a.hist_polinizadas == b.hist_polinizadas
    assert np.array_equal(a.ab_i[: a.n_abelhas], b.ab_i[: b.n_abelhas])


def test_sementes_diferentes_resultados_diferentes():
    a = ModeloPolinizacao(seed=1)
    b = ModeloPolinizacao(seed=2)
    assert not np.array_equal(a.paisagem, b.paisagem)


# ------------------------------------------------------------------ invariantes
def test_invariantes_a_cada_passo():
    m = ModeloPolinizacao(seed=3)
    p = m.p
    anterior_pol, anterior_n = 0, 0
    for _ in range(150):
        m.passo()
        n = m.n_abelhas
        # abelhas nunca somem: só crescem (geração) e respeitam o limite
        assert anterior_n <= n <= p.max_abelhas
        assert n - anterior_n in (0, 1)
        anterior_n = n
        # ocupação consistente: uma abelha por célula, nenhuma perdida
        pos = set(zip(m.ab_i[:n].tolist(), m.ab_j[:n].tolist()))
        assert len(pos) == n
        assert int(m.ocupacao.sum()) == n
        assert all(m.ocupacao[i, j] for i, j in pos)
        # nunca dentro de obstáculo
        assert all(m.paisagem[i, j] != VAZIO for i, j in pos)
        # carga limitada
        assert np.all(m.ab_carga[:n] >= 0) and np.all(m.ab_carga[:n] <= p.cap_max + 1e-9)
        # estoque limitado
        assert np.all(m.estoque >= -1e-9) and np.all(m.estoque <= p.estoque_max + 1e-9)
        # polinizadas só aumentam e batem com a grade
        assert m.n_polinizadas >= anterior_pol
        assert m.n_polinizadas == int(np.sum(m.paisagem == POLINIZADA))
        anterior_pol = m.n_polinizadas
        # colmeia intocada
        assert m.paisagem[m.centro] == COLMEIA


def test_receptoras_mais_polinizadas_constante():
    m = ModeloPolinizacao(seed=5)
    total = m.total_receptoras
    m.executar(100)
    assert int(np.sum((m.paisagem == FLOR_RECEPTORA) | (m.paisagem == POLINIZADA))) == total


def test_historicos_tem_um_valor_por_passo():
    m = ModeloPolinizacao(seed=0).executar(37)
    assert len(m.hist_polinizadas) == len(m.hist_visitas) == len(m.hist_abelhas) == 37
    assert len(m.hist_deposicao) == 37


def test_sem_abelhas_nada_acontece():
    m = ModeloPolinizacao(Parametros(max_abelhas=0), seed=0).executar(50)
    assert m.n_abelhas == 0 and m.n_polinizadas == 0


def test_abelha_presa_nao_some():
    """Abelha cercada (obstáculos e colmeia) fica onde está: nada se perde."""
    m = modelo_vazio(max_abelhas=1, prob_geracao=1.0)
    m.passo()  # gera a abelha, que sai da colmeia
    i, j = int(m.ab_i[0]), int(m.ab_j[0])
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) != (0, 0):
                m.paisagem[i + di, j + dj] = VAZIO
                m._peso[i + di, j + dj] = 0.0
    for _ in range(20):
        m.passo()
        assert (int(m.ab_i[0]), int(m.ab_j[0])) == (i, j)
        assert m.n_abelhas == 1 and m.ocupacao[i, j]


# ------------------------------------------------------------------- vizinhança
def test_vizinhanca_moore_tem_8_celulas():
    m = modelo_vazio()
    _, _, w = m._janela_pesos(30, 30)
    assert w[1, 1] == 0 and np.count_nonzero(w) == 8


def test_borda_fechada_vs_toroidal():
    fechada = modelo_vazio(toroidal=False)
    _, _, w = fechada._janela_pesos(0, 0)
    assert np.count_nonzero(w) == 3
    toro = modelo_vazio(toroidal=True)
    linhas, colunas, w = toro._janela_pesos(0, 0)
    assert np.count_nonzero(w) == 8
    assert set(linhas.tolist()) == {59, 0, 1} and set(colunas.tolist()) == {59, 0, 1}


def test_sem_flores_movimento_equiprovavel():
    m = modelo_vazio()
    n = 8000
    cont = {}
    for _ in range(n):
        d = m._escolher_destino(30, 30)
        cont[d] = cont.get(d, 0) + 1
    assert len(cont) == 8
    chi2, p = stats.chisquare(list(cont.values()))
    assert p > 0.001, f"movimento não uniforme (p={p})"


def test_atracao_floral_probabilidade():
    m = modelo_vazio(alpha_atracao=12.0)
    m.paisagem[31, 30] = FLOR_DOADORA
    m._peso[31, 30] = 12.0
    n = 6000
    hits = sum(m._escolher_destino(30, 30) == (31, 30) for _ in range(n))
    esperado = 12.0 / (12.0 + 7.0)
    sigma = np.sqrt(esperado * (1 - esperado) / n)
    assert abs(hits / n - esperado) < 4 * sigma


def test_bloqueios_obstaculo_colmeia_e_ocupada():
    m = modelo_vazio()
    m.paisagem[31, 30] = VAZIO
    m._peso[31, 30] = 0.0
    m.paisagem[29, 30] = COLMEIA
    m._peso[29, 30] = 0.0
    m.ocupacao[30, 31] = True
    _, _, w = m._janela_pesos(30, 30)
    assert w[2, 1] == 0 and w[0, 1] == 0 and w[1, 2] == 0
    assert np.count_nonzero(w) == 5
    for _ in range(300):
        assert m._escolher_destino(30, 30) not in {(31, 30), (29, 30), (30, 31)}


# ------------------------------------------------------------ regras de interação
def preparar_abelha(m, carga=0.0, i=10, j=10):
    m.ab_i[0], m.ab_j[0], m.ab_carga[0] = i, j, carga
    m.n_abelhas = max(m.n_abelhas, 1)
    m.ocupacao[i, j] = True


def test_coleta_na_doadora():
    m = modelo_vazio()
    m.paisagem[10, 10] = FLOR_DOADORA
    m.estoque[10, 10] = m.p.estoque_max
    preparar_abelha(m, carga=0.0)
    assert m._interagir(0, 10, 10) is True
    assert m.ab_carga[0] == pytest.approx(m.p.eficiencia_coleta)
    assert m.estoque[10, 10] == pytest.approx(m.p.estoque_max - m.p.eficiencia_coleta)


def test_coleta_limitada_por_capacidade_e_estoque():
    m = modelo_vazio()
    m.paisagem[10, 10] = FLOR_DOADORA
    m.estoque[10, 10] = 100.0
    preparar_abelha(m, carga=m.p.cap_max - 3.0)
    m._interagir(0, 10, 10)
    assert m.ab_carga[0] == pytest.approx(m.p.cap_max)  # só coube 3
    m.estoque[10, 10] = 2.0
    m.ab_carga[0] = 0.0
    m._interagir(0, 10, 10)
    assert m.ab_carga[0] == pytest.approx(2.0)  # só havia 2
    assert m.estoque[10, 10] == pytest.approx(0.0)


def test_deposicao_polinizadora():
    m = modelo_vazio()
    m.paisagem[10, 10] = FLOR_RECEPTORA
    preparar_abelha(m, carga=40.0)
    m._interagir(0, 10, 10)
    assert m.paisagem[10, 10] == POLINIZADA and m.n_polinizadas == 1
    assert m.ab_carga[0] == pytest.approx(40.0 * (1 - m.p.taxa_deposicao))


def test_deposicao_insuficiente_nao_poliniza():
    m = modelo_vazio()
    m.paisagem[10, 10] = FLOR_RECEPTORA
    preparar_abelha(m, carga=5.0)  # deposição = 3 < limiar 4
    m._interagir(0, 10, 10)
    assert m.paisagem[10, 10] == FLOR_RECEPTORA and m.n_polinizadas == 0


def test_abelha_sem_carga_nao_poliniza():
    m = modelo_vazio()
    m.paisagem[10, 10] = FLOR_RECEPTORA
    preparar_abelha(m, carga=0.0)
    m._interagir(0, 10, 10)
    assert m.paisagem[10, 10] == FLOR_RECEPTORA


def test_recarga_floral():
    m = modelo_vazio(t_recarga=5)
    m.paisagem[10, 10] = FLOR_DOADORA
    m.estoque[10, 10] = 0.0
    for k in range(4):
        m._recarregar_flores()
        assert m.estoque[10, 10] == 0.0, f"recarregou cedo (k={k})"
    m._recarregar_flores()
    assert m.estoque[10, 10] == m.p.estoque_max and m.timer_recarga[10, 10] == 0


def test_tempo_de_retencao_zera_carga():
    m = modelo_vazio(t_retencao=5)
    preparar_abelha(m, carga=30.0)
    for _ in range(4):
        m._interagir(0, 10, 10)
        assert m.ab_carga[0] == 30.0
    m._interagir(0, 10, 10)  # 5ª vez sem flor
    assert m.ab_carga[0] == 0.0


def test_visita_a_flor_zera_contador_de_retencao():
    m = modelo_vazio(t_retencao=5)
    m.paisagem[10, 10] = FLOR_RECEPTORA
    preparar_abelha(m, carga=0.0)
    m.ab_sem_flor[0] = 4
    m._interagir(0, 10, 10)
    assert m.ab_sem_flor[0] == 0


# ------------------------------------------------------------------------ métricas
def test_eficiencia_conta_apenas_receptoras():
    """Doadoras não entram no denominador (bug da versão original)."""
    m = modelo_vazio()
    cx, cy = m.centro
    m.paisagem[cx + 2, cy] = FLOR_DOADORA        # d=2
    m.paisagem[cx + 2, cy + 1] = FLOR_DOADORA    # d~2.24
    m.paisagem[cx, cy - 3] = POLINIZADA          # d=3
    _, ef, total = m.eficiencia_por_distancia(largura_bin=2.0)
    assert total.sum() == 1                      # só a polinizada conta
    assert ef[1] == pytest.approx(100.0)         # faixa [2,4): 1 de 1, apesar das 2 doadoras


def test_eficiencia_valores_exatos():
    m = modelo_vazio()
    cx, cy = m.centro
    m.paisagem[cx + 1, cy] = FLOR_RECEPTORA   # d=1  faixa 0
    m.paisagem[cx - 1, cy] = POLINIZADA       # d=1  faixa 0
    m.paisagem[cx, cy + 5] = FLOR_RECEPTORA   # d=5  faixa 2
    m.paisagem[cx + 1, cy + 1] = FLOR_DOADORA  # d~1.41 faixa 0, deve ser ignorada
    centros, ef, total = m.eficiencia_por_distancia(largura_bin=2.0)
    assert total[0] == 2 and ef[0] == pytest.approx(50.0)
    assert total[2] == 1 and ef[2] == pytest.approx(0.0)
    assert np.isnan(ef[1])  # faixa sem receptoras
    assert centros[0] == pytest.approx(1.0)


def test_eficiencia_sempre_entre_0_e_100():
    m = ModeloPolinizacao(seed=11).executar(120)
    _, ef, _ = m.eficiencia_por_distancia()
    validos = ef[~np.isnan(ef)]
    assert validos.size > 0 and np.all(validos >= 0) and np.all(validos <= 100)


def test_taxa_e_raio():
    m = modelo_vazio()
    cx, cy = m.centro
    assert np.isnan(m.raio_percentil())
    assert np.isnan(m.taxa_polinizacao)
    m.paisagem[cx + 3, cy] = POLINIZADA
    m.paisagem[cx, cy + 4] = POLINIZADA
    assert m.raio_percentil(1.0) == pytest.approx(4.0)
    assert m.raio_percentil(0.0) == pytest.approx(3.0)


def test_distancia_toroidal_igual_com_colmeia_central():
    """Com a colmeia no centro, dar a volta na grade nunca encurta a distância."""
    a = modelo_vazio(toroidal=False)._distancias
    b = modelo_vazio(toroidal=True)._distancias
    assert np.allclose(a, b)


# ------------------------------------------------------------------ paisagem
def test_fracao_exata_de_obstaculos():
    for frac in (0.0, 0.05, 0.2):
        m = ModeloPolinizacao(Parametros(frac_obstaculos=frac), seed=1)
        esperado = round(frac * 3600)
        assert abs(int(np.sum(m.paisagem == VAZIO)) - esperado) <= 1


def test_densidade_floral_media():
    dens = []
    for s in range(20):
        m = ModeloPolinizacao(Parametros(densidade_floral=0.2, frac_obstaculos=0.0), seed=s)
        flores = np.sum((m.paisagem == FLOR_DOADORA) | (m.paisagem == FLOR_RECEPTORA))
        dens.append(flores / 3600)
    assert np.mean(dens) == pytest.approx(0.2, abs=0.01)


def _pares_adjacentes(obst):
    return (np.mean(obst[:, :-1] & obst[:, 1:]) + np.mean(obst[:-1, :] & obst[1:, :])) / 2


def test_agregacao_cria_aglomerados():
    ind = ModeloPolinizacao(Parametros(frac_obstaculos=0.15, agregacao_obstaculos=0.0), seed=4)
    agr = ModeloPolinizacao(Parametros(frac_obstaculos=0.15, agregacao_obstaculos=2.5), seed=4)
    assert _pares_adjacentes(agr.paisagem == VAZIO) > 2 * _pares_adjacentes(ind.paisagem == VAZIO)


def test_estoque_inicial_so_nas_doadoras():
    m = ModeloPolinizacao(seed=0)
    assert np.all(m.estoque[m.paisagem == FLOR_DOADORA] == m.p.estoque_max)
    assert np.all(m.estoque[m.paisagem != FLOR_DOADORA] == 0)


def test_colmeia_no_centro_e_unica():
    m = ModeloPolinizacao(seed=0)
    assert m.paisagem[30, 30] == COLMEIA and int(np.sum(m.paisagem == COLMEIA)) == 1


# ------------------------------------------------------------------ parâmetros
@pytest.mark.parametrize("kw", [
    dict(linhas=2), dict(densidade_floral=1.5), dict(taxa_deposicao=-0.1),
    dict(alpha_atracao=-1), dict(cap_max=0), dict(t_retencao=0), dict(t_recarga=0),
    dict(max_abelhas=-1), dict(agregacao_obstaculos=-1),
])
def test_parametros_invalidos(kw):
    with pytest.raises(ValueError):
        Parametros(**kw)


def test_com_cria_copia():
    p = Parametros()
    q = p.com(alpha_atracao=3.0)
    assert q.alpha_atracao == 3.0 and p.alpha_atracao == 12.0 and q.linhas == p.linhas


# ----------------------------------------------------- tabela de pesos w(S) do artigo
def test_tabela_de_pesos_segue_o_artigo():
    """w = alpha (doadora/receptora), 1 (neutro), 0 (vazio, colmeia, abelha)."""
    m = ModeloPolinizacao(Parametros(alpha_atracao=7.0), seed=0)
    t = m._tabela_pesos()
    assert t[FLOR_DOADORA] == 7.0 and t[FLOR_RECEPTORA] == 7.0
    assert t[NEUTRO] == 1.0
    assert t[VAZIO] == 0.0 and t[COLMEIA] == 0.0 and t[5] == 0.0
    assert m._peso[m.centro] == 0.0  # a colmeia nunca é destino de movimento


def test_abelha_nunca_volta_para_a_colmeia():
    m = ModeloPolinizacao(Parametros(max_abelhas=1, prob_geracao=1.0), seed=3)
    m.passo()
    for _ in range(300):
        m.passo()
        assert (int(m.ab_i[0]), int(m.ab_j[0])) != m.centro
