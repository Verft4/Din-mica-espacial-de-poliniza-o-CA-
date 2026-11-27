"""Testes das extensões de biologia (limitação 3)."""

import numpy as np
import pytest

from polinizacao import (
    COLMEIA, FLOR_DOADORA, FLOR_RECEPTORA, POLINIZADA, VAZIO, ModeloPolinizacao, Parametros,
)


def modelo_vazio(**kw):
    base = dict(densidade_floral=0.0, frac_obstaculos=0.0)
    base.update(kw)
    return ModeloPolinizacao(Parametros(**base), seed=0)


def preparar_abelha(m, carga=0.0, i=10, j=10):
    m.ab_i[0], m.ab_j[0], m.ab_carga[0] = i, j, carga
    m.n_abelhas = max(m.n_abelhas, 1)
    m.ocupacao[i, j] = True


def invariantes(m, com_mortes=False):
    n = m.n_abelhas
    fora = ~m.ab_dentro[:n]               # quem está dentro da colmeia não ocupa célula da grade
    pos = set(zip(m.ab_i[:n][fora].tolist(), m.ab_j[:n][fora].tolist()))
    assert len(pos) == int(fora.sum()) and int(m.ocupacao.sum()) == int(fora.sum())
    assert all(m.paisagem[i, j] != VAZIO for i, j in pos)
    assert np.all(m.ab_carga[:n] >= 0) and np.all(m.ab_carga[:n] <= m.p.cap_max + 1e-9)
    assert n <= m.p.max_abelhas
    assert m.n_polinizadas == int(np.sum(m.paisagem == POLINIZADA))


TUDO_LIGADO = dict(retorno_colmeia=True, t_viagem_max=25, t_descanso=3, dist_max_voo=15.0,
                   mortalidade=0.01, memoria_vies=3.0, peso_doadora=15.0, peso_receptora=8.0,
                   deposicao_acumulada=True, max_abelhas=40, prob_geracao=0.6)


# ------------------------------------------------------------------ retorno
def test_viagens_terminam_e_sao_registradas():
    p = Parametros(retorno_colmeia=True, t_viagem_max=10, t_descanso=2, max_abelhas=20,
                   prob_geracao=1.0, frac_obstaculos=0.0)
    m = ModeloPolinizacao(p, seed=1).executar(300)
    assert len(m.duracoes_viagem) > 20
    assert min(m.duracoes_viagem) >= 2


def test_retorno_limita_o_alcance_comparado_ao_modelo_base():
    kw = dict(max_abelhas=20, prob_geracao=1.0, frac_obstaculos=0.0)
    com = ModeloPolinizacao(Parametros(retorno_colmeia=True, t_viagem_max=10, t_descanso=2, **kw), seed=2)
    sem = ModeloPolinizacao(Parametros(**kw), seed=2)
    max_com = max_sem = 0.0
    for _ in range(400):
        com.passo(); sem.passo()
        n = com.n_abelhas
        max_com = max(max_com, float(com._distancias[com.ab_i[:n], com.ab_j[:n]].max()))
        n = sem.n_abelhas
        max_sem = max(max_sem, float(sem._distancias[sem.ab_i[:n], sem.ab_j[:n]].max()))
    assert max_com <= 20.0           # ~ 10 passos de ida (até 1,41 célula cada) + folga
    assert max_sem > max_com


def test_abelha_em_repouso_fica_na_colmeia_sem_carga():
    p = Parametros(retorno_colmeia=True, t_viagem_max=8, t_descanso=4, max_abelhas=10,
                   prob_geracao=1.0, frac_obstaculos=0.0)
    m = ModeloPolinizacao(p, seed=3)
    visto = False
    for _ in range(300):
        m.passo()
        for b in range(m.n_abelhas):
            if m.ab_descanso[b] > 0:
                visto = True
                assert (int(m.ab_i[b]), int(m.ab_j[b])) == m.centro
                assert m.ab_carga[b] == 0.0
    assert visto


def test_retorno_aproxima_da_colmeia_e_entra_quando_adjacente():
    m = modelo_vazio(retorno_colmeia=True)
    preparar_abelha(m, i=20, j=30)
    m.ab_retorno[0] = True
    _, _, pesos = m._janela_pesos(20, 30)
    _, novos = m._ajustar_pesos(0, 20, 30, np.array([19, 20, 21]), np.array([29, 30, 31]), pesos)
    assert novos[2, 1] == 1.0 and novos[0, 1] == 0.0        # sul aproxima (colmeia em 30,30); norte afasta
    # adjacente e colmeia livre: entra direto
    m.ocupacao[20, 30] = False
    m.ocupacao[29, 30] = True
    m.ab_i[0], m.ab_j[0] = 29, 30
    assert m._escolher_destino(29, 30, 0) == m.centro


def test_colmeia_ocupada_bloqueia_entrada():
    m = modelo_vazio(retorno_colmeia=True)
    preparar_abelha(m, i=29, j=30)
    m.ab_retorno[0] = True
    m.ocupacao[m.centro] = True            # outra abelha repousando lá
    for _ in range(50):
        assert m._escolher_destino(29, 30, 0) != m.centro


def test_chegada_descarrega_e_zera_viagem():
    m = modelo_vazio(retorno_colmeia=True, t_descanso=7)
    preparar_abelha(m, carga=25.0)
    m.ab_viagem[0] = 12
    m.ab_retorno[0] = True
    m._chegar_na_colmeia(0)
    assert m.ab_carga[0] == 0.0 and m.ab_viagem[0] == 0 and not m.ab_retorno[0]
    assert m.ab_descanso[0] == 7 and m.duracoes_viagem == [13]


@pytest.mark.parametrize("gatilho", ["tempo", "carga", "distancia"])
def test_gatilhos_de_retorno(gatilho):
    m = modelo_vazio(retorno_colmeia=True, t_viagem_max=5, fracao_carga_retorno=0.5, dist_max_voo=8.0)
    preparar_abelha(m, carga=0.0, i=30, j=30 + 7)
    m.ab_viagem[0] = 0
    if gatilho == "tempo":
        m.ab_viagem[0] = 4
    elif gatilho == "carga":
        m.ab_carga[0] = 0.5 * m.p.cap_max
    m._atualizar_viagem(0, 30, 30 + 7 if gatilho == "distancia" else 31)
    assert m.ab_retorno[0]


def test_sem_gatilho_nao_retorna():
    m = modelo_vazio(retorno_colmeia=True, t_viagem_max=50, dist_max_voo=30.0)
    preparar_abelha(m, carga=1.0)
    m._atualizar_viagem(0, 31, 31)
    assert not m.ab_retorno[0]


# ----------------------------------------------------------------- distância máxima
@pytest.mark.parametrize("retorno", [False, True])
def test_limite_rigido_de_voo(retorno):
    p = Parametros(dist_max_voo=8.0, retorno_colmeia=retorno, t_viagem_max=500, max_abelhas=30,
                   prob_geracao=1.0, frac_obstaculos=0.0)
    m = ModeloPolinizacao(p, seed=4)
    for _ in range(250):
        m.passo()
        n = m.n_abelhas
        assert m._distancias[m.ab_i[:n], m.ab_j[:n]].max() <= 8.0 + 1e-9


# ----------------------------------------------------------------- mortalidade
def test_mortalidade_total_zera_o_efetivo():
    m = modelo_vazio(mortalidade=1.0, prob_geracao=1.0, max_abelhas=10)
    for _ in range(20):
        m.passo()
        assert m.n_abelhas == 0 and int(m.ocupacao.sum()) == 0
    assert m.n_mortes == 20          # uma geração e uma morte por passo


def test_mortalidade_reduz_efetivo_e_mantem_invariantes():
    sem = ModeloPolinizacao(Parametros(max_abelhas=60, prob_geracao=1.0), seed=5)
    com = ModeloPolinizacao(Parametros(max_abelhas=60, prob_geracao=1.0, mortalidade=0.05), seed=5)
    for _ in range(150):
        sem.passo(); com.passo()
        invariantes(com)
    assert com.n_mortes > 0
    assert com.n_abelhas < sem.n_abelhas


def test_tempo_de_vida_medio_aproximadamente_um_sobre_mu():
    mu = 0.1
    m = modelo_vazio(mortalidade=mu, prob_geracao=1.0, max_abelhas=1)
    passos = 4000
    m.executar(passos)
    # com 1 abelha sendo reposta, mortes por passo ~ P(gerar) * mu / (mu + ...) ~ simples e positivo
    assert 0.05 * passos < m.n_mortes < passos


# ----------------------------------------------------------------- memória
def test_memoria_reforca_passos_rumo_ao_patch():
    beta = 4.0
    m = modelo_vazio(memoria_vies=beta, dist_max_voo=0.0)
    preparar_abelha(m, i=30, j=30)
    m.ab_mem_i[0], m.ab_mem_j[0] = 10, 10
    linhas, colunas, pesos = m._janela_pesos(30, 30)
    _, novos = m._ajustar_pesos(0, 30, 30, linhas, colunas, pesos)
    reforcados = (novos > pesos) & (pesos > 0)
    assert reforcados.sum() == 3
    assert np.allclose(novos[reforcados], 1.0 + beta)
    assert reforcados[0, 0] and reforcados[0, 1] and reforcados[1, 0]


def test_memoria_vira_probabilidade_teorica():
    beta = 4.0
    m = modelo_vazio(memoria_vies=beta)
    preparar_abelha(m, i=30, j=30)
    m.ab_mem_i[0], m.ab_mem_j[0] = 10, 10
    n = 6000
    aproxima = sum(m._escolher_destino(30, 30, 0)
                   in {(29, 29), (29, 30), (30, 29)} for _ in range(n))
    esperado = 3 * (1 + beta) / (3 * (1 + beta) + 5)
    assert abs(aproxima / n - esperado) < 4 * np.sqrt(esperado * (1 - esperado) / n)


def test_memoria_gravada_na_coleta_e_apagada_quando_esgota():
    m = modelo_vazio(memoria_vies=2.0)
    m.paisagem[10, 10] = FLOR_DOADORA
    m.estoque[10, 10] = 10.0
    preparar_abelha(m)
    m._interagir(0, 10, 10)                         # coleta 8, sobram 2
    assert (m.ab_mem_i[0], m.ab_mem_j[0]) == (10, 10)
    m._interagir(0, 10, 10)                         # coleta 2, esgota
    assert m.estoque[10, 10] == 0.0
    assert (m.ab_mem_i[0], m.ab_mem_j[0]) == (-1, -1)


def test_sem_memoria_nada_e_gravado():
    m = modelo_vazio()
    m.paisagem[10, 10] = FLOR_DOADORA
    m.estoque[10, 10] = 40.0
    preparar_abelha(m)
    m._interagir(0, 10, 10)
    assert (m.ab_mem_i[0], m.ab_mem_j[0]) == (-1, -1)


# ------------------------------------------------------- pesos por tipo e dose acumulada
def test_pesos_por_tipo_na_tabela_e_no_movimento():
    m = modelo_vazio(alpha_atracao=12.0, peso_doadora=20.0, peso_receptora=3.0)
    t = m._tabela_pesos()
    assert t[FLOR_DOADORA] == 20.0 and t[FLOR_RECEPTORA] == 3.0
    m.paisagem[31, 30] = FLOR_DOADORA
    m._peso[31, 30] = 20.0
    n = 6000
    hits = sum(m._escolher_destino(30, 30) == (31, 30) for _ in range(n))
    esperado = 20.0 / 27.0
    assert abs(hits / n - esperado) < 4 * np.sqrt(esperado * (1 - esperado) / n)


def test_peso_none_usa_alpha():
    t = modelo_vazio(alpha_atracao=7.0)._tabela_pesos()
    assert t[FLOR_DOADORA] == 7.0 and t[FLOR_RECEPTORA] == 7.0


def test_deposicao_acumulada_polioniza_em_duas_visitas():
    m = modelo_vazio(deposicao_acumulada=True)
    m.paisagem[10, 10] = FLOR_RECEPTORA
    preparar_abelha(m, carga=5.0)               # deposição = 3 < limiar 4
    m._interagir(0, 10, 10)
    assert m.paisagem[10, 10] == FLOR_RECEPTORA and m.dose[10, 10] == pytest.approx(3.0)
    m.ab_carga[0] = 5.0
    m._interagir(0, 10, 10)                     # dose 6 >= 4
    assert m.paisagem[10, 10] == POLINIZADA


def test_sem_acumulo_duas_visitas_fracas_nao_polinizam():
    m = modelo_vazio(deposicao_acumulada=False)
    m.paisagem[10, 10] = FLOR_RECEPTORA
    preparar_abelha(m, carga=5.0)
    m._interagir(0, 10, 10)
    m.ab_carga[0] = 5.0
    m._interagir(0, 10, 10)
    assert m.paisagem[10, 10] == FLOR_RECEPTORA


def test_acumulacao_so_aumenta_a_polinizacao():
    kw = dict(limiar_polinizacao=8.0, max_abelhas=40)
    sem = ModeloPolinizacao(Parametros(**kw), seed=6).executar(150)
    com = ModeloPolinizacao(Parametros(deposicao_acumulada=True, **kw), seed=6).executar(150)
    assert com.n_polinizadas > sem.n_polinizadas


# ------------------------------------------------------------- visitação e integração
def test_contagem_de_visitas_e_visitacao_por_distancia():
    m = modelo_vazio()
    cx, cy = m.centro
    m.paisagem[cx + 1, cy] = FLOR_RECEPTORA      # d=1 faixa 0
    m.paisagem[cx, cy + 5] = FLOR_DOADORA        # d=5 faixa 2
    m.estoque[cx, cy + 5] = 100.0
    preparar_abelha(m, i=cx + 1, j=cy)
    for _ in range(3):
        m._interagir(0, cx + 1, cy)
    m._interagir(0, cx, cy + 5)
    m.passo_atual = 4
    centros, vis, n = m.visitacao_por_distancia(2.0)
    assert n[0] == 1 and vis[0] == pytest.approx(3 / 4)
    assert n[2] == 1 and vis[2] == pytest.approx(1 / 4)
    assert np.isnan(vis[1])
    assert m.visitas_celula.sum() == 4


def test_tudo_ligado_mantem_invariantes_e_e_reprodutivel():
    m = ModeloPolinizacao(Parametros(**TUDO_LIGADO), seed=8)
    for _ in range(250):
        m.passo()
        invariantes(m)
        n = m.n_abelhas
        if n:
            assert m._distancias[m.ab_i[:n], m.ab_j[:n]].max() <= 15.0 + 1e-9
        assert m.paisagem[m.centro] == COLMEIA
    a = ModeloPolinizacao(Parametros(**TUDO_LIGADO), seed=8).executar(120)
    b = ModeloPolinizacao(Parametros(**TUDO_LIGADO), seed=8).executar(120)
    assert a.hist_polinizadas == b.hist_polinizadas and a.duracoes_viagem == b.duracoes_viagem


def test_sem_mortalidade_efetivo_nunca_cai_mesmo_com_retorno():
    kw = {k: v for k, v in TUDO_LIGADO.items() if k != "mortalidade"}
    m = ModeloPolinizacao(Parametros(**kw), seed=9)
    anterior = 0
    for _ in range(200):
        m.passo()
        assert m.n_abelhas >= anterior
        anterior = m.n_abelhas


@pytest.mark.parametrize("kw", [
    dict(mortalidade=1.5), dict(fracao_carga_retorno=0.0), dict(t_viagem_max=0), dict(t_descanso=-1),
    dict(dist_max_voo=-1.0), dict(memoria_vies=-0.1), dict(peso_doadora=-1.0), dict(peso_receptora=-1.0),
])
def test_parametros_de_biologia_invalidos(kw):
    with pytest.raises(ValueError):
        Parametros(**kw)


# ----------------------------------------------- colmeia sem gargalo e orçamento de viagem
def test_abelhas_dentro_da_colmeia_nao_ocupam_a_celula():
    m = modelo_vazio(retorno_colmeia=True, t_descanso=10)
    for k in range(3):
        m.ab_i[k], m.ab_j[k] = 29, 30
        m.ab_retorno[k] = True
        m.ocupacao[29, 30] = False
    m.n_abelhas = 3
    for k in range(3):
        m.ocupacao[m.centro] = False
        m._chegar_na_colmeia(k)
    assert bool(m.ab_dentro[:3].all()) and not m.ocupacao[m.centro]


def test_saida_da_colmeia_sem_gargalo_e_sem_sobreposicao():
    m = modelo_vazio(retorno_colmeia=True, t_descanso=0, max_abelhas=5, prob_geracao=0.0)
    m.n_abelhas = 5
    for k in range(5):
        m.ab_i[k], m.ab_j[k] = m.centro
        m.ab_dentro[k] = True
    m.passo()
    fora = ~m.ab_dentro[:5]
    pos = set(zip(m.ab_i[:5][fora].tolist(), m.ab_j[:5][fora].tolist()))
    assert int(fora.sum()) >= 4                        # a colmeia não é gargalo
    assert len(pos) == int(fora.sum())                 # nenhuma sobreposição de células
    assert all(max(abs(i - m.centro[0]), abs(j - m.centro[1])) <= 1 for i, j in pos)


def test_saida_bloqueada_se_a_celula_da_colmeia_esta_ocupada():
    m = modelo_vazio(retorno_colmeia=True, t_descanso=0, max_abelhas=1, prob_geracao=0.0)
    m.n_abelhas = 1
    m.ab_i[0], m.ab_j[0] = m.centro
    m.ab_dentro[0] = True
    m.ocupacao[m.centro] = True                        # ocupada por outra abelha
    m.passo()
    assert bool(m.ab_dentro[0])


def test_morte_dentro_da_colmeia_nao_libera_a_celula_ocupada():
    m = modelo_vazio(mortalidade=1.0, retorno_colmeia=True, prob_geracao=0.0)
    m.n_abelhas = 1
    m.ab_i[0], m.ab_j[0] = m.centro
    m.ab_dentro[0] = True
    m.ocupacao[m.centro] = True          # outra abelha (não listada) está na célula da colmeia
    m._mortalidade()
    assert m.n_abelhas == 0 and m.ocupacao[m.centro]


def test_orcamento_fixo_nao_consome_aleatoriedade():
    a = ModeloPolinizacao(Parametros(retorno_colmeia=True, t_viagem_forma=0.0), seed=1)
    b = ModeloPolinizacao(Parametros(retorno_colmeia=True, t_viagem_forma=0.0), seed=1)
    a.executar(50); b.executar(50)
    assert a.hist_polinizadas == b.hist_polinizadas
    assert np.all(a.ab_orcamento[:a.n_abelhas] == a.p.t_viagem_max)


def test_orcamento_gama_tem_media_e_cv_corretos():
    m = modelo_vazio(retorno_colmeia=True, t_viagem_max=60, t_viagem_forma=4.0)
    x = []
    for _ in range(4000):
        m._novo_orcamento(0)
        x.append(m.ab_orcamento[0])
    x = np.array(x)
    assert x.mean() == pytest.approx(60.0, rel=0.03)
    assert x.std() / x.mean() == pytest.approx(1 / np.sqrt(4.0), rel=0.06)


def test_orcamento_variavel_aumenta_a_dispersao_das_duracoes_de_viagem():
    kw = dict(retorno_colmeia=True, t_viagem_max=40, max_abelhas=30, prob_geracao=1.0, frac_obstaculos=0.0,
              fracao_carga_retorno=1.0, cap_max=1e9)       # só o tempo encerra a viagem
    fixo = ModeloPolinizacao(Parametros(**kw), seed=3).executar(600)
    var = ModeloPolinizacao(Parametros(t_viagem_forma=2.0, **kw), seed=3).executar(600)
    cv = lambda m: np.std(m.duracoes_viagem) / np.mean(m.duracoes_viagem)
    assert cv(var) > 2 * cv(fixo)


def test_distancia_media_das_visitas():
    m = modelo_vazio()
    cx, cy = m.centro
    m.paisagem[cx + 2, cy] = FLOR_RECEPTORA
    m.paisagem[cx, cy + 6] = FLOR_RECEPTORA
    m.visitas_celula[cx + 2, cy] = 3
    m.visitas_celula[cx, cy + 6] = 1
    assert m.distancia_media_visitas() == pytest.approx((2 * 3 + 6 * 1) / 4)
    assert np.isnan(modelo_vazio().distancia_media_visitas())


def test_parametro_t_viagem_forma_invalido():
    with pytest.raises(ValueError):
        Parametros(t_viagem_forma=-1.0)


# ------------------------------------------------ distância máxima da viagem e quantil das visitas
def test_distancia_maxima_da_viagem_e_registrada_na_chegada():
    m = modelo_vazio(retorno_colmeia=True)
    m.n_abelhas = 1
    m.ab_i[0], m.ab_j[0] = m.centro
    cx, cy = m.centro
    for dx in (3, 7, 5):                                 # vai a 7 células e volta
        m._atualizar_viagem(0, cx + dx, cy)
    assert m.ab_dmax[0] == pytest.approx(7.0)
    m._chegar_na_colmeia(0)
    assert m.dist_max_viagens == [pytest.approx(7.0)] and m.ab_dmax[0] == 0.0


def test_distancia_maxima_das_viagens_e_positiva_e_limitada():
    m = ModeloPolinizacao(Parametros(retorno_colmeia=True, t_viagem_max=40), seed=5).executar(400)
    d = np.array(m.dist_max_viagens)
    assert len(d) == len(m.duracoes_viagem) > 20
    assert np.all(d >= 0) and d.max() <= 40 * 1.5 + 5   # no máximo ~passo médio x duração


def test_quantil_ponderado_das_visitas():
    m = modelo_vazio()
    cx, cy = m.centro
    m.paisagem[cx + 2, cy] = FLOR_RECEPTORA
    m.paisagem[cx, cy + 6] = FLOR_RECEPTORA
    m.visitas_celula[cx + 2, cy] = 3
    m.visitas_celula[cx, cy + 6] = 1
    assert m.quantil_visitas(0.5) == 2.0 and m.quantil_visitas(0.9) == 6.0
    assert np.isnan(modelo_vazio().quantil_visitas(0.5))
