"""Regressão: o comportamento do cenário-base NÃO pode mudar quando as extensões estão desligadas.

Valores gravados a partir da versão 1.0.0 (antes das extensões de biologia, fragmentação, ABM e calibração).
Se algum destes testes falhar, uma extensão alterou o modelo-base ou o consumo do gerador aleatório.
"""
import hashlib

import numpy as np

from polinizacao import ModeloPolinizacao, Parametros, executar_monte_carlo


def _hash(m):
    return hashlib.sha256(m.paisagem.tobytes() + m.ab_i[:m.n_abelhas].tobytes()
                          + m.ab_j[:m.n_abelhas].tobytes()).hexdigest()[:16]


def test_base_seed42_identico_a_v1():
    m = ModeloPolinizacao(Parametros(), seed=42).executar(120)
    assert m.n_polinizadas == 47
    assert m.n_abelhas == 40
    assert m.total_receptoras == 254
    assert round(m.deposicao_total, 6) == 697.552896
    assert _hash(m) == "f30e05d7e548ae02"


def test_monte_carlo_base_identico_a_v1():
    r = executar_monte_carlo(Parametros(), n_simulacoes=5, passos=50, seed=3)
    assert r.hist_polinizadas[:, -1].tolist() == [7.0, 9.0, 14.0, 6.0, 7.0]
    assert round(float(r.taxa_final.mean()), 8) == 0.03242627


def test_cenario_alternativo_identico_a_v1():
    p = Parametros(alpha_atracao=4.0, densidade_floral=0.25, frac_obstaculos=0.15, agregacao_obstaculos=2.0)
    m = ModeloPolinizacao(p, seed=7).executar(80)
    assert m.n_polinizadas == 42
    assert _hash(m) == "b77b5c7b9a0df34f"
