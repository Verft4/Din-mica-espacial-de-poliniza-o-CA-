"""Dinâmica espacial de polinização e fluxo de pólen com autômatos celulares."""

from .analise import ajustar_decaimento, ajustar_exponencial, ic95, media_dp, raio_eficaz, sd_binomial_esperado
from .modelo import (
    ABELHA,
    COLMEIA,
    FLOR_DOADORA,
    FLOR_RECEPTORA,
    NEUTRO,
    POLINIZADA,
    VAZIO,
    ModeloPolinizacao,
    Parametros,
)
from .monte_carlo import ResultadoMC, executar_monte_carlo, varrer_parametro

__all__ = [
    "ModeloPolinizacao", "Parametros", "ResultadoMC",
    "executar_monte_carlo", "varrer_parametro",
    "ajustar_decaimento", "ajustar_exponencial", "ic95", "media_dp", "raio_eficaz", "sd_binomial_esperado",
    "VAZIO", "NEUTRO", "FLOR_DOADORA", "FLOR_RECEPTORA", "COLMEIA", "ABELHA", "POLINIZADA",
]
__version__ = "1.0.0"
