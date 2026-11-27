"""ABM de referência em espaço contínuo (Limitação 1 do artigo).

Compara o formalismo do autômato celular (CA) com um modelo baseado em agentes (ABM) em que as
abelhas têm posição contínua. As duas versões compartilham:

* a **mesma paisagem** (para a mesma semente é idêntica, pois é gerada pelo mesmo código);
* as **mesmas regras de interação** (coleta, deposição, retenção, recarga, geração de abelhas),
  herdadas de :class:`ModeloPolinizacao`;
* a **mesma difusão**: o passo livre do CA (Moore) tem deslocamento quadrático médio de 1,5 célula²;
  o ABM dá passos de comprimento ``sqrt(1,5)`` em direção uniforme, com o mesmo deslocamento quadrático.

Diferenças deliberadas (e que o artigo discute):

* A posição é contínua; a célula visitada é a que contém o ponto (centros em coordenadas inteiras).
* A busca de flores usa uma árvore k-d (``cKDTree``) com raio de percepção ``raio_percepcao``
  (padrão 1,5 célula, que cobre a vizinhança de Moore).
* Não há exclusão de células entre abelhas (no ABM contínuo não existe "célula ocupada").
* Obstáculos e colmeia bloqueiam o ponto de chegada; direções bloqueadas são sorteadas de novo
  (até 8 vezes), como o CA exclui vizinhos de peso 0.

Só o comportamento-base é suportado (sem retorno à colmeia, mortalidade etc.); a paisagem pode
usar as extensões de fragmentação.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .modelo import (
    COLMEIA, FLOR_DOADORA, FLOR_RECEPTORA, NEUTRO, POLINIZADA, ModeloPolinizacao, Parametros,
)

PASSO_DIFUSAO = float(np.sqrt(1.5))   # raiz do deslocamento quadrático médio de um passo de Moore


class ModeloABMContinuo(ModeloPolinizacao):
    """Mesma paisagem e regras de interação do CA, com movimento em espaço contínuo."""

    def __init__(self, params: Parametros | None = None, seed=None, raio_percepcao: float = 1.5) -> None:
        super().__init__(params, seed)
        p = self.p
        if (p.retorno_colmeia or p.dist_max_voo > 0 or p.mortalidade > 0 or p.memoria_vies > 0
                or p.deposicao_acumulada or p.toroidal
                or p.expoente_deposicao != 0.0):
            raise ValueError("o ABM contínuo só suporta o comportamento-base (sem extensões de biologia, "
                             "sem grade toroidal)")
        if raio_percepcao <= 0:
            raise ValueError("raio_percepcao deve ser > 0")
        self.raio_percepcao = float(raio_percepcao)
        self.px = np.zeros(p.max_abelhas, dtype=float)
        self.py = np.zeros(p.max_abelhas, dtype=float)
        # Árvore k-d das flores iniciais (o estado e o peso mudam, as posições não).
        fi, fj = np.nonzero(np.isin(self.paisagem, (FLOR_DOADORA, FLOR_RECEPTORA)))
        self._flores = np.column_stack([fi, fj]).astype(float)
        self._arvore = cKDTree(self._flores) if len(fi) else None

    # ------------------------------------------------------------------ dinâmica
    def _gerar_abelha(self) -> None:
        if self.n_abelhas < self.p.max_abelhas and self.rng.random() < self.p.prob_geracao:
            b = self.n_abelhas
            cx, cy = self.centro
            self.px[b], self.py[b] = cx, cy
            self.ab_i[b], self.ab_j[b] = cx, cy
            self.ab_carga[b] = 0.0
            self.ab_sem_flor[b] = 0
            self.n_abelhas += 1

    def _livre(self, x: float, y: float) -> bool:
        i, j = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        if not (0 <= i < self.p.linhas and 0 <= j < self.p.colunas):
            return False
        return self._peso[i, j] > 0.0 or self.paisagem[i, j] == NEUTRO

    def _mover(self, b: int) -> None:
        """Escolhe flor próxima (peso alpha) ou passo livre, com a mesma razão de pesos do CA."""
        x, y = self.px[b], self.py[b]
        ci, cj = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        alvos, pesos = [], []
        if self._arvore is not None:
            for k in self._arvore.query_ball_point([x, y], self.raio_percepcao):
                fi, fj = int(self._flores[k, 0]), int(self._flores[k, 1])
                if (fi, fj) != (ci, cj) and self._peso[fi, fj] > 0.0:
                    alvos.append((fi, fj))
                    pesos.append(self._peso[fi, fj])
        livres = max(0.0, 8.0 - len(alvos))       # os 8 vizinhos de Moore: flores com peso alpha, o resto 1
        total = sum(pesos) + livres
        u = self.rng.random() * total
        if u < sum(pesos):                        # vai a uma flor (peso proporcional)
            acum = 0.0
            for (fi, fj), w in zip(alvos, pesos):
                acum += w
                if u < acum:
                    self.px[b], self.py[b] = fi, fj
                    return
        for _ in range(8):                        # passo livre de comprimento sqrt(1,5)
            ang = self.rng.random() * 2.0 * np.pi
            nx, ny = x + PASSO_DIFUSAO * np.cos(ang), y + PASSO_DIFUSAO * np.sin(ang)
            if self._livre(nx, ny):
                self.px[b], self.py[b] = nx, ny
                return                            # sem direção livre: fica parada

    def passo(self) -> None:
        self.passo_atual += 1
        self._gerar_abelha()
        self._recarregar_flores()
        visitas = 0
        deposicao_antes = self.deposicao_total
        for b in self.rng.permutation(self.n_abelhas):
            self._mover(b)
            i, j = int(np.floor(self.px[b] + 0.5)), int(np.floor(self.py[b] + 0.5))
            self.ab_i[b], self.ab_j[b] = i, j
            visitas += self._interagir(b, i, j)
        self.hist_polinizadas.append(self.n_polinizadas)
        self.hist_visitas.append(visitas)
        self.hist_deposicao.append(self.deposicao_total - deposicao_antes)
        self.hist_abelhas.append(self.n_abelhas)


__all__ = ["ModeloABMContinuo", "PASSO_DIFUSAO", "COLMEIA", "POLINIZADA", "FLOR_RECEPTORA"]
