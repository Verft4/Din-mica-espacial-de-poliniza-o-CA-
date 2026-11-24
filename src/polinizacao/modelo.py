"""Modelo de polinização por autômato celular com entidades móveis.

O espaço é uma grade 2D de células (camada estática da paisagem). As abelhas
são entidades móveis numa camada dinâmica separada (posição, carga de pólen e
tempo desde a última visita a uma flor). A vizinhança é a de Moore (8 células).

Estados da paisagem (conjunto Omega = {0, ..., 6} do artigo):

    0 VAZIO       célula sem recurso / obstáculo (intransponível)
    1 NEUTRO      célula transitável sem flor
    2 DOADORA     flor com estoque de pólen (recarrega após ``t_recarga``)
    3 RECEPTORA   flor que pode ser polinizada
    4 COLMEIA     origem das abelhas (peso de movimento 0)
    5 ABELHA      estado móvel; vive na camada dinâmica, não na paisagem
    6 POLINIZADA  receptora que já foi polinizada

Ordem de um passo de tempo (seção 4.4 do artigo):
geração de abelhas -> recarga floral -> movimento/interação -> métricas.

Garantias (verificadas em ``tests/``): o número de abelhas só aumenta pela
geração na colmeia; nunca há duas abelhas na mesma célula; ``0 <= carga <=
cap_max``; o número de flores polinizadas nunca diminui.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import gaussian_filter

VAZIO = 0
NEUTRO = 1
FLOR_DOADORA = 2
FLOR_RECEPTORA = 3
COLMEIA = 4
ABELHA = 5
POLINIZADA = 6

NOMES_ESTADOS = {
    VAZIO: "Obstáculo / vazio",
    NEUTRO: "Neutro",
    FLOR_DOADORA: "Flor doadora",
    FLOR_RECEPTORA: "Flor receptora",
    COLMEIA: "Colmeia",
    ABELHA: "Abelha",
    POLINIZADA: "Polinizada",
}

_DESLOCAMENTOS = np.array([-1, 0, 1])


@dataclass(frozen=True)
class Parametros:
    """Parâmetros do modelo. Os padrões reproduzem o cenário-base do artigo."""

    # --- Paisagem ---
    linhas: int = 60
    colunas: int = 60
    densidade_floral: float = 0.15   # fração das células livres (não obstáculo) com flor
    frac_doadoras: float = 0.5       # fração das flores que são doadoras
    frac_obstaculos: float = 0.05    # fração de todas as células que são obstáculo
    agregacao_obstaculos: float = 0.0  # sigma do filtro gaussiano; 0 = obstáculos independentes
    # --- Movimento ---
    alpha_atracao: float = 12.0      # peso de células de flor (neutra = 1); alpha=1 => passeio aleatório
    peso_polinizada: float = 1.0     # peso de flores já polinizadas
    # --- Interação ---
    cap_max: float = 40.0            # C_max: carga máxima de pólen da abelha
    eficiencia_coleta: float = 8.0   # tau_collect: pólen coletado por visita
    estoque_max: float = 40.0        # estoque inicial/máximo de uma flor doadora
    taxa_deposicao: float = 0.6      # tau_depose: fração da carga depositada por visita
    limiar_polinizacao: float = 4.0  # deposição mínima para polinizar uma receptora
    t_retencao: int = 20             # Delta t_max: passos sem visitar flor até perder a carga
    t_recarga: int = 50              # Delta t_recarga: passos para a doadora regenerar
    # --- Colônia ---
    max_abelhas: int = 60
    prob_geracao: float = 0.3        # prob. de gerar uma abelha por passo (se colmeia livre)
    # --- Fronteira ---
    toroidal: bool = False           # False: bordas fechadas; True: grade periódica

    def __post_init__(self) -> None:
        if self.linhas < 3 or self.colunas < 3:
            raise ValueError("a grade precisa ter pelo menos 3x3 células")
        for nome in ("densidade_floral", "frac_doadoras", "frac_obstaculos",
                     "taxa_deposicao", "prob_geracao"):
            valor = getattr(self, nome)
            if not 0.0 <= valor <= 1.0:
                raise ValueError(f"{nome} deve estar em [0, 1] (recebido {valor})")
        if self.alpha_atracao < 0 or self.peso_polinizada < 0:
            raise ValueError("pesos de atração devem ser >= 0")
        for nome in ("cap_max", "eficiencia_coleta", "estoque_max", "limiar_polinizacao"):
            if getattr(self, nome) <= 0:
                raise ValueError(f"{nome} deve ser > 0")
        if self.max_abelhas < 0 or self.t_retencao < 1 or self.t_recarga < 1:
            raise ValueError("max_abelhas >= 0, t_retencao >= 1 e t_recarga >= 1")
        if self.agregacao_obstaculos < 0:
            raise ValueError("agregacao_obstaculos deve ser >= 0")

    def com(self, **mudancas) -> "Parametros":
        """Cópia dos parâmetros com alguns valores alterados."""
        return replace(self, **mudancas)


class ModeloPolinizacao:
    """Simulação de uma colmeia polinizando uma paisagem heterogênea."""

    def __init__(self, params: Parametros | None = None, seed=None) -> None:
        self.p = params if params is not None else Parametros()
        self.rng = np.random.default_rng(seed)
        forma = (self.p.linhas, self.p.colunas)

        self.paisagem = np.full(forma, NEUTRO, dtype=np.int8)
        self.estoque = np.zeros(forma, dtype=float)
        self.timer_recarga = np.zeros(forma, dtype=int)
        self.ocupacao = np.zeros(forma, dtype=bool)

        n = self.p.max_abelhas
        self.ab_i = np.zeros(n, dtype=int)
        self.ab_j = np.zeros(n, dtype=int)
        self.ab_carga = np.zeros(n, dtype=float)
        self.ab_sem_flor = np.zeros(n, dtype=int)
        self.n_abelhas = 0

        self.passo_atual = 0
        self.n_polinizadas = 0
        self.deposicao_total = 0.0
        self.hist_polinizadas: list[int] = []
        self.hist_visitas: list[int] = []
        self.hist_deposicao: list[float] = []
        self.hist_abelhas: list[int] = []

        self.centro = (self.p.linhas // 2, self.p.colunas // 2)
        self._inicializar_paisagem()
        self._peso = self._tabela_pesos()[self.paisagem]
        self._distancias = self._calcular_distancias()
        self.total_receptoras = int(np.sum(self.paisagem == FLOR_RECEPTORA))

    # ------------------------------------------------------------------ setup
    def _inicializar_paisagem(self) -> None:
        p, rng = self.p, self.rng
        forma = self.paisagem.shape

        
        campo = rng.normal(size=forma)
        if p.agregacao_obstaculos > 0:
            campo = gaussian_filter(campo, sigma=p.agregacao_obstaculos, mode="wrap")
        if p.frac_obstaculos > 0:
            limite = np.quantile(campo, p.frac_obstaculos)
            obstaculo = campo <= limite
        else:
            obstaculo = np.zeros(forma, dtype=bool)

        # Flores: Bernoulli(densidade) sobre as células livres.
        flor = (rng.random(forma) < p.densidade_floral) & ~obstaculo
        doadora = rng.random(forma) < p.frac_doadoras

        self.paisagem[obstaculo] = VAZIO
        self.paisagem[flor & doadora] = FLOR_DOADORA
        self.paisagem[flor & ~doadora] = FLOR_RECEPTORA
        self.paisagem[self.centro] = COLMEIA
        self.estoque[self.paisagem == FLOR_DOADORA] = p.estoque_max

    def _tabela_pesos(self) -> np.ndarray:
        """w(S): atratividade de uma célula vizinha segundo o seu estado."""
        tabela = np.zeros(7, dtype=float)
        tabela[NEUTRO] = 1.0
        tabela[FLOR_DOADORA] = self.p.alpha_atracao
        tabela[FLOR_RECEPTORA] = self.p.alpha_atracao
        tabela[POLINIZADA] = self.p.peso_polinizada
        # VAZIO, COLMEIA e ABELHA ficam com peso 0 (bloqueados / ocupados).
        return tabela

    def _calcular_distancias(self) -> np.ndarray:
        cx, cy = self.centro
        i, j = np.indices(self.paisagem.shape)
        di = np.abs(i - cx)
        dj = np.abs(j - cy)
        if self.p.toroidal:
            di = np.minimum(di, self.p.linhas - di)
            dj = np.minimum(dj, self.p.colunas - dj)
        return np.hypot(di, dj)

    # -------------------------------------------------------------- dinâmica
    def _janela_pesos(self, i: int, j: int):
        """Linhas, colunas e pesos (3x3) da vizinhança de Moore de (i, j).

        O peso da célula central é 0, células fora da grade (bordas fechadas),
        obstáculos, colmeia e células ocupadas por outra abelha têm peso 0.
        """
        linhas = i + _DESLOCAMENTOS
        colunas = j + _DESLOCAMENTOS
        if self.p.toroidal:
            linhas = linhas % self.p.linhas
            colunas = colunas % self.p.colunas
            validas = np.ones((3, 3), dtype=bool)
        else:
            vl = (linhas >= 0) & (linhas < self.p.linhas)
            vc = (colunas >= 0) & (colunas < self.p.colunas)
            validas = vl[:, None] & vc[None, :]
            linhas = np.clip(linhas, 0, self.p.linhas - 1)
            colunas = np.clip(colunas, 0, self.p.colunas - 1)
        sub = np.ix_(linhas, colunas)
        pesos = self._peso[sub] * (~self.ocupacao[sub]) * validas
        pesos[1, 1] = 0.0
        return linhas, colunas, pesos

    def _escolher_destino(self, i: int, j: int) -> tuple[int, int]:
        """Sorteia a próxima célula com P = w / soma(w). Sem opção, fica."""
        linhas, colunas, pesos = self._janela_pesos(i, j)
        plano = pesos.ravel()
        acumulado = np.cumsum(plano)
        total = acumulado[-1]
        if total <= 0.0:
            return i, j
        k = int(np.searchsorted(acumulado, self.rng.random() * total, side="right"))
        if k > 8 or plano[k] <= 0.0:  # salvaguarda contra arredondamento
            k = int(np.argmax(plano))
        return int(linhas[k // 3]), int(colunas[k % 3])

    def _interagir(self, b: int, x: int, y: int) -> bool:
        """Aplica coleta/deposição na célula (x, y). Retorna se visitou flor."""
        p = self.p
        estado = self.paisagem[x, y]
        visitou = estado in (FLOR_DOADORA, FLOR_RECEPTORA, POLINIZADA)

        if estado == FLOR_DOADORA:
            coleta = min(p.eficiencia_coleta, self.estoque[x, y], p.cap_max - self.ab_carga[b])
            if coleta > 0.0:
                self.ab_carga[b] = min(p.cap_max, self.ab_carga[b] + coleta)
                self.estoque[x, y] -= coleta
        elif estado == FLOR_RECEPTORA:
            deposicao = self.ab_carga[b] * p.taxa_deposicao
            if deposicao > 0.0:
                self.ab_carga[b] -= deposicao
                self.deposicao_total += deposicao
                if deposicao >= p.limiar_polinizacao:
                    self.paisagem[x, y] = POLINIZADA
                    self._peso[x, y] = p.peso_polinizada
                    self.n_polinizadas += 1

        # Tempo de retenção: sem visitar flor por t_retencao passos => perde a carga.
        if visitou:
            self.ab_sem_flor[b] = 0
        else:
            self.ab_sem_flor[b] += 1
            if self.ab_sem_flor[b] >= p.t_retencao:
                self.ab_carga[b] = 0.0
                self.ab_sem_flor[b] = 0
        return visitou

    def _gerar_abelha(self) -> None:
        cx, cy = self.centro
        if (self.n_abelhas < self.p.max_abelhas and not self.ocupacao[cx, cy]
                and self.rng.random() < self.p.prob_geracao):
            b = self.n_abelhas
            self.ab_i[b], self.ab_j[b] = cx, cy
            self.ab_carga[b] = 0.0
            self.ab_sem_flor[b] = 0
            self.ocupacao[cx, cy] = True
            self.n_abelhas += 1

    def _recarregar_flores(self) -> None:
        esgotadas = (self.paisagem == FLOR_DOADORA) & (self.estoque < self.p.estoque_max)
        self.timer_recarga[esgotadas] += 1
        pronto = self.timer_recarga >= self.p.t_recarga
        self.estoque[pronto] = self.p.estoque_max
        self.timer_recarga[pronto] = 0

    def passo(self) -> None:
        """Avança um passo de tempo."""
        self.passo_atual += 1
        self._gerar_abelha()
        self._recarregar_flores()

        visitas = 0
        deposicao_antes = self.deposicao_total
        # Ordem aleatória, atualização sequencial: uma abelha só vai para uma
        # célula livre *no momento* da sua vez, então nenhuma abelha se perde.
        for b in self.rng.permutation(self.n_abelhas):
            i, j = int(self.ab_i[b]), int(self.ab_j[b])
            x, y = self._escolher_destino(i, j)
            if (x, y) != (i, j):
                self.ocupacao[i, j] = False
                self.ocupacao[x, y] = True
                self.ab_i[b], self.ab_j[b] = x, y
            visitas += self._interagir(b, x, y)

        self.hist_polinizadas.append(self.n_polinizadas)
        self.hist_visitas.append(visitas)
        self.hist_deposicao.append(self.deposicao_total - deposicao_antes)
        self.hist_abelhas.append(self.n_abelhas)

    def executar(self, passos: int) -> "ModeloPolinizacao":
        for _ in range(passos):
            self.passo()
        return self

    # --------------------------------------------------------------- métricas
    @property
    def taxa_polinizacao(self) -> float:
        """Fração das flores receptoras (iniciais) que foi polinizada."""
        if self.total_receptoras == 0:
            return float("nan")
        return self.n_polinizadas / self.total_receptoras

    def eficiencia_por_distancia(self, largura_bin: float = 2.0):
        """Eficiência de polinização (%) por faixa de distância à colmeia.

        Eficiência = receptoras polinizadas / receptoras totais (receptoras +
        polinizadas) em cada faixa. Doadoras NÃO entram na conta.

        Retorna (centros, eficiencia, n_receptoras); a eficiência é NaN em
        faixas sem nenhuma receptora.
        """
        if largura_bin <= 0:
            raise ValueError("largura_bin deve ser > 0")
        receptora = (self.paisagem == FLOR_RECEPTORA) | (self.paisagem == POLINIZADA)
        polinizada = self.paisagem == POLINIZADA
        faixas = np.floor(self._distancias / largura_bin).astype(int)
        n_faixas = int(faixas.max()) + 1
        total = np.bincount(faixas[receptora], minlength=n_faixas)
        feitas = np.bincount(faixas[polinizada], minlength=n_faixas)
        with np.errstate(invalid="ignore", divide="ignore"):
            eficiencia = np.where(total > 0, 100.0 * feitas / total, np.nan)
        centros = (np.arange(n_faixas) + 0.5) * largura_bin
        return centros, eficiencia, total

    def raio_percentil(self, q: float = 0.95) -> float:
        """Distância que contém a fração ``q`` das flores polinizadas (NaN se nenhuma)."""
        d = self._distancias[self.paisagem == POLINIZADA]
        if d.size == 0:
            return float("nan")
        return float(np.quantile(d, q))

    def mapa_com_abelhas(self) -> np.ndarray:
        """Cópia da paisagem com as abelhas (estado 5) sobrepostas, para desenho."""
        mapa = self.paisagem.copy()
        mapa[self.ab_i[: self.n_abelhas], self.ab_j[: self.n_abelhas]] = ABELHA
        return mapa
