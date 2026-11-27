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

Garantias (verificadas em ``tests/``): sem mortalidade, o número de abelhas só
aumenta pela geração na colmeia; nunca há duas abelhas na mesma célula;
``0 <= carga <= cap_max``; o número de flores polinizadas nunca diminui.

Extensões (todas DESLIGADAS por padrão; com os padrões o modelo é idêntico à v1.0.0,
o que é garantido por ``tests/test_regressao_base.py``):

* Biologia: viagens de ida e volta à colmeia (``retorno_colmeia``), distância máxima
  de voo, mortalidade, fidelidade a manchas (``memoria_vies``), pesos de atração por
  tipo de flor e deposição acumulada de pólen nas receptoras.
* Fragmentação: barreira em anel com corredores, blocos extensos de obstáculos e
  flores agrupadas em manchas.
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
    # --- Extensão: fragmentação da paisagem (0/desligado = modelo-base) ---
    agregacao_flores: float = 0.0    # sigma; 0 = flores independentes; >0 = flores em manchas
    n_blocos: int = 0                # blocos quadrados de obstáculos (além de frac_obstaculos)
    tamanho_bloco: int = 6           # lado de cada bloco (células)
    barreira_raio: float = 0.0       # raio interno do anel-barreira em torno da colmeia; 0 = sem anel
    barreira_espessura: float = 2.0  # espessura do anel (>= 2 evita vazamento pela diagonal)
    barreira_aberturas: int = 0      # número de corredores abertos no anel
    barreira_largura_abertura: float = 3.0  # largura de cada corredor (células)
    # --- Extensão: biologia ---
    retorno_colmeia: bool = False    # abelhas fazem viagens: saem, forrageiam, voltam e descansam
    t_viagem_max: int = 60           # passos fora da colmeia até iniciar o retorno (média, se forma > 0)
    t_viagem_forma: float = 0.0      # 0 = orçamento fixo; k > 0 = orçamento ~ Gamma(k) com média t_viagem_max
    fracao_carga_retorno: float = 0.9  # retorna quando a carga atinge esta fração de cap_max
    t_descanso: int = 5              # passos de repouso na colmeia entre viagens
    dist_max_voo: float = 0.0        # distância máxima da colmeia (células); 0 = sem limite
    mortalidade: float = 0.0         # probabilidade de morte por passo (a colmeia repõe)
    memoria_vies: float = 0.0        # beta: viés rumo à última doadora com estoque (0 = sem memória)
    peso_doadora: float | None = None    # atração de doadoras; None = alpha_atracao
    peso_receptora: float | None = None  # atração de receptoras; None = alpha_atracao
    deposicao_acumulada: bool = False    # pólen depositado em uma receptora acumula entre visitas
    tamanho_relativo: float = 1.0        # tamanho do corpo relativo a Apis (ITD); 1 = Apis
    expoente_deposicao: float = 0.0      # b: taxa efetiva = min(1, taxa_deposicao * tamanho_relativo**b); 0 = desligado

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
        if self.agregacao_obstaculos < 0 or self.agregacao_flores < 0:
            raise ValueError("agregacao_obstaculos e agregacao_flores devem ser >= 0")
        if self.n_blocos < 0 or self.tamanho_bloco < 1:
            raise ValueError("n_blocos >= 0 e tamanho_bloco >= 1")
        if self.barreira_raio < 0 or self.barreira_aberturas < 0:
            raise ValueError("barreira_raio e barreira_aberturas devem ser >= 0")
        if 0 < self.barreira_raio < 3:
            raise ValueError("barreira_raio deve ser 0 (sem barreira) ou >= 3")
        if self.barreira_espessura <= 0 or self.barreira_largura_abertura <= 0:
            raise ValueError("espessura e largura de abertura devem ser > 0")
        if self.t_viagem_forma < 0:
            raise ValueError("t_viagem_forma deve ser >= 0")
        if self.t_viagem_max < 1 or self.t_descanso < 0:
            raise ValueError("t_viagem_max >= 1 e t_descanso >= 0")
        if not 0.0 < self.fracao_carga_retorno <= 1.0:
            raise ValueError("fracao_carga_retorno deve estar em (0, 1]")
        if not 0.0 <= self.mortalidade <= 1.0:
            raise ValueError("mortalidade deve estar em [0, 1]")
        if self.tamanho_relativo <= 0:
            raise ValueError("tamanho_relativo deve ser > 0")
        if self.dist_max_voo < 0 or self.memoria_vies < 0:
            raise ValueError("dist_max_voo e memoria_vies devem ser >= 0")
        for nome in ("peso_doadora", "peso_receptora"):
            valor = getattr(self, nome)
            if valor is not None and valor < 0:
                raise ValueError(f"{nome} deve ser >= 0")

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
        self.ab_viagem = np.zeros(n, dtype=int)       # passos desde a saída da colmeia
        self.ab_retorno = np.zeros(n, dtype=bool)     # True: voltando para a colmeia
        self.ab_descanso = np.zeros(n, dtype=int)     # passos restantes de repouso
        self.ab_dentro = np.zeros(n, dtype=bool)      # True: dentro da colmeia (não ocupa a célula)
        self.ab_orcamento = np.full(n, float(self.p.t_viagem_max))  # duração da viagem atual
        self.ab_dmax = np.zeros(n, dtype=float)       # maior distância à colmeia na viagem atual (células)
        self.ab_mem_i = np.full(n, -1, dtype=int)     # memória da última doadora com estoque
        self.ab_mem_j = np.full(n, -1, dtype=int)
        self.n_abelhas = 0
        self.n_mortes = 0
        self.duracoes_viagem: list[int] = []
        self.dist_max_viagens: list[float] = []     # distância máxima de cada viagem concluída (células)
        self.visitas_celula = np.zeros(forma, dtype=int)   # visitas acumuladas por célula
        self.dose = np.zeros(forma, dtype=float)           # pólen acumulado por receptora
        self._modo_estendido = (self.p.retorno_colmeia or self.p.dist_max_voo > 0
                                or self.p.memoria_vies > 0)

        self.passo_atual = 0
        self.n_polinizadas = 0
        self.deposicao_total = 0.0
        self._taxa_dep = min(1.0, self.p.taxa_deposicao * self.p.tamanho_relativo ** self.p.expoente_deposicao)
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

        # Obstáculos: campo gaussiano (suavizado se agregacao > 0) limiarizado
        # no quantil exato, para que a fração seja a pedida e o agrupamento
        # (fragmentação) seja controlável de forma independente.
        campo = rng.normal(size=forma)
        if p.agregacao_obstaculos > 0:
            campo = gaussian_filter(campo, sigma=p.agregacao_obstaculos, mode="wrap")
        if p.frac_obstaculos > 0:
            limite = np.quantile(campo, p.frac_obstaculos)
            obstaculo = campo <= limite
        else:
            obstaculo = np.zeros(forma, dtype=bool)

        # Extensões de fragmentação (só consomem números aleatórios quando ligadas).
        if p.barreira_raio > 0:
            obstaculo |= self._mascara_barreira()
        if p.n_blocos > 0:
            obstaculo |= self._mascara_blocos()

        # Flores: Bernoulli(densidade) sobre as células livres, ou manchas.
        if p.agregacao_flores > 0:
            flor = self._flores_agrupadas(~obstaculo)
        else:
            flor = (rng.random(forma) < p.densidade_floral) & ~obstaculo
        doadora = rng.random(forma) < p.frac_doadoras

        self.paisagem[obstaculo] = VAZIO
        self.paisagem[flor & doadora] = FLOR_DOADORA
        self.paisagem[flor & ~doadora] = FLOR_RECEPTORA
        self.paisagem[self.centro] = COLMEIA
        self.estoque[self.paisagem == FLOR_DOADORA] = p.estoque_max

    def _mascara_barreira(self) -> np.ndarray:
        """Anel de obstáculos em torno da colmeia, com corredores equiespaçados.

        Ocupa as células com ``raio <= r < raio + espessura``. Com espessura >= 2 o anel
        é fechado para a vizinhança de Moore (nenhuma abelha atravessa na diagonal).
        Cada corredor tem largura ``barreira_largura_abertura`` medida em comprimento de arco.
        """
        p = self.p
        i, j = np.indices(self.paisagem.shape)
        dx, dy = i - self.centro[0], j - self.centro[1]
        r = np.hypot(dx, dy)
        anel = (r >= p.barreira_raio) & (r < p.barreira_raio + p.barreira_espessura)
        if p.barreira_aberturas > 0:
            angulo = np.arctan2(dy, dx)
            for k in range(p.barreira_aberturas):
                alvo = 2.0 * np.pi * k / p.barreira_aberturas
                dif = np.abs((angulo - alvo + np.pi) % (2.0 * np.pi) - np.pi)
                anel &= ~(dif * r < p.barreira_largura_abertura / 2.0)
        return anel

    def _mascara_blocos(self) -> np.ndarray:
        """Blocos quadrados de obstáculos em posições aleatórias (podem se sobrepor)."""
        p = self.p
        lado = min(p.tamanho_bloco, p.linhas, p.colunas)
        mascara = np.zeros(self.paisagem.shape, dtype=bool)
        for _ in range(p.n_blocos):
            i0 = int(self.rng.integers(0, p.linhas - lado + 1))
            j0 = int(self.rng.integers(0, p.colunas - lado + 1))
            mascara[i0:i0 + lado, j0:j0 + lado] = True
        cx, cy = self.centro
        mascara[cx - 1:cx + 2, cy - 1:cy + 2] = False   # a colmeia e seu entorno ficam livres
        return mascara

    def _flores_agrupadas(self, livres: np.ndarray) -> np.ndarray:
        """Flores em manchas: as ``densidade * n_livres`` células de maior valor de um campo suavizado.

        Mantém exatamente a mesma densidade total das flores independentes, mas agrupadas.
        """
        p = self.p
        campo = gaussian_filter(self.rng.normal(size=livres.shape), sigma=p.agregacao_flores, mode="wrap")
        n_flores = int(round(p.densidade_floral * int(livres.sum())))
        flor = np.zeros(livres.shape, dtype=bool)
        if n_flores > 0:
            valores = np.where(livres, campo, -np.inf).ravel()
            idx = np.argpartition(valores, -n_flores)[-n_flores:]
            flor.ravel()[idx] = True
        return flor

    def _tabela_pesos(self) -> np.ndarray:
        """w(S): atratividade de uma célula vizinha segundo o seu estado."""
        tabela = np.zeros(7, dtype=float)
        tabela[NEUTRO] = 1.0
        tabela[FLOR_DOADORA] = (self.p.alpha_atracao if self.p.peso_doadora is None
                                else self.p.peso_doadora)
        tabela[FLOR_RECEPTORA] = (self.p.alpha_atracao if self.p.peso_receptora is None
                                  else self.p.peso_receptora)
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

    def _ajustar_pesos(self, b: int, i: int, j: int, linhas, colunas, pesos):
        """Pesos de movimento das extensões de biologia.

        Retorna ``(destino_direto, pesos)``; ``destino_direto`` não é None quando a abelha em
        retorno está ao lado da colmeia livre e entra nela.
        """
        p = self.p
        d_viz = self._distancias[np.ix_(linhas, colunas)]
        d_atual = self._distancias[i, j]

        if p.retorno_colmeia and self.ab_retorno[b]:
            cx, cy = self.centro
            if (abs(i - cx) <= 1 and abs(j - cy) <= 1 and not self.ocupacao[cx, cy]):
                return (cx, cy), pesos
            validos = pesos > 0
            aproxima = validos & (d_viz < d_atual)
            novos = aproxima.astype(float)
            if not aproxima.any():       # sem caminho que se aproxime: qualquer movimento válido
                novos = validos.astype(float)
            return None, novos

        if p.dist_max_voo > 0:
            pesos = pesos * (d_viz <= p.dist_max_voo)
        if p.memoria_vies > 0 and self.ab_mem_i[b] >= 0:
            mi, mj = int(self.ab_mem_i[b]), int(self.ab_mem_j[b])
            d_alvo_viz = np.hypot(linhas[:, None] - mi, colunas[None, :] - mj)
            d_alvo_atual = np.hypot(i - mi, j - mj)
            pesos = pesos * np.where(d_alvo_viz < d_alvo_atual, 1.0 + p.memoria_vies, 1.0)
        return None, pesos

    def _escolher_destino(self, i: int, j: int, b: int | None = None) -> tuple[int, int]:
        """Sorteia a próxima célula com P = w / soma(w). Sem opção, fica.

        ``b`` (índice da abelha) só é necessário para as extensões de biologia.
        """
        linhas, colunas, pesos = self._janela_pesos(i, j)
        if b is not None and self._modo_estendido:
            direto, pesos = self._ajustar_pesos(b, i, j, linhas, colunas, pesos)
            if direto is not None:
                return direto
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

        if visitou:
            self.visitas_celula[x, y] += 1

        if estado == FLOR_DOADORA:
            coleta = min(p.eficiencia_coleta, self.estoque[x, y], p.cap_max - self.ab_carga[b])
            if coleta > 0.0:
                self.ab_carga[b] = min(p.cap_max, self.ab_carga[b] + coleta)
                self.estoque[x, y] -= coleta
                if p.memoria_vies > 0:
                    self.ab_mem_i[b], self.ab_mem_j[b] = x, y      # patch recompensador
            if p.memoria_vies > 0 and self.estoque[x, y] <= 0.0:
                self.ab_mem_i[b], self.ab_mem_j[b] = -1, -1        # patch esgotado: esquece
        elif estado == FLOR_RECEPTORA:
            deposicao = self.ab_carga[b] * self._taxa_dep
            if deposicao > 0.0:
                self.ab_carga[b] -= deposicao
                self.deposicao_total += deposicao
                if p.deposicao_acumulada:
                    self.dose[x, y] += deposicao
                    suficiente = self.dose[x, y] >= p.limiar_polinizacao
                else:
                    suficiente = deposicao >= p.limiar_polinizacao
                if suficiente:
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
            self.ab_viagem[b] = 0
            self.ab_dmax[b] = 0.0
            self.ab_retorno[b] = False
            self.ab_descanso[b] = 0
            self.ab_dentro[b] = False
            self._novo_orcamento(b)
            self.ab_mem_i[b] = self.ab_mem_j[b] = -1
            ocupa = True
            self.ocupacao[cx, cy] = ocupa
            self.n_abelhas += 1

    def _recarregar_flores(self) -> None:
        esgotadas = (self.paisagem == FLOR_DOADORA) & (self.estoque < self.p.estoque_max)
        self.timer_recarga[esgotadas] += 1
        pronto = self.timer_recarga >= self.p.t_recarga
        self.estoque[pronto] = self.p.estoque_max
        self.timer_recarga[pronto] = 0

    def _novo_orcamento(self, b: int) -> None:
        """Sorteia a duração da próxima viagem (só consome aleatoriedade se t_viagem_forma > 0)."""
        p = self.p
        if p.retorno_colmeia and p.t_viagem_forma > 0:
            k = p.t_viagem_forma
            self.ab_orcamento[b] = max(1.0, self.rng.gamma(k, p.t_viagem_max / k))
        else:
            self.ab_orcamento[b] = p.t_viagem_max

    def _mortalidade(self) -> None:
        """Remove abelhas que morrem neste passo (probabilidade ``mortalidade`` cada)."""
        n = self.n_abelhas
        if n == 0:
            return
        morre = self.rng.random(n) < self.p.mortalidade
        if not morre.any():
            return
        for b in np.flatnonzero(morre):
            if not self.ab_dentro[b]:                      # quem está na colmeia não ocupa a célula
                self.ocupacao[self.ab_i[b], self.ab_j[b]] = False
        vivas = ~morre
        k = int(vivas.sum())
        for arr in (self.ab_i, self.ab_j, self.ab_carga, self.ab_sem_flor, self.ab_viagem,
                    self.ab_retorno, self.ab_descanso, self.ab_mem_i, self.ab_mem_j,
                    self.ab_dentro, self.ab_orcamento, self.ab_dmax):
            arr[:k] = arr[:n][vivas]
        self.n_abelhas = k
        self.n_mortes += int(morre.sum())

    def _chegar_na_colmeia(self, b: int) -> None:
        """Fim de uma viagem: entra na colmeia (libera a célula), descarrega, registra e repousa."""
        self.duracoes_viagem.append(int(self.ab_viagem[b]) + 1)
        self.dist_max_viagens.append(float(self.ab_dmax[b]))
        self.ab_dmax[b] = 0.0
        self.ab_dentro[b] = True
        self.ocupacao[self.centro] = False
        self.ab_carga[b] = 0.0
        self.ab_retorno[b] = False
        self.ab_viagem[b] = 0
        self.ab_sem_flor[b] = 0
        self.ab_descanso[b] = self.p.t_descanso

    def _atualizar_viagem(self, b: int, x: int, y: int) -> None:
        """Conta o passo da viagem e decide se é hora de voltar."""
        p = self.p
        self.ab_viagem[b] += 1
        if self._distancias[x, y] > self.ab_dmax[b]:
            self.ab_dmax[b] = self._distancias[x, y]
        if self.ab_retorno[b]:
            return
        if (self.ab_viagem[b] >= self.ab_orcamento[b]
                or self.ab_carga[b] >= p.fracao_carga_retorno * p.cap_max
                or (p.dist_max_voo > 0 and self._distancias[x, y] >= p.dist_max_voo - 1.0)):
            self.ab_retorno[b] = True

    def passo(self) -> None:
        """Avança um passo de tempo."""
        self.passo_atual += 1
        self._gerar_abelha()
        self._recarregar_flores()
        if self.p.mortalidade > 0:
            self._mortalidade()

        visitas = 0
        deposicao_antes = self.deposicao_total
        # Ordem aleatória, atualização sequencial: uma abelha só vai para uma
        # célula livre *no momento* da sua vez, então nenhuma abelha se perde.
        retorno = self.p.retorno_colmeia
        for b in self.rng.permutation(self.n_abelhas):
            if retorno and self.ab_dentro[b]:            # dentro da colmeia
                if self.ab_descanso[b] > 0:
                    self.ab_descanso[b] -= 1
                    continue
                if self.ocupacao[self.centro]:           # saída ocupada: espera
                    continue
                self.ocupacao[self.centro] = True        # sai pela célula da colmeia
                self.ab_dentro[b] = False
                self._novo_orcamento(b)
            i, j = int(self.ab_i[b]), int(self.ab_j[b])
            x, y = self._escolher_destino(i, j, b)
            if (x, y) != (i, j):
                self.ocupacao[i, j] = False
                self.ocupacao[x, y] = True
                self.ab_i[b], self.ab_j[b] = x, y
            if retorno and self.ab_retorno[b] and (x, y) == self.centro:
                self._chegar_na_colmeia(b)
                continue
            visitas += self._interagir(b, x, y)
            if retorno:
                self._atualizar_viagem(b, x, y)

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

    def visitacao_por_distancia(self, largura_bin: float = 2.0):
        """Visitas por flor e por passo em função da distância à colmeia.

        Usa todas as células que contêm (ou continham) flor (estados 2, 3 e 6). É a quantidade
        análoga à "frequência de forrageio" dos modelos de núcleo exponencial.
        Retorna (centros, visitacao, n_flores); NaN em faixas sem flores.
        """
        if largura_bin <= 0:
            raise ValueError("largura_bin deve ser > 0")
        flor = np.isin(self.paisagem, (FLOR_DOADORA, FLOR_RECEPTORA, POLINIZADA))
        faixas = np.floor(self._distancias / largura_bin).astype(int)
        n_faixas = int(faixas.max()) + 1
        n_flores = np.bincount(faixas[flor], minlength=n_faixas)
        visitas = np.bincount(faixas[flor], weights=self.visitas_celula[flor], minlength=n_faixas)
        with np.errstate(invalid="ignore", divide="ignore"):
            vis = np.where(n_flores > 0, visitas / (n_flores * max(self.passo_atual, 1)), np.nan)
        centros = (np.arange(n_faixas) + 0.5) * largura_bin
        return centros, vis, n_flores

    def distancia_media_visitas(self) -> float:
        """Distância média (células) das visitas a flores, ponderada pelo número de visitas."""
        flor = np.isin(self.paisagem, (FLOR_DOADORA, FLOR_RECEPTORA, POLINIZADA))
        total = float(self.visitas_celula[flor].sum())
        if total == 0.0:
            return float("nan")
        return float((self._distancias[flor] * self.visitas_celula[flor]).sum() / total)

    def quantil_visitas(self, q: float = 0.5) -> float:
        """Quantil (células) da distância das visitas a flores, ponderado pelo número de visitas."""
        flor = np.isin(self.paisagem, (FLOR_DOADORA, FLOR_RECEPTORA, POLINIZADA))
        w = self.visitas_celula[flor].astype(float)
        if w.sum() == 0.0:
            return float("nan")
        d = self._distancias[flor]
        ordem = np.argsort(d, kind="stable")
        acum = np.cumsum(w[ordem]) / w.sum()
        return float(d[ordem][np.searchsorted(acum, q)])

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
