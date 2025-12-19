import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

# --- Definição de Constantes e Estados ---
VAZIO = 0
OBSTACULO = 1
FLOR_DOADORA = 2
FLOR_RECEPTORA = 3
COLMEIA = 4
# O Estado 5 (Abelha) é gerenciado como uma camada de agentes sobre a grade
FLOR_POLINIZADA = 6

# Parâmetros da Simulação
GRID_SIZE = 50           # Tamanho da grade (N x N)
NUM_ABELHAS = 20         # Número máximo de abelhas
K_ATRACAO = 10           # Fator de atração floral (K)
CAPACIDADE_MAX = 100     # Carga máxima de pólen (C_max)
TEMPO_RETENCAO_MAX = 20  # Tempo máximo sem visitar flor (T_max)
EFICIENCIA_COLETA = 10   # Quanto pólen a abelha pega por tick
TEMPO_RECARGA_FLOR = 50  # Ticks para a flor regenerar

class Abelha:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.carga_polen = 0
        self.tempo_sem_flor = 0  

class ModeloPolinizacaoCA:
    def __init__(self, size=GRID_SIZE):
        self.size = size
       
        self.grid = np.zeros((size, size), dtype=int)
       
        self.estoque_polen = np.zeros((size, size), dtype=int)
        
        self.timer_recarga = np.zeros((size, size), dtype=int)
        
        self.abelhas = []
        self.passo_atual = 0
        
        self._inicializar_paisagem()

    def _inicializar_paisagem(self):
       
        centro = self.size // 2
        self.grid[centro, centro] = COLMEIA
        
        
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == (centro, centro): continue
                
                r = random.random()
                if r < 0.05:
                    self.grid[i, j] = OBSTACULO
                elif r < 0.15:
                    self.grid[i, j] = FLOR_DOADORA
                    self.estoque_polen[i, j] = 100 
                elif r < 0.20:
                    self.grid[i, j] = FLOR_RECEPTORA

    def _obter_vizinhos(self, x, y):
       
        vizinhos = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                
                nx, ny = (x + dx) % self.size, (y + dy) % self.size
                vizinhos.append((nx, ny))
        return vizinhos

    def _calcular_movimento(self, abelha):
        
        vizinhos = self._obter_vizinhos(abelha.x, abelha.y)
        pesos = []
        coords_validas = []

        for vx, vy in vizinhos:
            estado = self.grid[vx, vy]
            if estado == OBSTACULO:
                continue # Não pode mover para obstáculo
            
            # Função de peso W = 1 + K * I_flor
            peso = 1.0
            if estado in [FLOR_DOADORA, FLOR_RECEPTORA, FLOR_POLINIZADA]:
                peso += K_ATRACAO
            
            pesos.append(peso)
            coords_validas.append((vx, vy))

        if not coords_validas:
            return # Sem movimento possível

        # Normalizar para obter probabilidade P
        total_peso = sum(pesos)
        probs = [p / total_peso for p in pesos]
        
        # Escolha aleatória baseada nos pesos
        escolha_idx = np.random.choice(len(coords_validas), p=probs)
        novo_x, novo_y = coords_validas[escolha_idx]
        
        
        abelha.x, abelha.y = novo_x, novo_y

    def _processar_interacao(self, abelha):
        
        x, y = abelha.x, abelha.y
        estado_celula = self.grid[x, y]
        
        interagiu = False

        
        if estado_celula == FLOR_DOADORA:
            disponivel = self.estoque_polen[x, y]
            if disponivel > 0:
                coleta = min(EFICIENCIA_COLETA, disponivel, CAPACIDADE_MAX - abelha.carga_polen)
                abelha.carga_polen += coleta
                self.estoque_polen[x, y] -= coleta
                interagiu = True

        # Transição de Deposição (Receptora)
        elif estado_celula == FLOR_RECEPTORA:
            if abelha.carga_polen > 0:
                prob_polinizacao = min(1.0, abelha.carga_polen / CAPACIDADE_MAX)
                if random.random() < prob_polinizacao:
                    self.grid[x, y] = FLOR_POLINIZADA 
                    abelha.carga_polen -= 1 
                    interagiu = True

       
        if interagiu:
            abelha.tempo_sem_flor = 0
        else:
            abelha.tempo_sem_flor += 1
            if abelha.tempo_sem_flor >= TEMPO_RETENCAO_MAX:
                abelha.carga_polen = 0 

    def _atualizar_recarga_floral(self):
        
        mask_esgotadas = (self.grid == FLOR_DOADORA) & (self.estoque_polen < 100)
        
        self.timer_recarga[mask_esgotadas] += 1
        
        
        mask_reset = (self.timer_recarga >= TEMPO_RECARGA_FLOR)
        self.estoque_polen[mask_reset] = 100
        self.timer_recarga[mask_reset] = 0

    def step(self):
        
        self.passo_atual += 1

        # 1. Geração de Abelhas (Colmeia)
        if len(self.abelhas) < NUM_ABELHAS:
            centro = self.size // 2
            self.abelhas.append(Abelha(centro, centro))

       
        self._atualizar_recarga_floral()

        # 3. Movimento e Interação
        for abelha in self.abelhas:
            self._calcular_movimento(abelha)
            self._processar_interacao(abelha)

    def visualizar(self):
       
        cmap = mcolors.ListedColormap(['white', 'black', 'orange', 'purple', 'blue', 'yellow', 'red'])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        
        grid_viz = self.grid.copy()
        for abelha in self.abelhas:
            grid_viz[abelha.x, abelha.y] = 5  

        plt.figure(figsize=(8, 8))
        plt.imshow(grid_viz, cmap=cmap, norm=norm)
        
        # Legenda
        legend_labels = {
            0: 'Vazio', 1: 'Obstáculo', 2: 'Flor Doadora', 
            3: 'Flor Receptora', 4: 'Colmeia', 5: 'Abelha', 6: 'Polinizada'
        }
        patches = [plt.Rectangle((0,0),1,1, color=cmap(i)) for i in range(7)]
        plt.legend(patches, legend_labels.values(), loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.title(f"Simulação de Polinização - Passo {self.passo_atual}")
        plt.grid(False)
        plt.show()


if __name__ == "__main__":
    modelo = ModeloPolinizacaoCA(size=30)
    
   
    for _ in range(100):
        modelo.step()
    
    #
    modelo.visualizar()