import numpy as np
import matplotlib.pyplot as plt
import random

# --- Constantes (Mesmas da versão anterior) ---
VAZIO = 0; NEUTRO = 1; FLOR_DOADORA = 2; FLOR_RECEPTORA = 3; COLMEIA = 4; POLINIZADA = 6
SEM_ABELHA = 0; COM_ABELHA = 1

class PolinizacaoCA:
    def __init__(self, linhas, colunas, params):
        self.linhas = linhas; self.colunas = colunas; self.params = params
        self.landscape = np.zeros((linhas, colunas), dtype=int)
        self.agents = np.zeros((linhas, colunas), dtype=int)
        self.agent_pollen = np.zeros((linhas, colunas), dtype=float)
        self.historico_temp = []
        self.inicializar()

    def inicializar(self):
        self.landscape.fill(NEUTRO)
        cx, cy = self.linhas//2, self.colunas//2
        self.landscape[cx, cy] = COLMEIA
        for i in range(self.linhas):
            for j in range(self.colunas):
                if (i,j) == (cx,cy): continue
                if random.random() < self.params['densidade_floral']:
                    self.landscape[i,j] = FLOR_DOADORA if random.random() < 0.5 else FLOR_RECEPTORA
                elif random.random() > 0.95: self.landscape[i,j] = VAZIO

    def obter_vizinhos(self, i, j):
        viz = []
        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                if di==0 and dj==0: continue
                viz.append(((i+di)%self.linhas, (j+dj)%self.colunas))
        return viz

    def rodar_passo(self):
        locais = np.argwhere(self.agents == COM_ABELHA)
        np.random.shuffle(locais)
        novos_agentes = np.zeros_like(self.agents)
        novo_polen = np.zeros_like(self.agent_pollen)
        
        
        for i, j in locais:
            vizinhos = self.obter_vizinhos(i, j)
            candidatos, pesos = [], []
            for ni, nj in vizinhos:
                terr = self.landscape[ni,nj]
                if novos_agentes[ni,nj] or terr == VAZIO: continue
                w = self.params['alpha_atracao'] if terr in [FLOR_DOADORA, FLOR_RECEPTORA] else 1.0
                candidatos.append((ni,nj)); pesos.append(w)
            
            ni, nj = (i, j)
            if candidatos:
                probs = [p/sum(pesos) for p in pesos]
                ni, nj = candidatos[np.random.choice(len(candidatos), p=probs)]
            
            carga = self.agent_pollen[i, j]
            tipo = self.landscape[ni, nj]
            
            if tipo == FLOR_DOADORA:
                carga += min(self.params['eficiencia_coleta'], self.params['cap_max'] - carga)
            elif tipo == FLOR_RECEPTORA:
                dep = carga * self.params['taxa_deposicao']
                if dep > 0:
                    carga -= dep
                    if dep >= self.params['limiar_polinizacao']: self.landscape[ni, nj] = POLINIZADA
            
            novos_agentes[ni, nj] = COM_ABELHA
            novo_polen[ni, nj] = carga

        
        cx, cy = self.linhas//2, self.colunas//2
        if novos_agentes[cx,cy] == 0 and np.sum(novos_agentes) < self.params['max_abelhas']:
            if random.random() < 0.3: novos_agentes[cx,cy] = 1

        self.agents = novos_agentes
        self.agent_pollen = novo_polen
        self.historico_temp.append(np.sum(self.landscape == POLINIZADA))

    def obter_dados_finais(self):
        
        cx, cy = self.linhas//2, self.colunas//2
        y, x = np.indices(self.landscape.shape)
        dists = np.sqrt((x-cy)**2 + (y-cx)**2)
        
        
        dados_flores = []
        for i in range(self.linhas):
            for j in range(self.colunas):
                if self.landscape[i,j] in [FLOR_DOADORA, FLOR_RECEPTORA, POLINIZADA]:
                    sucesso = 1 if self.landscape[i,j] == POLINIZADA else 0
                    dados_flores.append((dists[i,j], sucesso))
        return dados_flores, self.historico_temp

# --- O GERENTE DE MONTE CARLO ---
def executar_monte_carlo(n_simulacoes=50, passos=150):
    params = {
        'densidade_floral': 0.15, 'alpha_atracao': 12.0,
        'cap_max': 40.0, 'eficiencia_coleta': 8.0,
        'taxa_deposicao': 0.6, 'limiar_polinizacao': 4.0, 'max_abelhas': 60
    }
    
    print(f"Iniciando Monte Carlo ({n_simulacoes} simulações)...")
    
    # Armazenadores de resultados
    todos_historicos = [] # Para gráfico temporal
    dados_espaciais_agrupados = {} # Para gráfico de distância
    
    for n in range(n_simulacoes):
        if n % 10 == 0: print(f"  Simulação {n}/{n_simulacoes}...")
        modelo = PolinizacaoCA(60, 60, params)
        for _ in range(passos):
            modelo.rodar_passo()
        
        flores, hist = modelo.obter_dados_finais()
        todos_historicos.append(hist)
        
        # Agrupar dados espaciais por bin de distância
        for dist, sucesso in flores:
            d_bin = int(dist // 2) * 2 
            if d_bin not in dados_espaciais_agrupados:
                dados_espaciais_agrupados[d_bin] = []
            dados_espaciais_agrupados[d_bin].append(sucesso)

    print("Processando estatísticas...")
    
    # 1. Estatísticas Temporais
    arr_hist = np.array(todos_historicos) 
    media_temp = np.mean(arr_hist, axis=0)
    std_temp = np.std(arr_hist, axis=0)
    
    # 2. Estatísticas Espaciais
    distancias = sorted(dados_espaciais_agrupados.keys())
    medias_esp = []
    stds_esp = []
    
    for d in distancias:
        vals = dados_espaciais_agrupados[d]
        if len(vals) > 0:
            
            m = np.mean(vals) * 100
            s = np.std(vals) * 100
            medias_esp.append(m)
            stds_esp.append(s)
        else:
            medias_esp.append(0)
            stds_esp.append(0)

    # --- PLOTAGEM ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico Temporal com "Sombra" de Desvio Padrão
    x_tempo = range(len(media_temp))
    ax1.plot(x_tempo, media_temp, color='darkgreen', linewidth=2, label='Média')
    ax1.fill_between(x_tempo, media_temp - std_temp, media_temp + std_temp, 
                     color='lightgreen', alpha=0.4, label='Desvio Padrão (σ)')
    ax1.set_title(f"Evolução Temporal Média (N={n_simulacoes})")
    ax1.set_xlabel("Tempo (Steps)")
    ax1.set_ylabel("Flores Polinizadas")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Gráfico Espacial com Barras de Erro
    ax2.bar(distancias, medias_esp, width=1.6, color='orange', edgecolor='black', 
            yerr=stds_esp, capsize=4, alpha=0.7, label='Eficiência Média')
    
    # Linha de tendência
    ax2.plot(distancias, medias_esp, 'r--', marker='o')
    
    ax2.set_title("Decaimento de Polinização (Média ± σ)")
    ax2.set_xlabel("Distância da Colmeia")
    ax2.set_ylabel("% de Sucesso")
    ax2.set_ylim(0, 110)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    executar_monte_carlo(n_simulacoes=50, passos=120)