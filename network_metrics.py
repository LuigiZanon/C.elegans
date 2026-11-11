"""
Módulo de Métricas Topológicas e Redes Sintéticas.
Implementa o cálculo do Coeficiente de Agrupamento (Clustering) e o modelo 
gerador de redes aleatórias de Erdős-Rényi, conforme os conceitos teóricos.
"""

import networkx as nx
from typing import Any, Optional


class NetworkMetricsAnalyzer:
    """
    Classe utilitária para calcular métricas locais e globais de grafos
    e gerar redes sintéticas para comparações.
    """

    @staticmethod
    def calcular_coeficiente_agrupamento_local(grafo: nx.Graph, no: Any) -> float:
        """
        Calcula o coeficiente de agrupamento local (C_i) para um nó específico.
        
        O coeficiente local mede a tendência de formação de triângulos ao redor
        de um vértice específico. Mede a proporção das ligações existentes entre 
        os vizinhos do nó em comparação com as ligações possíveis.
        
        Fórmula teórica: C_i = (2 * e_i) / (k_i * (k_i - 1))
        Onde:
        - e_i: número de arestas existentes entre os vizinhos do nó 'i'.
        - k_i: grau do nó 'i' (número de vizinhos).
        
        Args:
            grafo (nx.Graph): O grafo não-direcionado (simples) a ser analisado.
            no (Any): O identificador do nó.
            
        Returns:
            float: O valor do coeficiente de agrupamento local (entre 0 e 1).
        """
        # Obtém a lista de vizinhos do nó passado como parâmetro
        vizinhos = list(grafo.neighbors(no))
        k = len(vizinhos)
        
        # Se o nó tiver menos de 2 vizinhos, é impossível formar um triângulo.
        # Portanto, o agrupamento local é 0.
        if k < 2:
            return 0.0
            
        # Conta as arestas existentes apenas entre os vizinhos do nó 'no' (e_i)
        e_i = 0
        for i in range(k):
            for j in range(i + 1, k):
                # Verifica se o vizinho 'i' está conectado ao vizinho 'j'
                if grafo.has_edge(vizinhos[i], vizinhos[j]):
                    e_i += 1
                    
        # Aplica a fórmula da densidade de vizinhança
        c_i = (2.0 * e_i) / (k * (k - 1))
        
        return c_i

    @classmethod
    def calcular_coeficiente_agrupamento_global(cls, grafo: nx.Graph) -> float:
        """
        Calcula o coeficiente de agrupamento global (C) da rede inteira.
        
        O coeficiente global é obtido através da média aritmética simples dos 
        coeficientes de agrupamento locais de todos os vértices da rede.
        
        Fórmula teórica: C = (1 / n) * Σ C_i
        
        Args:
            grafo (nx.Graph): O grafo não-direcionado a ser analisado.
            
        Returns:
            float: O coeficiente de agrupamento global da rede.
        """
        n = grafo.number_of_nodes()
        
        # Se a rede for vazia, retorna 0 para evitar divisão por zero.
        if n == 0:
            return 0.0
            
        soma_clustering = 0.0
        
        # Itera sobre todos os nós para somar os coeficientes locais
        for no in grafo.nodes():
            soma_clustering += cls.calcular_coeficiente_agrupamento_local(grafo, no)
            
        # Calcula a média (divisão da soma pelo número total de nós)
        clustering_global = soma_clustering / n
        
        return clustering_global


class SyntheticNetworkGenerator:
    """
    Classe responsável por gerar modelos nulos/sintéticos de redes, permitindo
    comparações (benchmarks) com redes empíricas/reais.
    """

    @staticmethod
    def gerar_rede_aleatoria(n: int, p: float, seed: Optional[int] = None) -> nx.Graph:
        """
        Gera uma rede aleatória sintética utilizando o modelo Erdős-Rényi G(n, p).
        
        Neste modelo, o grafo começa apenas com os nós isolados. 
        Em seguida, cada par possível de vértices é conectado de forma independente
        com uma probabilidade constante 'p'. 
        
        As redes aleatórias costumam apresentar coeficiente de clustering global baixo
        (aproximadamente igual à probabilidade 'p') e não possuem estruturas como hubs,
        servindo de 'baseline' sem a topologia específica do sistema biológico.
        
        Args:
            n (int): Número total de vértices (nós) desejados na rede.
            p (float): Probabilidade de conexão entre qualquer par de vértices (0 <= p <= 1).
            seed (int, optional): Semente de aleatoriedade para reprodutibilidade.
            
        Returns:
            nx.Graph: O grafo gerado aleatoriamente (não-direcionado).
        """
        return nx.gnp_random_graph(n, p, seed=seed)