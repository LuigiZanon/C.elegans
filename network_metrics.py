"""
Módulo de Métricas Topológicas e Redes Sintéticas.
Implementa o cálculo do Coeficiente de Agrupamento (Clustering) e o modelo 
gerador de redes aleatórias de Erdős-Rényi, conforme os conceitos teóricos.
"""

import networkx as nx
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, shortest_path

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

    @staticmethod
    def _converter_para_grafo_simples(grafo: nx.Graph) -> nx.Graph:
        """Converte MultiDiGraph ou DiGraph em grafo simples não-direcionado ponderado."""
        if not grafo.is_directed() and not grafo.is_multigraph():
            return grafo
        g_simples = nx.Graph()
        g_simples.add_nodes_from(grafo.nodes())
        for u, v, dados in grafo.edges(data=True):
            if u != v:
                w = float(dados.get("weight", 1.0))
                if g_simples.has_edge(u, v):
                    g_simples[u][v]["weight"] += w
                else:
                    g_simples.add_edge(u, v, weight=w)
        return g_simples

    @staticmethod
    def _metricas_fragmentacao_csr(adj_sub, n_total: int) -> Tuple[int, float]:
        """Calcula |GCC| e Eficiência Global normalizada pelo N original via matriz esparsa."""
        if adj_sub.shape[0] == 0:
            return 0, 0.0
        adj_sub.indices = adj_sub.indices.astype(np.int32, copy=False)
        adj_sub.indptr = adj_sub.indptr.astype(np.int32, copy=False)
        _, labels = connected_components(adj_sub, directed=False)
        gcc_size = int(np.bincount(labels).max())
        if n_total <= 1 or adj_sub.nnz == 0:
            return gcc_size, 0.0
        dist = shortest_path(adj_sub, directed=False, unweighted=True)
        mask = (dist > 0) & np.isfinite(dist)
        soma_inv = float(np.sum(1.0 / dist[mask]))
        return gcc_size, soma_inv / (n_total * (n_total - 1))

    @classmethod
    def simular_robustez_percolacao(
        cls,
        grafo: nx.Graph,
        membros_rich_club: Optional[List[str]] = None,
        passo: float = 0.02,
        n_sim_aleatorio: int = 30,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Simula percolação (remoção progressiva de fração f in [0, 1]) sob 4 regimes:
        Falhas Aleatórias, Ataque por Grau/Força, Ataque por Intermediação e Ataque ao Rich-Club.
        """
        g_base = cls._converter_para_grafo_simples(grafo)
        nos = list(g_base.nodes())
        n_total = len(nos)
        if n_total == 0:
            return pd.DataFrame(), {}

        idx_no = {n: i for i, n in enumerate(nos)}
        adj_full = nx.to_scipy_sparse_array(g_base, nodelist=nos, format="csr", dtype=np.float64)

        gcc_0, e_glob_0 = cls._metricas_fragmentacao_csr(adj_full.copy(), n_total)
        gcc_0 = max(gcc_0, 1)
        e_glob_0 = max(e_glob_0, 1e-12)
        fracoes = np.round(np.arange(0.0, 1.0 + 1e-9, passo), 4)

        forca = dict(g_base.degree(weight="weight"))
        ordem_grau = [idx_no[n] for n in sorted(nos, key=lambda n: (forca[n], g_base.degree(n)), reverse=True)]

        g_dist = g_base.copy()
        for _, _, d in g_dist.edges(data=True):
            d["distance"] = 1.0 / max(float(d.get("weight", 1.0)), 1e-6)
        betw = nx.betweenness_centrality(g_dist, weight="distance", normalized=True)
        ordem_interm = [idx_no[n] for n in sorted(nos, key=lambda n: betw[n], reverse=True)]

        rc_validos = [idx_no[n] for n in (membros_rich_club or []) if n in idx_no]
        rc_set = set(rc_validos)
        ordem_rc = rc_validos + [i for i in ordem_grau if i not in rc_set]

        registros = []
        rng = np.random.default_rng(seed)

        s_rand = np.zeros((n_sim_aleatorio, len(fracoes)))
        e_rand = np.zeros((n_sim_aleatorio, len(fracoes)))
        for sim in range(n_sim_aleatorio):
            perm = rng.permutation(n_total)
            for idx, f in enumerate(fracoes):
                k_rem = int(round(f * n_total))
                if k_rem < n_total:
                    keep = perm[k_rem:]
                    gcc_f, e_f = cls._metricas_fragmentacao_csr(adj_full[keep][:, keep], n_total)
                    s_rand[sim, idx] = gcc_f / gcc_0
                    e_rand[sim, idx] = e_f / e_glob_0

        for idx, f in enumerate(fracoes):
            registros.append({
                "Regime": "Falhas Aleatórias",
                "Fracao_Removida": float(f),
                "S_GCC": float(np.mean(s_rand[:, idx])),
                "S_GCC_Std": float(np.std(s_rand[:, idx])),
                "E_Glob_Norm": float(np.mean(e_rand[:, idx])),
                "E_Glob_Std": float(np.std(e_rand[:, idx])),
            })

        ataques = {
            "Ataque por Grau/Força": ordem_grau,
            "Ataque por Intermediação": ordem_interm,
            "Ataque ao Rich-Club": ordem_rc,
        }
        for regime, ordem in ataques.items():
            ordem_arr = np.asarray(ordem, dtype=int)
            for f in fracoes:
                k_rem = int(round(f * n_total))
                if k_rem >= n_total:
                    s_val, e_val = 0.0, 0.0
                else:
                    keep = ordem_arr[k_rem:]
                    gcc_f, e_f = cls._metricas_fragmentacao_csr(adj_full[keep][:, keep], n_total)
                    s_val, e_val = gcc_f / gcc_0, e_f / e_glob_0
                registros.append({
                    "Regime": regime,
                    "Fracao_Removida": float(f),
                    "S_GCC": float(s_val),
                    "S_GCC_Std": 0.0,
                    "E_Glob_Norm": float(e_val),
                    "E_Glob_Std": 0.0,
                })

        df_rob = pd.DataFrame(registros)
        limiares_fc: Dict[str, float] = {}
        for regime, grp in df_rob.groupby("Regime"):
            abaixo = grp[grp["S_GCC"] < 0.5]
            limiares_fc[str(regime)] = float(abaixo["Fracao_Removida"].iloc[0]) if not abaixo.empty else 1.0

        return df_rob, limiares_fc


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