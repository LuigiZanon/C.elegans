"""
Módulo de métricas de percolação e fragmentação em matriz esparsa.
"""

from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components, shortest_path
from graph_utils import projetar_grafo


class NetworkMetricsAnalyzer:
    """Calcula métricas de fragmentação e percolação via representação esparsa."""

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
        """Simula percolação sob 4 regimes: Falhas Aleatórias, Grau/Força, Intermediação e Rich-Club."""
        g_base = (
            projetar_grafo(grafo, camada="Combinada", direcionado=False, preservar_atributos_nos=False)
            if isinstance(grafo, nx.MultiDiGraph)
            else grafo.to_undirected()
        )
        nos = list(g_base.nodes())
        n_total = len(nos)
        if n_total == 0:
            return pd.DataFrame(), {}

        idx_no = {n: i for i, n in enumerate(nos)}
        adj_full = nx.to_scipy_sparse_array(g_base, nodelist=nos, format="csr", dtype=np.float64)
        gcc_0, e_glob_0 = cls._metricas_fragmentacao_csr(adj_full.copy(), n_total)
        gcc_0, e_glob_0 = max(gcc_0, 1), max(e_glob_0, 1e-12)
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
