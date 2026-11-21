"""
Módulo de benchmark comparativo entre o conectoma real do C. elegans e modelos
de redes sintéticas equivalentes (Erdős-Rényi, Watts-Strogatz e Barabási-Albert).
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from graph_utils import (
    calibrar_parametros_sinteticos,
    extrair_componente_principal,
    gerar_rede_sintetica,
    projetar_grafo,
)


class SyntheticNetworkBenchmark:
    """Projeta o conectoma em grafo simples, calibra ER/WS/BA/Lattice e calcula índices Small-World."""

    MODELO_REAL = "C. elegans (Real)"
    MODELO_ER = "Erdős-Rényi (ER)"
    MODELO_WS = "Watts-Strogatz (WS)"
    MODELO_BA = "Barabási-Albert (BA)"

    CHAVES_METRICAS = (
        "N", "M", "Densidade", "Grau_Medio", "Grau_Desvio", "Grau_Max",
        "Heterogeneidade", "Agrupamento_C", "Transitividade_T",
        "Caminho_Medio_L", "Diametro_D", "Eficiencia_Global",
        "Assortatividade_r", "Fracao_GCC",
    )

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        n_runs: int = 30,
        seed_base: int = 42,
        ws_rewiring_p: float = 0.10,
    ) -> None:
        self.graph = graph
        self.n_runs = int(n_runs)
        self.seed_base = int(seed_base)
        self.ws_rewiring_p = float(ws_rewiring_p)

        self.g_real: nx.Graph = projetar_grafo(
            graph, camada="Combinada", direcionado=False, preservar_atributos_nos=False
        )
        params = calibrar_parametros_sinteticos(self.g_real)
        self.n: int = int(params["N"])
        self.m: int = int(params["M"])
        self.k_medio: float = float(params["k_medio"])
        self.densidade: float = float(params["densidade"])
        self.k_ws: int = int(params["k_ws"])
        self.m_ba: int = int(params["m_ba"])
        self.baselines_er: Optional[Dict[str, Any]] = None

    @staticmethod
    def _calcular_metricas_grafo(g: nx.Graph) -> Dict[str, float]:
        """Extrai métricas topológicas com L e D calculados na Maior Componente Conexa (GCC)."""
        n_nodes = g.number_of_nodes()
        if n_nodes == 0:
            return {chave: 0.0 for chave in SyntheticNetworkBenchmark.CHAVES_METRICAS}

        gcc = extrair_componente_principal(g)
        graus = np.array([deg for _, deg in g.degree()], dtype=float)
        k_med = float(np.mean(graus))
        het = float(np.mean(graus ** 2) / (k_med ** 2)) if k_med > 0.0 else 0.0

        if gcc.number_of_nodes() > 1:
            l_val = float(nx.average_shortest_path_length(gcc))
            d_val = float(nx.diameter(gcc))
        else:
            l_val, d_val = 0.0, 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                assort = float(nx.degree_assortativity_coefficient(g))
            except Exception:
                assort = 0.0
        if np.isnan(assort) or np.isinf(assort):
            assort = 0.0

        return {
            "N": float(n_nodes),
            "M": float(g.number_of_edges()),
            "Densidade": float(nx.density(g)),
            "Grau_Medio": k_med,
            "Grau_Desvio": float(np.std(graus, ddof=1)) if n_nodes > 1 else 0.0,
            "Grau_Max": float(np.max(graus)),
            "Heterogeneidade": het,
            "Agrupamento_C": float(nx.average_clustering(g)),
            "Transitividade_T": float(nx.transitivity(g)),
            "Caminho_Medio_L": l_val,
            "Diametro_D": d_val,
            "Eficiencia_Global": float(nx.global_efficiency(g)),
            "Assortatividade_r": assort,
            "Fracao_GCC": float(gcc.number_of_nodes() / n_nodes),
        }

    def _agregar_execucoes(
        self, nome_modelo: str, lista_metricas: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Agrega média e desvio padrão amostral das métricas de um conjunto de execuções."""
        linha: Dict[str, Any] = {"Modelo": nome_modelo}
        for chave in self.CHAVES_METRICAS:
            vals = np.array([m[chave] for m in lista_metricas], dtype=float)
            linha[f"{chave}_Media"] = float(np.mean(vals))
            linha[f"{chave}_Std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return linha

    def executar_benchmark(self) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Executa o benchmark Monte Carlo comparando o conectoma real com ER, WS e BA."""
        linhas: List[Dict[str, Any]] = []
        graus_por_modelo: Dict[str, np.ndarray] = {}

        linhas.append(self._agregar_execucoes(self.MODELO_REAL, [self._calcular_metricas_grafo(self.g_real)]))
        graus_por_modelo[self.MODELO_REAL] = np.array([deg for _, deg in self.g_real.degree()], dtype=int)

        modelos_spec = [
            (self.MODELO_ER, "ER"),
            (self.MODELO_WS, "WS"),
            (self.MODELO_BA, "BA"),
        ]
        er_rodadas: List[Dict[str, float]] = []

        for nome_modelo, codigo in modelos_spec:
            rodadas: List[Dict[str, float]] = []
            graus_acum: List[np.ndarray] = []
            for i in range(self.n_runs):
                g_sint = gerar_rede_sintetica(
                    codigo, self.n, self.m, self.k_ws, self.m_ba, self.ws_rewiring_p, self.seed_base + i
                )
                rodadas.append(self._calcular_metricas_grafo(g_sint))
                graus_acum.append(np.array([deg for _, deg in g_sint.degree()], dtype=int))
            if codigo == "ER":
                er_rodadas = rodadas
            linhas.append(self._agregar_execucoes(nome_modelo, rodadas))
            graus_por_modelo[nome_modelo] = np.concatenate(graus_acum)

        g_latt = gerar_rede_sintetica("LATTICE", self.n, self.m, self.k_ws, self.m_ba, 0.0, self.seed_base)
        gcc_latt = extrair_componente_principal(g_latt)
        c_latt = float(nx.average_clustering(g_latt))
        l_latt = float(nx.average_shortest_path_length(gcc_latt)) if gcc_latt.number_of_nodes() > 1 else 0.0

        self.baselines_er = {
            "n_runs": self.n_runs,
            "seed_base": self.seed_base,
            "c_er_arr": np.array([r["Agrupamento_C"] for r in er_rodadas], dtype=float),
            "l_er_arr": np.array([r["Caminho_Medio_L"] for r in er_rodadas], dtype=float),
            "kappa_er_arr": np.array([r["Heterogeneidade"] for r in er_rodadas], dtype=float),
            "graus_er_arr": graus_por_modelo[self.MODELO_ER].astype(float),
            "c_latt": c_latt,
            "l_latt": l_latt,
        }

        colunas = ["Modelo"] + [f"{c}_{s}" for c in self.CHAVES_METRICAS for s in ("Media", "Std")]
        return pd.DataFrame(linhas)[colunas], graus_por_modelo

    def calcular_diagnostico_topologico(
        self, df_comparacao: pd.DataFrame, graus_por_modelo: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Calcula sigma, omega e estatísticas de cauda (MLE + R² log-log vs semi-log)."""
        por_m = df_comparacao.set_index("Modelo")
        c_real = float(por_m.loc[self.MODELO_REAL, "Agrupamento_C_Media"])
        c_er = float(por_m.loc[self.MODELO_ER, "Agrupamento_C_Media"])
        c_ws = float(por_m.loc[self.MODELO_WS, "Agrupamento_C_Media"])
        c_ba = float(por_m.loc[self.MODELO_BA, "Agrupamento_C_Media"])
        l_real = float(por_m.loc[self.MODELO_REAL, "Caminho_Medio_L_Media"])
        l_er = float(por_m.loc[self.MODELO_ER, "Caminho_Medio_L_Media"])
        l_ws = float(por_m.loc[self.MODELO_WS, "Caminho_Medio_L_Media"])
        l_ba = float(por_m.loc[self.MODELO_BA, "Caminho_Medio_L_Media"])
        het_real = float(por_m.loc[self.MODELO_REAL, "Heterogeneidade_Media"])
        het_er = float(por_m.loc[self.MODELO_ER, "Heterogeneidade_Media"])
        het_ba = float(por_m.loc[self.MODELO_BA, "Heterogeneidade_Media"])

        if self.baselines_er is not None:
            c_lattice = float(self.baselines_er["c_latt"])
        else:
            g_latt = gerar_rede_sintetica("LATTICE", self.n, self.m, self.k_ws, self.m_ba, 0.0, self.seed_base)
            c_lattice = float(nx.average_clustering(g_latt))

        razao_c_er = float(c_real / c_er) if c_er > 0.0 else 0.0
        razao_l_er = float(l_real / l_er) if l_er > 0.0 else 0.0
        sigma_sw = float(razao_c_er / razao_l_er) if razao_l_er > 0.0 else 0.0
        omega_sw = float((l_er / l_real if l_real > 0 else 0.0) - (c_real / c_lattice if c_lattice > 0 else 0.0))

        graus_real = np.asarray(graus_por_modelo[self.MODELO_REAL], dtype=float)
        k_min = int(np.ceil(self.k_medio))
        cauda = graus_real[graus_real >= k_min]
        if len(cauda) < 5:
            k_min = max(1, int(np.percentile(graus_real, 50)))
            cauda = graus_real[graus_real >= k_min]

        k_unicos = np.unique(cauda)
        ccdf = np.array([np.mean(graus_real >= k) for k in k_unicos], dtype=float)
        log_k, log_p = np.log(k_unicos), np.log(ccdf)
        ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))
        if len(k_unicos) >= 2 and ss_tot > 0.0:
            c_pl = np.polyfit(log_k, log_p, 1)
            r2_ll = float(1.0 - np.sum((log_p - (c_pl[0] * log_k + c_pl[1])) ** 2) / ss_tot)
            c_ex = np.polyfit(k_unicos, log_p, 1)
            r2_sl = float(1.0 - np.sum((log_p - (c_ex[0] * k_unicos + c_ex[1])) ** 2) / ss_tot)
        else:
            r2_ll, r2_sl = 0.0, 0.0

        n_tail = int(len(cauda))
        x_min_c = max(0.5, float(k_min) - 0.5)
        soma_log = float(np.sum(np.log(cauda / x_min_c)))
        gamma_mle = float(1.0 + n_tail / soma_log) if soma_log > 0.0 else 0.0
        desloc = float(np.mean(cauda) - x_min_c)
        lam_mle = float(1.0 / desloc) if desloc > 0.0 else 0.0
        if gamma_mle > 1.0 and lam_mle > 0.0:
            ll_pl = float(n_tail * np.log((gamma_mle - 1.0) / x_min_c) - gamma_mle * soma_log)
            ll_exp = float(n_tail * np.log(lam_mle) - lam_mle * float(np.sum(cauda - x_min_c)))
            delta_ll = float(ll_pl - ll_exp)
        else:
            delta_ll = 0.0

        eh_sw = sigma_sw > 1.0 and razao_c_er > 2.0 and abs(omega_sw) <= 0.5
        if eh_sw and delta_ll > 0.0 and r2_ll > r2_sl:
            classe = "Small-World + Cauda Scale-Free"
        elif eh_sw:
            classe = "Small-World + Cauda Pesada Truncada"
        elif sigma_sw > 1.0 and razao_c_er > 1.5:
            classe = "Small-World Moderado"
        else:
            classe = "Próximo ao Aleatório"

        return {
            "N": self.n, "M": self.m, "k_medio": self.k_medio,
            "k_ws": self.k_ws, "m_ba": self.m_ba, "p_ws": self.ws_rewiring_p,
            "C_real": c_real, "C_ER": c_er, "C_WS": c_ws, "C_BA": c_ba, "C_lattice": c_lattice,
            "L_real": l_real, "L_ER": l_er, "L_WS": l_ws, "L_BA": l_ba,
            "razao_C_ER": razao_c_er, "razao_L_ER": razao_l_er,
            "sigma_small_world": sigma_sw, "omega_small_world": omega_sw,
            "k_min_cauda": k_min, "gamma_mle": gamma_mle, "lambda_exp_mle": lam_mle,
            "r2_log_log": r2_ll, "r2_semi_log": r2_sl, "delta_log_likelihood": delta_ll,
            "heterogeneidade_real": het_real, "heterogeneidade_ER": het_er, "heterogeneidade_BA": het_ba,
            "veredito_classificacao": classe,
        }
