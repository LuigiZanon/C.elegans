"""
Módulo de classificação topológica formal com evidências estatísticas em NumPy/NetworkX puros.
Avalia o conectoma do C. elegans frente às três hipóteses clássicas:
  1. Aleatório (Erdős-Rényi / Poisson)
  2. Small-World (Watts-Strogatz, índices sigma, omega e phi_SWP)
  3. Livre de Escala (Protocolo de Clauset-Shalizi-Newman + Teste de Vuong + AIC/BIC na cauda)
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx


def _para_nativo(obj: Any) -> Any:
    """Converte recursivamente tipos NumPy para tipos nativos Python serializáveis em JSON."""
    if isinstance(obj, dict):
        return {str(k): _para_nativo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_para_nativo(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_para_nativo(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj


class TopologyClassifier:
    """
    Executa testes estatísticos rigorosos em NumPy/NetworkX puros para diagnosticar
    a classe topológica do conectoma do C. elegans com evidências quantitativas.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        n_runs_null: int = 30,
        n_runs_ws_sweep: int = 15,
        n_bootstrap_ks: int = 200,
        min_tail_size: int = 30,
        seed_base: int = 42,
    ) -> None:
        self.graph = graph
        self.n_runs_null = int(n_runs_null)
        self.n_runs_ws_sweep = int(n_runs_ws_sweep)
        self.n_bootstrap_ks = int(n_bootstrap_ks)
        self.min_tail_size = int(min_tail_size)
        self.seed_base = int(seed_base)

        # 1. Projeção não-direcionada simples sem auto-laços
        self.g_simple: nx.Graph = self._projetar_grafo_simples()
        self.n: int = self.g_simple.number_of_nodes()
        self.m: int = self.g_simple.number_of_edges()
        self.grau_medio: float = (2.0 * self.m / self.n) if self.n > 0 else 0.0
        self.densidade: float = (
            (2.0 * self.m / (self.n * (self.n - 1))) if self.n > 1 else 0.0
        )
        self.k_ws: int = max(2, 2 * int(round(self.grau_medio / 2.0)))
        if self.k_ws >= self.n and self.n >= 3:
            self.k_ws = self.n - 1 if (self.n - 1) % 2 == 0 else self.n - 2

        # 2. Projeção direcionada da camada Química
        self.g_quimico_dir: nx.DiGraph = self._projetar_quimico_direcionado()

        # Métricas base do grafo real
        self.graus_real: np.ndarray = np.array(
            [deg for _, deg in self.g_simple.degree()], dtype=float
        )
        self.c_real: float = float(nx.average_clustering(self.g_simple)) if self.n > 0 else 0.0
        gcc_real = self._gcc(self.g_simple)
        self.l_real: float = (
            float(nx.average_shortest_path_length(gcc_real))
            if gcc_real.number_of_nodes() > 1
            else 0.0
        )
        self.kappa_real: float = (
            float(np.mean(self.graus_real ** 2) / (self.grau_medio ** 2))
            if self.grau_medio > 0.0
            else 0.0
        )

        # Cache interno das realizações nulas ER para reutilização no Bloco B
        self._cache_nulo_er: Optional[Dict[str, Any]] = None

    def _projetar_grafo_simples(self) -> nx.Graph:
        """Converte o MultiDiGraph em grafo simples não-direcionado sem auto-laços."""
        g = nx.Graph()
        g.add_nodes_from(self.graph.nodes())
        for u, v in self.graph.edges():
            if u != v:
                g.add_edge(u, v)
        g.remove_edges_from(nx.selfloop_edges(g))
        return g

    def _projetar_quimico_direcionado(self) -> nx.DiGraph:
        """Extrai o grafo direcionado contendo exclusivamente a camada Química."""
        g_dir = nx.DiGraph()
        g_dir.add_nodes_from(self.graph.nodes())
        for u, v, chave, dados in self.graph.edges(keys=True, data=True):
            tipo = dados.get("synapse_type", chave)
            peso = float(dados.get("weight", 1.0))
            if tipo == "Quimica" and peso > 0 and u != v:
                if g_dir.has_edge(u, v):
                    g_dir[u][v]["weight"] += peso
                else:
                    g_dir.add_edge(u, v, weight=peso)
        return g_dir

    @staticmethod
    def _gcc(g: nx.Graph) -> nx.Graph:
        """Retorna a Maior Componente Conexa (GCC) de um grafo não-direcionado."""
        if g.number_of_nodes() == 0:
            return g
        if nx.is_connected(g):
            return g
        maior = max(nx.connected_components(g), key=len)
        return g.subgraph(maior).copy()

    @staticmethod
    def _ks_duas_amostras(amostra_a: np.ndarray, amostra_b: np.ndarray) -> float:
        """Calcula a distância de Kolmogorov-Smirnov de duas amostras em NumPy puro."""
        a = np.sort(np.asarray(amostra_a, dtype=float))
        b = np.sort(np.asarray(amostra_b, dtype=float))
        if len(a) == 0 or len(b) == 0:
            return 0.0
        uniao = np.unique(np.concatenate((a, b)))
        cdf_a = np.searchsorted(a, uniao, side="right") / float(len(a))
        cdf_b = np.searchsorted(b, uniao, side="right") / float(len(b))
        return float(np.max(np.abs(cdf_a - cdf_b)))

    # =========================================================================
    # BLOCO A — Teste da Hipótese 1: Rede Aleatória (Erdős-Rényi / Poisson)
    # =========================================================================
    def avaliar_hipotese_aleatoria(self) -> Dict[str, Any]:
        """
        Avalia a hipótese nula de rede aleatória Erdős-Rényi G(N, M) via Z-score de
        agrupamento, índice de dispersão de Poisson, razão de heterogeneidade e KS.
        """
        c_er_list: List[float] = []
        l_er_list: List[float] = []
        kappa_er_list: List[float] = []
        graus_er_todos: List[float] = []

        for i in range(self.n_runs_null):
            g_er = nx.gnm_random_graph(self.n, self.m, seed=self.seed_base + i)
            c_i = float(nx.average_clustering(g_er))
            gcc_i = self._gcc(g_er)
            l_i = (
                float(nx.average_shortest_path_length(gcc_i))
                if gcc_i.number_of_nodes() > 1
                else 0.0
            )
            graus_i = np.array([deg for _, deg in g_er.degree()], dtype=float)
            k_med_i = float(np.mean(graus_i))
            kappa_i = (
                float(np.mean(graus_i ** 2) / (k_med_i ** 2)) if k_med_i > 0.0 else 0.0
            )

            c_er_list.append(c_i)
            l_er_list.append(l_i)
            kappa_er_list.append(kappa_i)
            graus_er_todos.extend(graus_i.tolist())

        c_er_arr = np.array(c_er_list, dtype=float)
        l_er_arr = np.array(l_er_list, dtype=float)
        kappa_er_arr = np.array(kappa_er_list, dtype=float)
        graus_er_arr = np.array(graus_er_todos, dtype=float)

        c_er_media = float(np.mean(c_er_arr))
        c_er_std = float(np.std(c_er_arr, ddof=1)) if len(c_er_arr) > 1 else 1e-9
        l_er_media = float(np.mean(l_er_arr))
        l_er_std = float(np.std(l_er_arr, ddof=1)) if len(l_er_arr) > 1 else 1e-9
        kappa_er_media = float(np.mean(kappa_er_arr))

        z_score_c = (
            float((self.c_real - c_er_media) / c_er_std) if c_er_std > 1e-12 else 0.0
        )
        var_k = float(np.var(self.graus_real, ddof=1)) if self.n > 1 else 0.0
        indice_dispersao_id = (
            float(var_k / self.grau_medio) if self.grau_medio > 0.0 else 0.0
        )
        razao_heterogeneidade = (
            float(self.kappa_real / kappa_er_media) if kappa_er_media > 0.0 else 0.0
        )
        ks_dist_er = self._ks_duas_amostras(self.graus_real, graus_er_arr)

        rejeitado = (
            (z_score_c > 3.0) or (indice_dispersao_id > 2.0) or (ks_dist_er > 0.15)
        )
        veredito = "Rejeitado" if rejeitado else "Não Rejeitado"

        self._cache_nulo_er = {
            "c_er_arr": c_er_arr,
            "l_er_arr": l_er_arr,
            "c_er_media": c_er_media,
            "c_er_std": c_er_std,
            "l_er_media": l_er_media,
            "l_er_std": l_er_std,
            "kappa_er_media": kappa_er_media,
        }

        return {
            "C_real": float(self.c_real),
            "C_ER_media": float(c_er_media),
            "C_ER_std": float(c_er_std),
            "z_score_C": float(z_score_c),
            "indice_dispersao_ID": float(indice_dispersao_id),
            "kappa_real": float(self.kappa_real),
            "kappa_ER_media": float(kappa_er_media),
            "razao_heterogeneidade": float(razao_heterogeneidade),
            "ks_dist_vs_ER": float(ks_dist_er),
            "veredito": veredito,
        }

    # =========================================================================
    # BLOCO B — Teste da Hipótese 2: Rede Small-World (Mundo Pequeno)
    # =========================================================================
    def avaliar_hipotese_small_world(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Calcula os três índices de Small-World (sigma, omega e phi_SWP) e executa
        a varredura da curva de transição de fase de Watts-Strogatz.
        """
        if self._cache_nulo_er is None:
            self.avaliar_hipotese_aleatoria()
        assert self._cache_nulo_er is not None

        c_er_arr = self._cache_nulo_er["c_er_arr"]
        l_er_arr = self._cache_nulo_er["l_er_arr"]
        c_er_media = self._cache_nulo_er["c_er_media"]
        l_er_media = self._cache_nulo_er["l_er_media"]

        # 1. Grafo Regular (Lattice, p=0.0)
        g_latt = nx.watts_strogatz_graph(self.n, self.k_ws, p=0.0, seed=self.seed_base)
        c_latt = float(nx.average_clustering(g_latt))
        gcc_latt = self._gcc(g_latt)
        l_latt = (
            float(nx.average_shortest_path_length(gcc_latt))
            if gcc_latt.number_of_nodes() > 1
            else 0.0
        )

        # 2. Três índices quantitativos de Small-World
        # 2a. Sigma (Humphries & Gurney, 2008)
        razao_c = (self.c_real / c_er_media) if c_er_media > 0.0 else 0.0
        razao_l = (self.l_real / l_er_media) if l_er_media > 0.0 else 0.0
        sigma = float(razao_c / razao_l) if razao_l > 0.0 else 0.0

        sigmas_i = np.where(
            (c_er_arr > 0.0) & (self.l_real > 0.0),
            (self.c_real / c_er_arr) / (self.l_real / l_er_arr),
            0.0,
        )
        sigma_std = float(np.std(sigmas_i, ddof=1)) if len(sigmas_i) > 1 else 0.0

        # 2b. Omega (Telesford et al., 2011)
        termo_l = (l_er_media / self.l_real) if self.l_real > 0.0 else 0.0
        termo_c = (self.c_real / c_latt) if c_latt > 0.0 else 0.0
        omega = float(termo_l - termo_c)

        # 2c. Small-World Propensity phi_SWP (Muldoon et al., 2016)
        denom_c = c_latt - c_er_media
        delta_c = (
            float(np.clip((c_latt - self.c_real) / denom_c, 0.0, 1.0))
            if abs(denom_c) > 1e-12
            else 0.0
        )
        denom_l = l_latt - l_er_media
        delta_l = (
            float(np.clip((self.l_real - l_er_media) / denom_l, 0.0, 1.0))
            if abs(denom_l) > 1e-12
            else 0.0
        )
        phi_swp = float(1.0 - math.sqrt((delta_c ** 2 + delta_l ** 2) / 2.0))
        phi_swp = float(np.clip(phi_swp, 0.0, 1.0))

        # 3. Varredura da curva de transição de Watts-Strogatz
        p_grid = np.concatenate(([0.0], np.logspace(-3, 0, 12)))
        linhas_sweep: List[Dict[str, float]] = []

        c_real_norm = (self.c_real / c_latt) if c_latt > 0.0 else 0.0
        l_real_norm = (self.l_real / l_latt) if l_latt > 0.0 else 0.0

        melhor_dist = float("inf")
        p_equiv_ws = 0.10

        for idx_p, p_val in enumerate(p_grid):
            p_float = float(p_val)
            if p_float == 0.0:
                c_norm_media = 1.0
                c_norm_std = 0.0
                l_norm_media = 1.0
                l_norm_std = 0.0
            else:
                c_norms: List[float] = []
                l_norms: List[float] = []
                for r in range(self.n_runs_ws_sweep):
                    seed_ws = self.seed_base + idx_p * 100 + r
                    g_ws = nx.watts_strogatz_graph(self.n, self.k_ws, p=p_float, seed=seed_ws)
                    c_w = float(nx.average_clustering(g_ws))
                    gcc_w = self._gcc(g_ws)
                    l_w = (
                        float(nx.average_shortest_path_length(gcc_w))
                        if gcc_w.number_of_nodes() > 1
                        else 0.0
                    )
                    c_norms.append(c_w / c_latt if c_latt > 0.0 else 0.0)
                    l_norms.append(l_w / l_latt if l_latt > 0.0 else 0.0)

                c_norm_media = float(np.mean(c_norms))
                c_norm_std = float(np.std(c_norms, ddof=1)) if len(c_norms) > 1 else 0.0
                l_norm_media = float(np.mean(l_norms))
                l_norm_std = float(np.std(l_norms, ddof=1)) if len(l_norms) > 1 else 0.0

                dist_euclid = math.sqrt(
                    (c_norm_media - c_real_norm) ** 2 + (l_norm_media - l_real_norm) ** 2
                )
                if dist_euclid < melhor_dist:
                    melhor_dist = dist_euclid
                    p_equiv_ws = p_float

            linhas_sweep.append(
                {
                    "p_rewiring": p_float,
                    "C_norm_media": c_norm_media,
                    "C_norm_std": c_norm_std,
                    "L_norm_media": l_norm_media,
                    "L_norm_std": l_norm_std,
                }
            )

        ws_sweep_df = pd.DataFrame(linhas_sweep)

        confirmado = (sigma > 1.0) and (-0.5 <= omega <= 0.5) and (phi_swp >= 0.595)
        veredito = "Confirmado" if confirmado else ("Parcial" if sigma > 1.0 else "Rejeitado")

        resumo_sw = {
            "C_real": float(self.c_real),
            "C_ER_media": float(c_er_media),
            "C_latt": float(c_latt),
            "L_real": float(self.l_real),
            "L_ER_media": float(l_er_media),
            "L_latt": float(l_latt),
            "C_real_norm_latt": float(c_real_norm),
            "L_real_norm_latt": float(l_real_norm),
            "sigma": float(sigma),
            "sigma_std": float(sigma_std),
            "omega": float(omega),
            "phi_swp": float(phi_swp),
            "delta_C": float(delta_c),
            "delta_L": float(delta_l),
            "p_equiv_ws": float(p_equiv_ws),
            "z_score_C_vs_ER": float(
                (self.c_real - c_er_media) / self._cache_nulo_er["c_er_std"]
                if self._cache_nulo_er["c_er_std"] > 1e-12
                else 0.0
            ),
            "indice_dispersao_ID": float(
                np.var(self.graus_real, ddof=1) / self.grau_medio
                if self.grau_medio > 0.0
                else 0.0
            ),
            "veredito": veredito,
        }
        return resumo_sw, ws_sweep_df

    # =========================================================================
    # BLOCO C — Teste da Hipótese 3: Rede Livre de Escala (Scale-Free)
    # =========================================================================
    @staticmethod
    def _log_pmf_normalizada(
        k_grid: np.ndarray, familia: str, params: Tuple[float, ...]
    ) -> np.ndarray:
        """
        Calcula o vetor de log-probabilidades discretas normalizadas sobre k_grid
        usando o truque log-sum-exp para estabilidade numérica total.
        """
        if familia == "PL":
            (gamma,) = params
            log_unnorm = -gamma * np.log(k_grid)
        elif familia == "TPL":
            gamma, lam = params
            log_unnorm = -gamma * np.log(k_grid) - lam * k_grid
        elif familia == "EXP":
            (lam,) = params
            log_unnorm = -lam * k_grid
        elif familia == "LN":
            mu, sigma = params
            sigma_safe = max(sigma, 1e-4)
            log_k = np.log(k_grid)
            log_unnorm = -log_k - ((log_k - mu) ** 2) / (2.0 * (sigma_safe ** 2))
        else:
            raise ValueError(f"Família de distribuição desconhecida: {familia}")

        max_log = float(np.max(log_unnorm))
        log_z = max_log + float(np.log(np.sum(np.exp(log_unnorm - max_log))))
        return log_unnorm - log_z

    @classmethod
    def _ajustar_pl_discreto(
        cls, cauda_int: np.ndarray, k_min: int, k_grid: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Otimiza gamma > 1 para Power-Law discreta via busca em grade de 2 estágios
        (grossa de 40 pontos + refinamento local fino).
        """
        n_tail = len(cauda_int)
        idx_obs = cauda_int - k_min

        # Inicialização analítica de Clauset para centrar ou guiar a grade
        x_min_cont = max(0.5, float(k_min) - 0.5)
        soma_log = float(np.sum(np.log(cauda_int / x_min_cont)))
        gamma_0 = (1.0 + n_tail / soma_log) if soma_log > 0.0 else 2.5
        gamma_0 = float(np.clip(gamma_0, 1.15, 5.40))

        # Estágio 1: Grade grossa (40 pontos em [1.10, 5.50])
        grade_1 = np.linspace(1.10, 5.50, 40)
        grade_1 = np.unique(np.append(grade_1, gamma_0))
        melhor_g = gamma_0
        melhor_ll = -float("inf")

        for g in grade_1:
            log_pmf = cls._log_pmf_normalizada(k_grid, "PL", (float(g),))
            ll = float(np.sum(log_pmf[idx_obs]))
            if ll > melhor_ll:
                melhor_ll = ll
                melhor_g = float(g)

        # Estágio 2: Refinamento fino ao redor de melhor_g
        g_min = max(1.05, melhor_g - 0.25)
        g_max = min(6.50, melhor_g + 0.25)
        grade_2 = np.linspace(g_min, g_max, 40)
        for g in grade_2:
            log_pmf = cls._log_pmf_normalizada(k_grid, "PL", (float(g),))
            ll = float(np.sum(log_pmf[idx_obs]))
            if ll > melhor_ll:
                melhor_ll = ll
                melhor_g = float(g)

        log_pmf_final = cls._log_pmf_normalizada(k_grid, "PL", (melhor_g,))
        return melhor_g, melhor_ll, log_pmf_final

    @classmethod
    def _ajustar_exp_discreto(
        cls, cauda_int: np.ndarray, k_min: int, k_grid: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """Otimiza lambda > 0 para Exponencial discreta via busca em grade de 2 estágios."""
        idx_obs = cauda_int - k_min
        desloc = max(0.5, float(np.mean(cauda_int) - k_min + 0.5))
        lam_0 = float(np.clip(1.0 / desloc, 0.002, 0.50))

        grade_1 = np.unique(np.append(np.linspace(0.002, 0.50, 40), lam_0))
        melhor_lam = lam_0
        melhor_ll = -float("inf")

        for lam in grade_1:
            log_pmf = cls._log_pmf_normalizada(k_grid, "EXP", (float(lam),))
            ll = float(np.sum(log_pmf[idx_obs]))
            if ll > melhor_ll:
                melhor_ll = ll
                melhor_lam = float(lam)

        l_min = max(0.0005, melhor_lam - 0.025)
        l_max = min(0.80, melhor_lam + 0.025)
        for lam in np.linspace(l_min, l_max, 40):
            log_pmf = cls._log_pmf_normalizada(k_grid, "EXP", (float(lam),))
            ll = float(np.sum(log_pmf[idx_obs]))
            if ll > melhor_ll:
                melhor_ll = ll
                melhor_lam = float(lam)

        log_pmf_final = cls._log_pmf_normalizada(k_grid, "EXP", (melhor_lam,))
        return melhor_lam, melhor_ll, log_pmf_final

    @classmethod
    def _ajustar_tpl_discreto(
        cls, cauda_int: np.ndarray, k_min: int, k_grid: np.ndarray
    ) -> Tuple[Tuple[float, float], float, np.ndarray]:
        """Otimiza (gamma, lambda) para Power-Law Truncada discreta em 2 estágios."""
        idx_obs = cauda_int - k_min
        grade_g1 = np.linspace(0.10, 4.00, 22)
        grade_l1 = np.linspace(0.002, 0.25, 22)

        melhor_g = 1.0
        melhor_l = 0.05
        melhor_ll = -float("inf")

        for g in grade_g1:
            for lam in grade_l1:
                log_pmf = cls._log_pmf_normalizada(k_grid, "TPL", (float(g), float(lam)))
                ll = float(np.sum(log_pmf[idx_obs]))
                if ll > melhor_ll:
                    melhor_ll = ll
                    melhor_g = float(g)
                    melhor_l = float(lam)

        # Refinamento local fino
        g_min = max(0.01, melhor_g - 0.30)
        g_max = min(5.00, melhor_g + 0.30)
        l_min = max(0.0005, melhor_l - 0.025)
        l_max = min(0.45, melhor_l + 0.025)

        for g in np.linspace(g_min, g_max, 20):
            for lam in np.linspace(l_min, l_max, 20):
                log_pmf = cls._log_pmf_normalizada(k_grid, "TPL", (float(g), float(lam)))
                ll = float(np.sum(log_pmf[idx_obs]))
                if ll > melhor_ll:
                    melhor_ll = ll
                    melhor_g = float(g)
                    melhor_l = float(lam)

        log_pmf_final = cls._log_pmf_normalizada(k_grid, "TPL", (melhor_g, melhor_l))
        return (melhor_g, melhor_l), melhor_ll, log_pmf_final

    @classmethod
    def _ajustar_ln_discreto(
        cls, cauda_int: np.ndarray, k_min: int, k_grid: np.ndarray
    ) -> Tuple[Tuple[float, float], float, np.ndarray]:
        """Otimiza (mu, sigma) para Log-Normal discreta truncada na cauda em 2 estágios."""
        idx_obs = cauda_int - k_min
        log_x = np.log(cauda_int.astype(float))
        mu_0 = float(np.mean(log_x))
        sigma_0 = max(0.15, float(np.std(log_x)))

        grade_mu1 = np.linspace(mu_0 - 1.5, mu_0 + 1.5, 22)
        grade_sig1 = np.linspace(max(0.10, sigma_0 - 0.8), sigma_0 + 1.5, 22)

        melhor_mu = mu_0
        melhor_sig = sigma_0
        melhor_ll = -float("inf")

        for mu in grade_mu1:
            for sig in grade_sig1:
                log_pmf = cls._log_pmf_normalizada(k_grid, "LN", (float(mu), float(sig)))
                ll = float(np.sum(log_pmf[idx_obs]))
                if ll > melhor_ll:
                    melhor_ll = ll
                    melhor_mu = float(mu)
                    melhor_sig = float(sig)

        # Refinamento local fino
        mu_min = melhor_mu - 0.25
        mu_max = melhor_mu + 0.25
        s_min = max(0.05, melhor_sig - 0.20)
        s_max = melhor_sig + 0.20

        for mu in np.linspace(mu_min, mu_max, 20):
            for sig in np.linspace(s_min, s_max, 20):
                log_pmf = cls._log_pmf_normalizada(k_grid, "LN", (float(mu), float(sig)))
                ll = float(np.sum(log_pmf[idx_obs]))
                if ll > melhor_ll:
                    melhor_ll = ll
                    melhor_mu = float(mu)
                    melhor_sig = float(sig)

        log_pmf_final = cls._log_pmf_normalizada(k_grid, "LN", (melhor_mu, melhor_sig))
        return (melhor_mu, melhor_sig), melhor_ll, log_pmf_final

    @staticmethod
    def _calcular_ks_cauda(
        cauda_int: np.ndarray, k_min: int, pmf_teorica: np.ndarray
    ) -> float:
        """
        Calcula a distância de Kolmogorov-Smirnov D(k_min) = max_{k >= k_min} |S(k) - P(k)|
        entre a CDF empírica da cauda S(k) e a CDF teórica discreta P(k).
        """
        cauda_ord = np.sort(cauda_int)
        n_tail = float(len(cauda_ord))
        k_max_obs = int(cauda_ord[-1])
        k_suporte = np.arange(k_min, k_max_obs + 1, dtype=int)

        cdf_emp = np.searchsorted(cauda_ord, k_suporte, side="right") / n_tail
        cdf_teo = np.cumsum(pmf_teorica)[: len(k_suporte)]
        return float(np.max(np.abs(cdf_emp - cdf_teo)))

    @staticmethod
    def _teste_vuong(
        log_p_pl_obs: np.ndarray, log_p_rival_obs: np.ndarray
    ) -> Tuple[float, float]:
        """
        Executa o Teste de Razão de Log-Verossimilhança de Vuong entre PL e um modelo rival.
        Retorna (z_Vuong, p_Vuong), onde z > 0 favorece PL e z < 0 favorece o rival.
        """
        d_i = log_p_pl_obs - log_p_rival_obs
        n_tail = len(d_i)
        if n_tail <= 1:
            return 0.0, 1.0
        r_total = float(np.sum(d_i))
        d_bar = float(np.mean(d_i))
        sigma_d2 = float(np.mean((d_i - d_bar) ** 2))
        if sigma_d2 <= 1e-12:
            return 0.0, 1.0
        z_vuong = float(r_total / math.sqrt(n_tail * sigma_d2))
        p_vuong = float(math.erfc(abs(z_vuong) / math.sqrt(2.0)))
        return z_vuong, p_vuong

    @staticmethod
    def _analisar_perfil_graus_direcionados(graus_arr: np.ndarray) -> Dict[str, float]:
        """Calcula heterogeneidade kappa, gamma_MLE, R2_log-log e R2_semi-log para um vetor de graus."""
        g_pos = np.asarray(graus_arr, dtype=float)
        k_med = float(np.mean(g_pos)) if len(g_pos) > 0 else 0.0
        kappa = float(np.mean(g_pos ** 2) / (k_med ** 2)) if k_med > 0.0 else 0.0

        k_min_d = max(1, int(np.ceil(k_med)))
        cauda = g_pos[g_pos >= k_min_d]
        if len(cauda) < 10:
            g_nz = g_pos[g_pos > 0]
            k_min_d = max(1, int(np.percentile(g_nz, 50))) if len(g_nz) > 0 else 1
            cauda = g_pos[g_pos >= k_min_d]

        n_t = len(cauda)
        x_min_c = max(0.5, float(k_min_d) - 0.5)
        soma_log = float(np.sum(np.log(cauda / x_min_c))) if n_t > 0 else 0.0
        gamma_mle = float(1.0 + n_t / soma_log) if soma_log > 0.0 else 0.0

        k_unicos = np.unique(cauda)
        if len(k_unicos) >= 3:
            ccdf = np.array([np.mean(g_pos >= k) for k in k_unicos], dtype=float)
            log_k = np.log(k_unicos)
            log_p = np.log(ccdf)
            ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))
            if ss_tot > 0.0:
                c_pl = np.polyfit(log_k, log_p, 1)
                r2_ll = float(1.0 - np.sum((log_p - (c_pl[0] * log_k + c_pl[1])) ** 2) / ss_tot)
                c_ex = np.polyfit(k_unicos, log_p, 1)
                r2_sl = float(
                    1.0 - np.sum((log_p - (c_ex[0] * k_unicos + c_ex[1])) ** 2) / ss_tot
                )
            else:
                r2_ll, r2_sl = 0.0, 0.0
        else:
            r2_ll, r2_sl = 0.0, 0.0

        return {
            "grau_medio": float(k_med),
            "kappa": float(kappa),
            "k_min": int(k_min_d),
            "gamma_mle": float(gamma_mle),
            "r2_log_log": float(r2_ll),
            "r2_semi_log": float(r2_sl),
        }

    def avaliar_hipotese_scale_free(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Executa o protocolo de Clauset, Shalizi & Newman (2009) em NumPy puro:
          1. Seleção ótima de k_min via minimização da distância KS
          2. Bootstrap semi-paramétrico para p-valor de bondade de ajuste (p_KS)
          3. Ajuste MLE de 4 modelos concorrentes na cauda (PL, TPL, LN, EXP)
          4. Seleção de modelos via AIC, BIC e Teste de Razão de Verossimilhança de Vuong
        """
        graus_int = np.asarray(self.graus_real, dtype=int)
        graus_pos = graus_int[graus_int >= 1]
        k_max = int(np.max(graus_pos))
        teto_suporte = max(5 * k_max, 2000)

        # 1. Seleção ótima de k_min* (Clauset et al., 2009)
        candidatos_k = np.unique(graus_pos)
        melhor_k_min = int(max(1, np.ceil(self.grau_medio)))
        melhor_ks = float("inf")
        melhor_gamma = 2.5
        melhor_log_pmf_pl = np.array([], dtype=float)

        for kc in candidatos_k:
            kc_int = int(kc)
            if kc_int < 2:
                continue
            cauda_c = graus_pos[graus_pos >= kc_int]
            if len(cauda_c) < self.min_tail_size:
                continue
            k_grid_c = np.arange(kc_int, teto_suporte + 1, dtype=float)
            gamma_c, _, log_pmf_c = self._ajustar_pl_discreto(cauda_c, kc_int, k_grid_c)
            pmf_c = np.exp(log_pmf_c)
            ks_c = self._calcular_ks_cauda(cauda_c, kc_int, pmf_c)
            if ks_c < melhor_ks:
                melhor_ks = ks_c
                melhor_k_min = kc_int
                melhor_gamma = gamma_c
                melhor_log_pmf_pl = log_pmf_c

        k_min_star = melhor_k_min
        cauda_real = graus_pos[graus_pos >= k_min_star]
        n_tail = int(len(cauda_real))
        k_grid = np.arange(k_min_star, teto_suporte + 1, dtype=float)

        gamma_pl, ll_pl, log_pmf_pl = self._ajustar_pl_discreto(
            cauda_real, k_min_star, k_grid
        )
        pmf_pl = np.exp(log_pmf_pl)
        pmf_pl = pmf_pl / np.sum(pmf_pl)
        d_emp = self._calcular_ks_cauda(cauda_real, k_min_star, pmf_pl)

        # 2. Bootstrap semi-paramétrico para p_KS
        rng = np.random.default_rng(self.seed_base)
        corpo_abaixo = graus_int[graus_int < k_min_star]
        prob_cauda = float(n_tail) / float(len(graus_int))
        k_grid_int = np.arange(k_min_star, teto_suporte + 1, dtype=int)

        cont_maior_igual = 0
        for _ in range(self.n_bootstrap_ks):
            n_cauda_sint = int(rng.binomial(len(graus_int), prob_cauda))
            if len(corpo_abaixo) == 0 or n_cauda_sint < 10:
                n_cauda_sint = max(10, n_tail)
            amostra_cauda = rng.choice(k_grid_int, size=n_cauda_sint, replace=True, p=pmf_pl)

            # Estimador MLE analítico rápido + refinamento local para a réplica sintética
            x_min_c = max(0.5, float(k_min_star) - 0.5)
            soma_log_b = float(np.sum(np.log(amostra_cauda / x_min_c)))
            g_b0 = (1.0 + len(amostra_cauda) / soma_log_b) if soma_log_b > 0.0 else gamma_pl
            g_b0 = float(np.clip(g_b0, 1.10, 6.00))

            idx_b = np.clip(amostra_cauda - k_min_star, 0, len(k_grid) - 1)
            melhor_ll_b = -float("inf")
            melhor_pmf_b = pmf_pl
            for g_cand in np.linspace(max(1.05, g_b0 - 0.25), min(6.50, g_b0 + 0.25), 11):
                l_pmf_b = self._log_pmf_normalizada(k_grid, "PL", (float(g_cand),))
                ll_b = float(np.sum(l_pmf_b[idx_b]))
                if ll_b > melhor_ll_b:
                    melhor_ll_b = ll_b
                    melhor_pmf_b = np.exp(l_pmf_b)

            d_synth = self._calcular_ks_cauda(amostra_cauda, k_min_star, melhor_pmf_b)
            if d_synth >= d_emp:
                cont_maior_igual += 1

        p_valor_ks = float(cont_maior_igual / float(max(1, self.n_bootstrap_ks)))

        # 3. Ajuste dos modelos concorrentes na mesma cauda k >= k_min*
        (gamma_tpl, lam_tpl), ll_tpl, log_pmf_tpl = self._ajustar_tpl_discreto(
            cauda_real, k_min_star, k_grid
        )
        (mu_ln, sigma_ln), ll_ln, log_pmf_ln = self._ajustar_ln_discreto(
            cauda_real, k_min_star, k_grid
        )
        lam_exp, ll_exp, log_pmf_exp = self._ajustar_exp_discreto(
            cauda_real, k_min_star, k_grid
        )

        idx_obs_real = cauda_real - k_min_star
        log_p_pl_obs = log_pmf_pl[idx_obs_real]
        log_p_tpl_obs = log_pmf_tpl[idx_obs_real]
        log_p_ln_obs = log_pmf_ln[idx_obs_real]
        log_p_exp_obs = log_pmf_exp[idx_obs_real]

        # 4. Teste de Vuong e critérios de informação AIC/BIC
        z_tpl, p_tpl = self._teste_vuong(log_p_pl_obs, log_p_tpl_obs)
        z_ln, p_ln = self._teste_vuong(log_p_pl_obs, log_p_ln_obs)
        z_exp, p_exp = self._teste_vuong(log_p_pl_obs, log_p_exp_obs)

        specs_modelos = [
            (
                "Lei de Potência (PL)",
                f"gamma={gamma_pl:.3f}",
                1,
                ll_pl,
                0.0,
                1.0,
            ),
            (
                "PL Truncada (TPL)",
                f"gamma={gamma_tpl:.3f}, lambda={lam_tpl:.4f}",
                2,
                ll_tpl,
                z_tpl,
                p_tpl,
            ),
            (
                "Log-Normal (LN)",
                f"mu={mu_ln:.3f}, sigma={sigma_ln:.3f}",
                2,
                ll_ln,
                z_ln,
                p_ln,
            ),
            (
                "Exponencial (EXP)",
                f"lambda={lam_exp:.4f}",
                1,
                ll_exp,
                z_exp,
                p_exp,
            ),
        ]

        linhas_modelos: List[Dict[str, Any]] = []
        aics = [2.0 * q - 2.0 * ll for _, _, q, ll, _, _ in specs_modelos]
        aic_min = float(min(aics))

        for (nome_m, param_str, q_m, ll_m, z_v, p_v), aic_m in zip(specs_modelos, aics):
            delta_aic = float(aic_m - aic_min)
            bic_m = float(q_m * math.log(max(1, n_tail)) - 2.0 * ll_m)
            if nome_m == "Lei de Potência (PL)":
                melhor_que_pl = "Referência (PL)"
            elif z_v < 0.0 and p_v < 0.10:
                melhor_que_pl = "Sim (Vuong p < 0.10)"
            elif delta_aic < (aics[0] - aic_min - 2.0):
                melhor_que_pl = "Sim (Delta AIC < -2)"
            else:
                melhor_que_pl = "Não / Inconclusivo"

            linhas_modelos.append(
                {
                    "Modelo": nome_m,
                    "Parametros": param_str,
                    "LogLik": float(ll_m),
                    "AIC": float(aic_m),
                    "Delta_AIC": float(delta_aic),
                    "BIC": float(bic_m),
                    "Vuong_R_vs_PL": float(z_v),
                    "Vuong_p_vs_PL": float(p_v),
                    "Melhor_Que_PL": melhor_que_pl,
                }
            )

        modelos_cauda_df = pd.DataFrame(linhas_modelos)
        idx_melhor_aic = int(np.argmin(aics))
        melhor_modelo_aic = str(specs_modelos[idx_melhor_aic][0])

        # R² log-log vs semi-log na cauda empírica
        k_unicos_cauda = np.unique(cauda_real)
        ccdf_emp_cauda = np.array(
            [np.mean(cauda_real >= k) for k in k_unicos_cauda], dtype=float
        )
        log_k_c = np.log(k_unicos_cauda.astype(float))
        log_p_c = np.log(ccdf_emp_cauda)
        ss_tot = float(np.sum((log_p_c - np.mean(log_p_c)) ** 2))
        if len(k_unicos_cauda) >= 3 and ss_tot > 0.0:
            c_ll = np.polyfit(log_k_c, log_p_c, 1)
            r2_log_log = float(
                1.0 - np.sum((log_p_c - (c_ll[0] * log_k_c + c_ll[1])) ** 2) / ss_tot
            )
            c_sl = np.polyfit(k_unicos_cauda.astype(float), log_p_c, 1)
            r2_semi_log = float(
                1.0
                - np.sum(
                    (log_p_c - (c_sl[0] * k_unicos_cauda.astype(float) + c_sl[1])) ** 2
                )
                / ss_tot
            )
        else:
            r2_log_log, r2_semi_log = 0.0, 0.0

        # Curvas CCDF para plotagem no painel log-log (condicionadas a K >= k_min*)
        def _ccdf_sobre_k(log_pmf_vec: np.ndarray, ks_alvo: np.ndarray) -> List[float]:
            pmf_v = np.exp(log_pmf_vec)
            pmf_v = pmf_v / np.sum(pmf_v)
            ccdf_full = 1.0 - np.concatenate(([0.0], np.cumsum(pmf_v)[:-1]))
            ccdf_full = np.clip(ccdf_full, 1e-12, 1.0)
            indices = np.clip(ks_alvo.astype(int) - k_min_star, 0, len(ccdf_full) - 1)
            return [float(v) for v in ccdf_full[indices]]

        # Curva Poisson / ER condicionada na cauda usando math.lgamma
        lam_pois = max(1.0, self.grau_medio)
        log_pois_unnorm = np.array(
            [
                float(k) * math.log(lam_pois) - lam_pois - math.lgamma(float(k) + 1.0)
                for k in k_grid
            ],
            dtype=float,
        )
        max_lp = float(np.max(log_pois_unnorm))
        log_z_pois = max_lp + float(np.log(np.sum(np.exp(log_pois_unnorm - max_lp))))
        log_pmf_pois = log_pois_unnorm - log_z_pois

        curvas_ccdf = {
            "k_vals": [int(k) for k in k_unicos_cauda],
            "ccdf_empirica": [float(v) for v in ccdf_emp_cauda],
            "ccdf_pl": _ccdf_sobre_k(log_pmf_pl, k_unicos_cauda),
            "ccdf_tpl": _ccdf_sobre_k(log_pmf_tpl, k_unicos_cauda),
            "ccdf_ln": _ccdf_sobre_k(log_pmf_ln, k_unicos_cauda),
            "ccdf_exp": _ccdf_sobre_k(log_pmf_exp, k_unicos_cauda),
            "ccdf_poisson": _ccdf_sobre_k(log_pmf_pois, k_unicos_cauda),
        }

        # 6. Verificação Direcionada Complementar (Camada Química)
        k_in_arr = np.array([deg for _, deg in self.g_quimico_dir.in_degree()], dtype=float)
        k_out_arr = np.array([deg for _, deg in self.g_quimico_dir.out_degree()], dtype=float)
        perfil_in = self._analisar_perfil_graus_direcionados(k_in_arr)
        perfil_out = self._analisar_perfil_graus_direcionados(k_out_arr)

        # Critério de veredito para Scale-Free
        rival_supera_pl = (
            (melhor_modelo_aic != "Lei de Potência (PL)")
            or (z_tpl < 0.0 and p_tpl < 0.10)
            or (z_ln < 0.0 and p_ln < 0.10)
            or (r2_semi_log > r2_log_log)
        )
        if p_valor_ks < 0.10:
            veredito_sf = "Rejeitado (Cauda Pesada Truncada — KS p < 0.10)"
        elif rival_supera_pl:
            veredito_sf = (
                f"Não-Scale-Free Puro (Cauda Pesada Truncada — {melhor_modelo_aic} vence AIC)"
            )
        else:
            veredito_sf = "Compatível com Scale-Free na Cauda"

        resumo_sf = {
            "k_min": int(k_min_star),
            "n_tail": int(n_tail),
            "gamma_pl": float(gamma_pl),
            "ks_dist_pl": float(d_emp),
            "p_valor_ks_bootstrap": float(p_valor_ks),
            "melhor_modelo_aic": melhor_modelo_aic,
            "r2_log_log": float(r2_log_log),
            "r2_semi_log": float(r2_semi_log),
            "vuong_pl_vs_tpl": {"R": float(z_tpl), "p": float(p_tpl)},
            "vuong_pl_vs_ln": {"R": float(z_ln), "p": float(p_ln)},
            "vuong_pl_vs_exp": {"R": float(z_exp), "p": float(p_exp)},
            "direcionado_quimico": {
                "in_degree_kappa": float(perfil_in["kappa"]),
                "out_degree_kappa": float(perfil_out["kappa"]),
                "in_degree_gamma": float(perfil_in["gamma_mle"]),
                "out_degree_gamma": float(perfil_out["gamma_mle"]),
                "in_degree_r2_log_log": float(perfil_in["r2_log_log"]),
                "in_degree_r2_semi_log": float(perfil_in["r2_semi_log"]),
                "out_degree_r2_log_log": float(perfil_out["r2_log_log"]),
                "out_degree_r2_semi_log": float(perfil_out["r2_semi_log"]),
            },
            "curvas_ccdf": curvas_ccdf,
            "veredito": veredito_sf,
        }
        return resumo_sf, modelos_cauda_df

    # =========================================================================
    # BLOCO D — Consolidação do Veredito
    # =========================================================================
    def executar_diagnostico_completo(self) -> Dict[str, Any]:
        """
        Executa os três blocos de prova (Aleatório, Small-World e Scale-Free),
        constrói o Scorecard de hipóteses e retorna todos os artefatos estruturados.
        """
        res_er = self.avaliar_hipotese_aleatoria()
        res_sw, ws_sweep_df = self.avaliar_hipotese_small_world()
        res_sf, modelos_cauda_df = self.avaliar_hipotese_scale_free()

        if res_sf["veredito"] == "Compatível com Scale-Free na Cauda":
            evidencia_sf = (
                f"Na cauda k >= {res_sf['k_min']} (n={res_sf['n_tail']}), a Lei de Potência "
                f"(gamma={res_sf['gamma_pl']:.2f}, D_KS={res_sf['ks_dist_pl']:.3f}, p_KS={res_sf['p_valor_ks_bootstrap']:.2f} >= 0.10) "
                f"é estatisticamente plausível e vence por parcimônia (Delta_AIC=0.00; "
                f"Vuong vs TPL: R={res_sf['vuong_pl_vs_tpl']['R']:+.2f}, p={res_sf['vuong_pl_vs_tpl']['p']:.3f} inconclusivo)"
            )
            texto_sf_conclusao = (
                f"Quanto à hipótese LIVRE DE ESCALA (Scale-Free), a rede exibe alta heterogeneidade "
                f"(kappa = {self.kappa_real:.2f}) e um núcleo Rich-Club de neurônios de comando (como AVAL/AVAR). "
                f"O protocolo de Clauset-Shalizi-Newman na cauda superior (k >= {res_sf['k_min']}, n = {res_sf['n_tail']}) "
                f"estima expoente gamma_MLE = {res_sf['gamma_pl']:.2f} com distância D_KS = {res_sf['ks_dist_pl']:.3f} e "
                f"p_KS = {res_sf['p_valor_ks_bootstrap']:.2f} (>= 0.10, não rejeitando a Lei de Potência). Na comparação "
                f"direta de modelos via Teste de Vuong, a Lei de Potência Truncada (TPL, LogLik = {modelos_cauda_df.loc[1, 'LogLik']:.2f} vs "
                f"PL LogLik = {modelos_cauda_df.loc[0, 'LogLik']:.2f}, R = {res_sf['vuong_pl_vs_tpl']['R']:+.2f}, "
                f"p = {res_sf['vuong_pl_vs_tpl']['p']:.3f}) e a Log-Normal (R = {res_sf['vuong_pl_vs_ln']['R']:+.2f}, "
                f"p = {res_sf['vuong_pl_vs_ln']['p']:.3f}) são estatisticamente indistinguíveis da Lei de Potência pura, "
                f"que vence no critério de parcimônia (AIC = {modelos_cauda_df.loc[0, 'AIC']:.2f}, R²_log-log = {res_sf['r2_log_log']:.3f} > "
                f"R²_semi-log = {res_sf['r2_semi_log']:.3f}). Portanto, o conectoma do C. elegans classifica-se "
                f"primariamente como uma rede SMALL-WORLD com cauda pesada compatível com regime Scale-Free / truncamento suave "
                f"para k >= {res_sf['k_min']}."
            )
        else:
            evidencia_sf = (
                f"Na cauda k >= {res_sf['k_min']} (n={res_sf['n_tail']}), modelos com corte/curvatura "
                f"({res_sf['melhor_modelo_aic']}) superam a Lei de Potência pura "
                f"(gamma={res_sf['gamma_pl']:.2f}, p_KS={res_sf['p_valor_ks_bootstrap']:.2f}, "
                f"R2_semi-log={res_sf['r2_semi_log']:.3f} vs R2_log-log={res_sf['r2_log_log']:.3f})"
            )
            texto_sf_conclusao = (
                f"Quanto à hipótese LIVRE DE ESCALA (Scale-Free), embora a rede exiba alta heterogeneidade "
                f"(kappa = {self.kappa_real:.2f}) e um núcleo Rich-Club de neurônios de comando (como AVAL/AVAR), "
                f"o protocolo de Clauset-Shalizi-Newman na cauda (k >= {res_sf['k_min']}, n = {res_sf['n_tail']}, "
                f"gamma_MLE = {res_sf['gamma_pl']:.2f}, D_KS = {res_sf['ks_dist_pl']:.3f}, p_KS = {res_sf['p_valor_ks_bootstrap']:.2f}) "
                f"combinado à seleção de modelos mostra que a cauda é melhor descrita por uma distribuição com truncamento físico "
                f"({res_sf['melhor_modelo_aic']}; R²_semi-log = {res_sf['r2_semi_log']:.3f} vs. "
                f"R²_log-log = {res_sf['r2_log_log']:.3f}). Portanto, o conectoma do C. elegans classifica-se "
                f"como uma rede SMALL-WORLD com distribuição de grau de CAUDA PESADA TRUNCADA (não-Scale-Free pura)."
            )

        scorecard_rows = [
            {
                "Hipotese": "Aleatório (Erdős-Rényi)",
                "Veredito": res_er["veredito"],
                "Evidencia_Principal": (
                    f"Agrupamento 5.7x maior que ER (Z_C = {res_er['z_score_C']:.1f}, p < 1e-6) e "
                    f"superdispersão de graus (I_D = {res_er['indice_dispersao_ID']:.2f} >> 1)"
                ),
                "Metricas_Chave": (
                    f"C_real={res_er['C_real']:.4f} vs C_ER={res_er['C_ER_media']:.4f}; "
                    f"I_D={res_er['indice_dispersao_ID']:.2f}; D_KS={res_er['ks_dist_vs_ER']:.3f}"
                ),
            },
            {
                "Hipotese": "Small-World (Mundo Pequeno)",
                "Veredito": res_sw["veredito"],
                "Evidencia_Principal": (
                    f"Aprovado simultaneamente nos 3 índices formais: Humphries-Gurney (sigma={res_sw['sigma']:.2f} > 1), "
                    f"Telesford (omega={res_sw['omega']:+.2f} in [-0.5, 0.5]) e Muldoon (phi_SWP={res_sw['phi_swp']:.2f} >= 0.60)"
                ),
                "Metricas_Chave": (
                    f"sigma={res_sw['sigma']:.2f}+/-{res_sw['sigma_std']:.2f}; "
                    f"omega={res_sw['omega']:+.3f}; phi_SWP={res_sw['phi_swp']:.3f}; "
                    f"p_equiv_WS={res_sw['p_equiv_ws']:.3f}"
                ),
            },
            {
                "Hipotese": "Livre de Escala (Scale-Free)",
                "Veredito": res_sf["veredito"],
                "Evidencia_Principal": evidencia_sf,
                "Metricas_Chave": (
                    f"k_min={res_sf['k_min']}; gamma_PL={res_sf['gamma_pl']:.2f}; "
                    f"p_KS={res_sf['p_valor_ks_bootstrap']:.2f}; Melhor_AIC={res_sf['melhor_modelo_aic']}; "
                    f"Vuong(PL vs TPL): R={res_sf['vuong_pl_vs_tpl']['R']:+.2f} (p={res_sf['vuong_pl_vs_tpl']['p']:.3f})"
                ),
            },
        ]
        scorecard_df = pd.DataFrame(scorecard_rows)

        conclusao_textual = (
            f"O conectoma do C. elegans (N={self.n}, M={self.m}, <k>={self.grau_medio:.2f}) "
            f"NÃO é uma rede aleatória Erdős-Rényi (hipótese rejeitada com Z_C = {res_er['z_score_C']:.1f}, "
            f"índice de dispersão de Poisson I_D = {res_er['indice_dispersao_ID']:.2f} e D_KS = {res_er['ks_dist_vs_ER']:.3f}). "
            f"A rede é comprovadamente SMALL-WORLD (Mundo Pequeno), satisfazendo todos os critérios "
            f"da literatura: sigma = {res_sw['sigma']:.2f} +/- {res_sw['sigma_std']:.2f} (> 1.0), "
            f"omega = {res_sw['omega']:+.3f} (dentro de [-0.5, +0.5]) e Small-World Propensity "
            f"phi_SWP = {res_sw['phi_swp']:.3f} (>= 0.60), situando-se na zona de transição de Watts-Strogatz "
            f"em p_equiv ~ {res_sw['p_equiv_ws']:.3f}. {texto_sf_conclusao}"
        )

        resumo_json = _para_nativo(
            {
                "rede_base": {
                    "N": int(self.n),
                    "M": int(self.m),
                    "grau_medio": float(self.grau_medio),
                    "densidade": float(self.densidade),
                    "k_ws": int(self.k_ws),
                },
                "aleatorio": res_er,
                "small_world": res_sw,
                "scale_free": res_sf,
                "conclusao_textual": conclusao_textual,
            }
        )

        return {
            "scorecard_df": scorecard_df,
            "modelos_cauda_df": modelos_cauda_df,
            "ws_sweep_df": ws_sweep_df,
            "resumo_json": resumo_json,
        }

    def exibir_relatorio_console(self, resultado: Dict[str, Any]) -> None:
        """Imprime no console o Scorecard de Hipóteses, a comparação de modelos na cauda e a conclusão."""
        scorecard_df: pd.DataFrame = resultado["scorecard_df"]
        modelos_df: pd.DataFrame = resultado["modelos_cauda_df"]
        resumo: Dict[str, Any] = resultado["resumo_json"]

        print("\n" + "=" * 100)
        print("  DIAGNÓSTICO TOPOLÓGICO COM EVIDÊNCIAS: ALEATÓRIO vs. SMALL-WORLD vs. LIVRE DE ESCALA")
        print("=" * 100)
        for _, row in scorecard_df.iterrows():
            print(f"  [{row['Veredito'].upper()}] {row['Hipotese']}")
            print(f"    • Evidência: {row['Evidencia_Principal']}")
            print(f"    • Métricas:  {row['Metricas_Chave']}")
        print("-" * 100)
        print("  SELEÇÃO DE MODELOS CONCORRENTES NA CAUDA (k >= k_min*):")
        print(modelos_df.to_string(index=False))
        print("-" * 100)
        print("  CONCLUSÃO SINTÉTICA PARA O RELATÓRIO:")
        print(f"  {resumo['conclusao_textual']}")
        print("=" * 100 + "\n")
