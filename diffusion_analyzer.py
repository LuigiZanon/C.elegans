"""
Módulo de análise da facilidade de difusão de informação no conectoma do C. elegans.

Avalia a propagação de sinais em dois pilares integrados:
  1. Métricas estruturais de distância e eficiência: Caminho Mínimo Médio (L),
     Diâmetro (D), Eficiência Global (E_glob), distribuição de distâncias P(d),
     limiar epidêmico teórico (lambda_c) e canal biológico Sensorial -> Motor,
     comparados contra redes aleatórias equivalentes de Erdős-Rényi (ER).
  2. Simulações dinâmicas Monte Carlo com semente única:
     - Modelo SI discreto síncrono (velocidade temporal, t50, t80, AUC_SI, t_motor_50).
     - Modelo de Cascata Independente / SIR (mu = 1) (alcance final, curva de transição
       de fase vs. beta, probabilidade de cascata global e variante ponderada).
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd


VISAO_COMB_UNDIR = "Combinada (Não-Direcionada)"
VISAO_COMB_DIR = "Combinada (Direcionada)"
VISAO_QUIM_DIR = "Química (Direcionada)"

CENARIO_REAL_HUBS = "C. elegans (Hubs / Rich-Club)"
CENARIO_REAL_SENS = "C. elegans (Sensorial)"
CENARIO_REAL_ALEAT = "C. elegans (Aleatório)"
CENARIO_ER_HUBS = "Erdős-Rényi (Top Hubs)"
CENARIO_ER_ALEAT = "Erdős-Rényi (Aleatório)"

ORDEM_CENARIOS = [
    CENARIO_REAL_HUBS,
    CENARIO_REAL_SENS,
    CENARIO_REAL_ALEAT,
    CENARIO_ER_HUBS,
    CENARIO_ER_ALEAT,
]


def _sanitizar_json(obj: Any) -> Any:
    """Converte recursivamente tipos NumPy e NaN/Inf para estruturas JSON nativas."""
    if isinstance(obj, dict):
        return {str(k): _sanitizar_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitizar_json(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return obj


def _nanmean_seguro(valores: np.ndarray) -> float:
    """Calcula a média ignorando NaN sem emitir avisos quando o vetor é vazio."""
    if valores.size == 0:
        return float("nan")
    validos = valores[~np.isnan(valores)]
    if validos.size == 0:
        return float("nan")
    return float(np.mean(validos))


def _nanstd_seguro(valores: np.ndarray) -> float:
    """Calcula o desvio padrão ignorando NaN sem emitir avisos quando o vetor é vazio."""
    if valores.size == 0:
        return float("nan")
    validos = valores[~np.isnan(valores)]
    if validos.size == 0:
        return float("nan")
    return float(np.std(validos))


class InformationDiffusionAnalyzer:
    """Analisa a facilidade de difusão de informação estrutural e dinâmica no conectoma."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        membros_rich_club: Optional[Set[str]] = None,
        n_runs_er_estrutural: int = 20,
        n_runs_si: int = 150,
        n_runs_ic: int = 150,
        beta_si_padrao: float = 0.08,
        max_steps_si: int = 25,
        betas_ic: Optional[List[float]] = None,
        beta_ic_referencia: float = 0.08,
        seed_base: int = 42,
    ) -> None:
        self.graph = graph
        self.n_runs_er_estrutural = int(n_runs_er_estrutural)
        self.n_runs_si = int(n_runs_si)
        self.n_runs_ic = int(n_runs_ic)
        self.beta_si_padrao = float(beta_si_padrao)
        self.max_steps_si = int(max_steps_si)
        self.betas_ic = (
            [float(b) for b in betas_ic]
            if betas_ic is not None
            else [0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35]
        )
        self.beta_ic_referencia = float(beta_ic_referencia)
        self.seed_base = int(seed_base)

        # Projeções livres de auto-laços preservando atributos dos neurônios
        self.g_comb_undir = self._construir_projecao_combinada_undir()
        self.g_comb_dir = self._construir_projecao_direcionada(camada="Combinada")
        self.g_quim_dir = self._construir_projecao_direcionada(camada="Quimica")

        # Grupos de sementes biológicas no conectoma real
        self.nos_sensoriais: List[str] = sorted(
            n for n, d in self.g_comb_undir.nodes(data=True)
            if d.get("tipo_funcional") == "Sensorial"
        )
        self.nos_motores: List[str] = sorted(
            n for n, d in self.g_comb_undir.nodes(data=True)
            if d.get("tipo_funcional") == "Motor"
        )

        hubs_validos = (
            sorted(n for n in membros_rich_club if n in self.g_comb_undir)
            if membros_rich_club
            else []
        )
        if hubs_validos:
            self.nos_hubs: List[str] = hubs_validos
        else:
            top_grau = sorted(
                self.g_comb_undir.degree(),
                key=lambda item: (-item[1], item[0]),
            )[:6]
            self.nos_hubs = sorted(n for n, _ in top_grau)

        # Cache de pools de redes Erdős-Rényi equivalentes por visão
        self._pools_er: Dict[str, List[Union[nx.Graph, nx.DiGraph]]] = {}

    def _construir_projecao_combinada_undir(self) -> nx.Graph:
        """Projeta a camada Combinada (Química + Elétrica) em grafo não-direcionado sem auto-laços."""
        g_undir = nx.Graph()
        for no, dados in self.graph.nodes(data=True):
            g_undir.add_node(no, **dados)

        pesos_eletricos: Dict[Tuple[str, str], float] = {}
        pesos_quimicos: Dict[Tuple[str, str], float] = {}

        for u, v, chave, dados in self.graph.edges(keys=True, data=True):
            if u == v:
                continue
            tipo = dados.get("synapse_type", chave)
            peso = float(dados.get("weight", 1.0))
            if peso <= 0.0:
                continue
            par = (u, v) if u <= v else (v, u)
            if tipo == "Eletrica":
                pesos_eletricos[par] = max(pesos_eletricos.get(par, 0.0), peso)
            elif tipo == "Quimica":
                pesos_quimicos[par] = pesos_quimicos.get(par, 0.0) + peso

        todos_pares = set(pesos_eletricos.keys()) | set(pesos_quimicos.keys())
        for u, v in todos_pares:
            peso_total = pesos_eletricos.get((u, v), 0.0) + pesos_quimicos.get((u, v), 0.0)
            if peso_total > 0.0:
                g_undir.add_edge(u, v, weight=peso_total, distance=1.0 / peso_total)

        return g_undir

    def _construir_projecao_direcionada(self, camada: str) -> nx.DiGraph:
        """
        Projeta a camada especificada em DiGraph livre de auto-laços.
        Na camada Combinada, junções elétricas são inseridas nos dois sentidos (u <-> v).
        """
        g_dir = nx.DiGraph()
        for no, dados in self.graph.nodes(data=True):
            g_dir.add_node(no, **dados)

        pesos_eletricos: Dict[Tuple[str, str], float] = {}

        for u, v, chave, dados in self.graph.edges(keys=True, data=True):
            if u == v:
                continue
            tipo = dados.get("synapse_type", chave)
            peso = float(dados.get("weight", 1.0))
            if peso <= 0.0:
                continue

            if tipo == "Eletrica" and camada == "Combinada":
                par = (u, v) if u <= v else (v, u)
                pesos_eletricos[par] = max(pesos_eletricos.get(par, 0.0), peso)
            elif tipo == "Quimica" and camada in {"Quimica", "Combinada"}:
                peso_atual = g_dir[u][v]["weight"] if g_dir.has_edge(u, v) else 0.0
                g_dir.add_edge(u, v, weight=peso_atual + peso)

        for (u, v), peso in pesos_eletricos.items():
            peso_uv = g_dir[u][v]["weight"] if g_dir.has_edge(u, v) else 0.0
            g_dir.add_edge(u, v, weight=peso_uv + peso)
            peso_vu = g_dir[v][u]["weight"] if g_dir.has_edge(v, u) else 0.0
            g_dir.add_edge(v, u, weight=peso_vu + peso)

        for _, _, dados in g_dir.edges(data=True):
            dados["distance"] = 1.0 / dados["weight"]

        return g_dir

    def _obter_pool_er(
        self, visao: str, n: int, m: int, directed: bool, offset_seed: int
    ) -> List[Union[nx.Graph, nx.DiGraph]]:
        """Pré-gera e reutiliza um pool determinístico de redes Erdős-Rényi equivalentes."""
        if visao not in self._pools_er:
            pool: List[Union[nx.Graph, nx.DiGraph]] = []
            for r in range(self.n_runs_er_estrutural):
                seed = self.seed_base + offset_seed + r
                g_er = nx.gnm_random_graph(n, m, seed=seed, directed=directed)
                pool.append(g_er)
            self._pools_er[visao] = pool
        return self._pools_er[visao]

    # ------------------------------------------------------------------
    # PILAR 1: MÉTRICAS ESTRUTURAIS DE DIFUSÃO
    # ------------------------------------------------------------------

    @staticmethod
    def _calcular_metricas_estruturais_grafo(
        g: Union[nx.Graph, nx.DiGraph]
    ) -> Dict[str, Any]:
        """
        Calcula L, D (na maior componente), E_glob, distribuição de distâncias P(d)
        e limiar epidêmico teórico lambda_c com suporte seguro a Graph e DiGraph.
        """
        n = g.number_of_nodes()
        m = g.number_of_edges()
        if n == 0:
            return {
                "N": 0,
                "M": 0,
                "Fracao_Componente": 0.0,
                "L": 0.0,
                "D": 0.0,
                "E_Glob": 0.0,
                "Frac_Pares_Alcancaveis": 0.0,
                "Lambda_c": float("nan"),
                "Dist_Fracoes": {},
                "Pct_Ate_3_Saltos": 0.0,
            }

        if g.is_directed():
            comp_nos = max(nx.strongly_connected_components(g), key=len)
        else:
            comp_nos = max(nx.connected_components(g), key=len)

        frac_comp = len(comp_nos) / float(n)
        if len(comp_nos) <= 1:
            l_val = 0.0
            d_val = 0.0
        else:
            comp = g.subgraph(comp_nos)
            l_val = float(nx.average_shortest_path_length(comp))
            d_val = float(nx.diameter(comp))

        # Eficiência Global unificada via all_pairs_shortest_path_length (aceita DiGraph e desconexos)
        total_pares = n * (n - 1)
        soma_inv = 0.0
        pares_alcancaveis = 0
        contagem_dist: Dict[int, int] = {}

        for u, dists in nx.all_pairs_shortest_path_length(g):
            for v, d in dists.items():
                if u == v or d <= 0:
                    continue
                soma_inv += 1.0 / d
                pares_alcancaveis += 1
                contagem_dist[d] = contagem_dist.get(d, 0) + 1

        e_glob = soma_inv / float(total_pares) if total_pares > 0 else 0.0
        frac_alcancaveis = pares_alcancaveis / float(total_pares) if total_pares > 0 else 0.0

        dist_fracoes: Dict[int, float] = {}
        if total_pares > 0:
            for d, cnt in contagem_dist.items():
                dist_fracoes[d] = cnt / float(total_pares)

        pct_ate_3 = sum(dist_fracoes.get(d, 0.0) for d in (1, 2, 3)) * 100.0

        # Limiar epidêmico teórico (Pastor-Satorras & Vespignani) para projeções não-direcionadas
        if not g.is_directed():
            graus = np.array([deg for _, deg in g.degree()], dtype=float)
            k_med = float(np.mean(graus))
            k2_med = float(np.mean(graus ** 2))
            lambda_c = (k_med / k2_med) if k2_med > 0.0 else float("nan")
        else:
            lambda_c = float("nan")

        return {
            "N": n,
            "M": m,
            "Fracao_Componente": frac_comp,
            "L": l_val,
            "D": d_val,
            "E_Glob": e_glob,
            "Frac_Pares_Alcancaveis": frac_alcancaveis,
            "Lambda_c": lambda_c,
            "Dist_Fracoes": dist_fracoes,
            "Pct_Ate_3_Saltos": pct_ate_3,
        }

    def _calcular_canal_biologico(self, g_dir: nx.DiGraph) -> Dict[str, float]:
        """Calcula caminhos mínimos (topológicos e ponderados) de Sensorial -> Motor e Hubs -> Motor."""
        motores_set = set(self.nos_motores)

        def _medir_origem_destino(origens: List[str]) -> Tuple[float, float, float]:
            dists_topo: List[float] = []
            dists_pond: List[float] = []
            total_pares = 0

            for s in origens:
                if s not in g_dir:
                    continue
                sp_topo = nx.single_source_shortest_path_length(g_dir, s)
                sp_pond = nx.single_source_dijkstra_path_length(g_dir, s, weight="distance")
                for m in motores_set:
                    if m == s or m not in g_dir:
                        continue
                    total_pares += 1
                    if m in sp_topo:
                        dists_topo.append(float(sp_topo[m]))
                    if m in sp_pond:
                        dists_pond.append(float(sp_pond[m]))

            alcance_pct = (len(dists_topo) / float(total_pares) * 100.0) if total_pares > 0 else 0.0
            l_topo = float(np.mean(dists_topo)) if dists_topo else float("nan")
            l_pond = float(np.mean(dists_pond)) if dists_pond else float("nan")
            return l_topo, alcance_pct, l_pond

        l_sm_topo, alc_sm_pct, l_sm_pond = _medir_origem_destino(self.nos_sensoriais)
        l_hm_topo, alc_hm_pct, l_hm_pond = _medir_origem_destino(self.nos_hubs)

        return {
            "L_sensorial_para_motor": l_sm_topo,
            "alcance_sensorial_para_motor_pct": alc_sm_pct,
            "L_ponderado_sensorial_para_motor": l_sm_pond,
            "L_hubs_para_motor": l_hm_topo,
            "alcance_hubs_para_motor_pct": alc_hm_pct,
            "L_ponderado_hubs_para_motor": l_hm_pond,
        }

    def _executar_analise_estrutural(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Executa o Pilar 1 comparando as 3 visões do conectoma contra redes ER equivalentes."""
        configuracoes_visao = [
            (VISAO_COMB_UNDIR, self.g_comb_undir, False, 1000),
            (VISAO_COMB_DIR, self.g_comb_dir, True, 2000),
            (VISAO_QUIM_DIR, self.g_quim_dir, True, 3000),
        ]

        linhas_estrutural: List[Dict[str, Any]] = []
        linhas_distancias: List[Dict[str, Any]] = []
        detalhes_visoes: Dict[str, Dict[str, Any]] = {}

        for visao, g_real, directed, offset_seed in configuracoes_visao:
            met_real = self._calcular_metricas_estruturais_grafo(g_real)
            pool_er = self._obter_pool_er(
                visao, met_real["N"], met_real["M"], directed, offset_seed
            )

            er_l, er_d, er_e, er_lc, er_ate3 = [], [], [], [], []
            er_dist_acum: Dict[str, List[float]] = {k: [] for k in ("1", "2", "3", "4", "5+")}

            for g_er in pool_er:
                met_er = self._calcular_metricas_estruturais_grafo(g_er)
                er_l.append(met_er["L"])
                er_d.append(met_er["D"])
                er_e.append(met_er["E_Glob"])
                er_lc.append(met_er["Lambda_c"])
                er_ate3.append(met_er["Pct_Ate_3_Saltos"])

                if visao in {VISAO_COMB_UNDIR, VISAO_COMB_DIR}:
                    df_er = met_er["Dist_Fracoes"]
                    for d_int in (1, 2, 3, 4):
                        er_dist_acum[str(d_int)].append(df_er.get(d_int, 0.0) * 100.0)
                    soma_5_mais = sum(v for k, v in df_er.items() if k >= 5) * 100.0
                    er_dist_acum["5+"].append(soma_5_mais)

            arr_l = np.asarray(er_l, dtype=float)
            arr_d = np.asarray(er_d, dtype=float)
            arr_e = np.asarray(er_e, dtype=float)
            arr_lc = np.asarray(er_lc, dtype=float)
            arr_ate3 = np.asarray(er_ate3, dtype=float)

            l_er_med, l_er_std = float(np.mean(arr_l)), float(np.std(arr_l))
            d_er_med, d_er_std = float(np.mean(arr_d)), float(np.std(arr_d))
            e_er_med, e_er_std = float(np.mean(arr_e)), float(np.std(arr_e))
            lc_er_med = _nanmean_seguro(arr_lc)
            ate3_er_med = float(np.mean(arr_ate3))

            razao_l = met_real["L"] / l_er_med if l_er_med > 0.0 else float("nan")
            razao_e = met_real["E_Glob"] / e_er_med if e_er_med > 0.0 else float("nan")
            z_e = (
                (met_real["E_Glob"] - e_er_med) / e_er_std
                if e_er_std > 1e-12
                else 0.0
            )

            linhas_estrutural.append({
                "Visao": visao,
                "N": met_real["N"],
                "M": met_real["M"],
                "Fracao_Componente_Real": met_real["Fracao_Componente"],
                "L_Real": met_real["L"],
                "L_ER_Media": l_er_med,
                "L_ER_Std": l_er_std,
                "Razao_L_ER": razao_l,
                "D_Real": met_real["D"],
                "D_ER_Media": d_er_med,
                "D_ER_Std": d_er_std,
                "E_Glob_Real": met_real["E_Glob"],
                "E_Glob_ER_Media": e_er_med,
                "E_Glob_ER_Std": e_er_std,
                "Razao_E_ER": razao_e,
                "Z_Score_E_Glob": z_e,
                "Lambda_c_Real": met_real["Lambda_c"],
                "Lambda_c_ER_Media": lc_er_med,
            })

            detalhes_visoes[visao] = {
                "L_real": met_real["L"],
                "L_er_media": l_er_med,
                "D_real": met_real["D"],
                "D_er_media": d_er_med,
                "E_glob_real": met_real["E_Glob"],
                "E_glob_er_media": e_er_med,
                "lambda_c_real": met_real["Lambda_c"],
                "lambda_c_er_media": lc_er_med,
                "pct_pares_ate_3_saltos_real": met_real["Pct_Ate_3_Saltos"],
                "pct_pares_ate_3_saltos_er": ate3_er_med,
            }

            if visao in {VISAO_COMB_UNDIR, VISAO_COMB_DIR}:
                df_real = met_real["Dist_Fracoes"]
                for rotulo in ("1", "2", "3", "4", "5+"):
                    if rotulo == "5+":
                        pct_real = sum(v for k, v in df_real.items() if k >= 5) * 100.0
                    else:
                        pct_real = df_real.get(int(rotulo), 0.0) * 100.0
                    vals_er = np.asarray(er_dist_acum[rotulo], dtype=float)
                    linhas_distancias.append({
                        "Visao": visao,
                        "Distancia": rotulo,
                        "Pct_Real": pct_real,
                        "Pct_ER_Media": float(np.mean(vals_er)),
                        "Pct_ER_Std": float(np.std(vals_er)),
                    })

        canal_comb_dir = self._calcular_canal_biologico(self.g_comb_dir)
        canal_quim_dir = self._calcular_canal_biologico(self.g_quim_dir)
        detalhes_visoes[VISAO_COMB_DIR].update(canal_comb_dir)
        detalhes_visoes[VISAO_QUIM_DIR].update(canal_quim_dir)

        return (
            pd.DataFrame(linhas_estrutural),
            pd.DataFrame(linhas_distancias),
            detalhes_visoes,
        )

    # ------------------------------------------------------------------
    # PILAR 2: MOTOR DE SIMULAÇÃO DE PROPAGAÇÃO (SI e IC / SIR)
    # ------------------------------------------------------------------

    @staticmethod
    def _preparar_adjacencia_indexada(
        g: Union[nx.Graph, nx.DiGraph],
    ) -> Tuple[List[Any], Dict[Any, int], List[List[int]], List[List[float]], List[int]]:
        """Converte o grafo para listas de adjacência compactas indexadas por inteiros 0..N-1."""
        nos = list(g.nodes())
        no_para_idx = {no: i for i, no in enumerate(nos)}
        n = len(nos)
        adj: List[List[int]] = [[] for _ in range(n)]
        adj_weights: List[List[float]] = [[] for _ in range(n)]

        for u_idx, u in enumerate(nos):
            vizinhos = g.successors(u) if g.is_directed() else g.neighbors(u)
            lista_v: List[int] = []
            lista_w: List[float] = []
            for v in vizinhos:
                v_idx = no_para_idx[v]
                if v_idx == u_idx:
                    continue
                lista_v.append(v_idx)
                dados = g[u][v]
                lista_w.append(max(float(dados.get("weight", 1.0)), 0.0))
            adj[u_idx] = lista_v
            adj_weights[u_idx] = lista_w

        # Top 6 hubs pelo grau de saída (direcionado) ou grau total (não-direcionado)
        graus_idx = sorted(
            range(n),
            key=lambda i: (-len(adj[i]), i),
        )[:6]
        return nos, no_para_idx, adj, adj_weights, graus_idx

    def _simular_si_rodada(
        self,
        n: int,
        adj: List[List[int]],
        seed_idx: int,
        beta: float,
        max_steps: int,
        rng: np.random.Generator,
        motor_mask: Optional[np.ndarray],
        n_motores: int,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        """Executa uma rodada síncrona exata do modelo SI com semente única."""
        ativos = np.zeros(n, dtype=bool)
        ativos[seed_idx] = True
        ativos_lista = [seed_idx]
        n_ativos = 1

        curva_total = np.zeros(max_steps + 1, dtype=float)
        curva_motores = np.full(max_steps + 1, np.nan, dtype=float)

        curva_total[0] = n_ativos / float(n)
        if motor_mask is not None and n_motores > 0:
            n_mot_ativos = 1 if motor_mask[seed_idx] else 0
            curva_motores[0] = n_mot_ativos / float(n_motores)
        else:
            n_mot_ativos = 0

        t50 = 0.0 if curva_total[0] >= 0.50 else float("nan")
        t80 = 0.0 if curva_total[0] >= 0.80 else float("nan")
        t_motor_50 = (
            0.0
            if (motor_mask is not None and n_motores > 0 and curva_motores[0] >= 0.50)
            else float("nan")
        )

        um_menos_beta = 1.0 - beta

        for passo in range(1, max_steps + 1):
            if n_ativos < n:
                # Contagem síncrona exata de vizinhos ativos por nó suscetível
                contagem: Dict[int, int] = {}
                for u in ativos_lista:
                    for v in adj[u]:
                        if not ativos[v]:
                            contagem[v] = contagem.get(v, 0) + 1

                if contagem:
                    cand_nos = list(contagem.keys())
                    m_vals = np.fromiter(contagem.values(), dtype=np.float64, count=len(cand_nos))
                    if beta >= 1.0:
                        probs = np.ones_like(m_vals)
                    else:
                        probs = 1.0 - np.power(um_menos_beta, m_vals)
                    sorteios = rng.random(len(cand_nos))
                    for idx_c, v in enumerate(cand_nos):
                        if sorteios[idx_c] < probs[idx_c]:
                            ativos[v] = True
                            ativos_lista.append(v)
                            n_ativos += 1
                            if motor_mask is not None and motor_mask[v]:
                                n_mot_ativos += 1

            frac_total = n_ativos / float(n)
            curva_total[passo] = frac_total

            if math.isnan(t50) and frac_total >= 0.50:
                t50 = float(passo)
            if math.isnan(t80) and frac_total >= 0.80:
                t80 = float(passo)

            if motor_mask is not None and n_motores > 0:
                frac_mot = n_mot_ativos / float(n_motores)
                curva_motores[passo] = frac_mot
                if math.isnan(t_motor_50) and frac_mot >= 0.50:
                    t_motor_50 = float(passo)

        auc_si = float(np.mean(curva_total[1:])) if max_steps > 0 else curva_total[0]
        return curva_total, curva_motores, t50, t80, auc_si, t_motor_50

    @staticmethod
    def _simular_ic_rodada(
        n: int,
        adj: List[List[int]],
        adj_weights: Optional[List[List[float]]],
        seed_idx: int,
        beta: float,
        rng: np.random.Generator,
        motor_mask: Optional[np.ndarray],
        n_motores: int,
        ponderado: bool = False,
    ) -> Tuple[float, int, float, float]:
        """Executa uma rodada síncrona exata da Cascata Independente (IC / SIR com mu=1)."""
        ativos = np.zeros(n, dtype=bool)
        ativos[seed_idx] = True
        fronteira = [seed_idx]
        n_ativos = 1
        n_mot_ativos = 1 if (motor_mask is not None and motor_mask[seed_idx]) else 0
        passos = 0

        beta_clamped = min(max(float(beta), 0.0), 1.0)
        um_menos_beta = 1.0 - beta_clamped

        while fronteira:
            proxima_fronteira: List[int] = []

            if not ponderado:
                contagem: Dict[int, int] = {}
                for u in fronteira:
                    for v in adj[u]:
                        if not ativos[v]:
                            contagem[v] = contagem.get(v, 0) + 1

                if contagem:
                    cand_nos = list(contagem.keys())
                    m_vals = np.fromiter(contagem.values(), dtype=np.float64, count=len(cand_nos))
                    if beta_clamped >= 1.0:
                        probs = np.ones_like(m_vals)
                    else:
                        probs = 1.0 - np.power(um_menos_beta, m_vals)
                    sorteios = rng.random(len(cand_nos))
                    for idx_c, v in enumerate(cand_nos):
                        if sorteios[idx_c] < probs[idx_c]:
                            ativos[v] = True
                            proxima_fronteira.append(v)
                            n_ativos += 1
                            if motor_mask is not None and motor_mask[v]:
                                n_mot_ativos += 1
            else:
                assert adj_weights is not None
                sobrev: Dict[int, float] = {}
                for u in fronteira:
                    viz_u = adj[u]
                    pesos_u = adj_weights[u]
                    for v, w in zip(viz_u, pesos_u):
                        if not ativos[v]:
                            w_uv = max(float(w), 0.0)
                            if beta_clamped >= 1.0:
                                p_sobrev_aresta = 0.0 if w_uv > 0.0 else 1.0
                            else:
                                p_sobrev_aresta = math.pow(um_menos_beta, w_uv)
                            sobrev[v] = sobrev.get(v, 1.0) * p_sobrev_aresta

                if sobrev:
                    cand_nos = list(sobrev.keys())
                    probs = np.array([1.0 - sobrev[v] for v in cand_nos], dtype=np.float64)
                    sorteios = rng.random(len(cand_nos))
                    for idx_c, v in enumerate(cand_nos):
                        if sorteios[idx_c] < probs[idx_c]:
                            ativos[v] = True
                            proxima_fronteira.append(v)
                            n_ativos += 1
                            if motor_mask is not None and motor_mask[v]:
                                n_mot_ativos += 1

            if proxima_fronteira:
                passos += 1
            fronteira = proxima_fronteira

        alcance_pct = (n_ativos / float(n)) * 100.0
        cascata_global = 100.0 if alcance_pct >= 20.0 else 0.0
        alcance_mot_pct = (
            (n_mot_ativos / float(n_motores)) * 100.0
            if (motor_mask is not None and n_motores > 0)
            else float("nan")
        )
        return alcance_pct, passos, cascata_global, alcance_mot_pct

    def _executar_simulacoes_visao(
        self,
        visao: str,
        g_real: Union[nx.Graph, nx.DiGraph],
        directed: bool,
        offset_seed: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Executa baterias SI e IC (binária) para os 5 cenários na visão informada."""
        nos_real, mapa_real, adj_real, _, _ = self._preparar_adjacencia_indexada(g_real)
        n_real = len(nos_real)

        motor_mask_real = np.array(
            [no in set(self.nos_motores) for no in nos_real], dtype=bool
        )
        n_motores_real = int(motor_mask_real.sum())

        hubs_idx_real = [mapa_real[h] for h in self.nos_hubs if h in mapa_real]
        sens_idx_real = [mapa_real[s] for s in self.nos_sensoriais if s in mapa_real]

        # Reutiliza pool ER pré-gerado
        pool_er = self._obter_pool_er(
            visao, g_real.number_of_nodes(), g_real.number_of_edges(), directed, offset_seed
        )
        er_estruturas = [self._preparar_adjacencia_indexada(g_er) for g_er in pool_er]
        n_pool = len(er_estruturas)

        linhas_si_curvas: List[Dict[str, Any]] = []
        linhas_si_resumo: List[Dict[str, Any]] = []
        linhas_ic_curvas: List[Dict[str, Any]] = []

        # --- 1. Modelo SI ---
        for c_idx, cenario in enumerate(ORDEM_CENARIOS):
            rng = np.random.default_rng(self.seed_base + offset_seed + 500 + c_idx)
            matriz_curvas = np.zeros((self.n_runs_si, self.max_steps_si + 1), dtype=float)
            matriz_motores = np.full((self.n_runs_si, self.max_steps_si + 1), np.nan, dtype=float)
            t50_arr = np.full(self.n_runs_si, np.nan, dtype=float)
            t80_arr = np.full(self.n_runs_si, np.nan, dtype=float)
            auc_arr = np.zeros(self.n_runs_si, dtype=float)
            tmot50_arr = np.full(self.n_runs_si, np.nan, dtype=float)

            eh_real = cenario.startswith("C. elegans")

            for r in range(self.n_runs_si):
                if eh_real:
                    adj_cur = adj_real
                    n_cur = n_real
                    mask_cur = motor_mask_real
                    n_mot_cur = n_motores_real
                    if cenario == CENARIO_REAL_HUBS:
                        seed_idx = hubs_idx_real[r % len(hubs_idx_real)]
                    elif cenario == CENARIO_REAL_SENS:
                        seed_idx = int(sens_idx_real[int(rng.integers(0, len(sens_idx_real)))])
                    else:
                        seed_idx = int(rng.integers(0, n_real))
                else:
                    nos_er, _, adj_er, _, hubs_er = er_estruturas[r % n_pool]
                    adj_cur = adj_er
                    n_cur = len(nos_er)
                    mask_cur = None
                    n_mot_cur = 0
                    if cenario == CENARIO_ER_HUBS:
                        seed_idx = hubs_er[r % len(hubs_er)]
                    else:
                        seed_idx = int(rng.integers(0, n_cur))

                c_tot, c_mot, t50, t80, auc_si, tmot50 = self._simular_si_rodada(
                    n_cur,
                    adj_cur,
                    seed_idx,
                    self.beta_si_padrao,
                    self.max_steps_si,
                    rng,
                    mask_cur,
                    n_mot_cur,
                )
                matriz_curvas[r, :] = c_tot * 100.0
                matriz_motores[r, :] = c_mot * 100.0
                t50_arr[r] = t50
                t80_arr[r] = t80
                auc_arr[r] = auc_si
                tmot50_arr[r] = tmot50

            for t in range(self.max_steps_si + 1):
                linhas_si_curvas.append({
                    "Passo_t": t,
                    "Cenario": cenario,
                    "Visao": visao,
                    "Infectados_Pct_Media": float(np.mean(matriz_curvas[:, t])),
                    "Infectados_Pct_Std": float(np.std(matriz_curvas[:, t])),
                    "Motores_Pct_Media": _nanmean_seguro(matriz_motores[:, t]),
                })

            frac_atingiu_50 = float(np.mean(~np.isnan(t50_arr)) * 100.0)
            linhas_si_resumo.append({
                "Visao": visao,
                "Cenario": cenario,
                "Beta_SI": self.beta_si_padrao,
                "t50_Media": _nanmean_seguro(t50_arr),
                "t50_Std": _nanstd_seguro(t50_arr),
                "t80_Media": _nanmean_seguro(t80_arr),
                "Frac_Atingiu_50_Pct": frac_atingiu_50,
                "AUC_SI_Media": float(np.mean(auc_arr)),
                "AUC_SI_Std": float(np.std(auc_arr)),
                "Infectados_Final_Pct": float(np.mean(matriz_curvas[:, -1])),
                "t_motor_50_Media": _nanmean_seguro(tmot50_arr),
            })

        # --- 2. Modelo de Cascata Independente (IC / SIR) ---
        for c_idx, cenario in enumerate(ORDEM_CENARIOS):
            eh_real = cenario.startswith("C. elegans")
            for b_idx, beta in enumerate(self.betas_ic):
                rng = np.random.default_rng(
                    self.seed_base + offset_seed + 700 + c_idx * 50 + b_idx
                )
                alc_arr = np.zeros(self.n_runs_ic, dtype=float)
                dur_arr = np.zeros(self.n_runs_ic, dtype=float)
                glob_arr = np.zeros(self.n_runs_ic, dtype=float)
                mot_arr = np.full(self.n_runs_ic, np.nan, dtype=float)

                for r in range(self.n_runs_ic):
                    if eh_real:
                        adj_cur = adj_real
                        n_cur = n_real
                        mask_cur = motor_mask_real
                        n_mot_cur = n_motores_real
                        if cenario == CENARIO_REAL_HUBS:
                            seed_idx = hubs_idx_real[r % len(hubs_idx_real)]
                        elif cenario == CENARIO_REAL_SENS:
                            seed_idx = int(sens_idx_real[int(rng.integers(0, len(sens_idx_real)))])
                        else:
                            seed_idx = int(rng.integers(0, n_real))
                    else:
                        nos_er, _, adj_er, _, hubs_er = er_estruturas[r % n_pool]
                        adj_cur = adj_er
                        n_cur = len(nos_er)
                        mask_cur = None
                        n_mot_cur = 0
                        if cenario == CENARIO_ER_HUBS:
                            seed_idx = hubs_er[r % len(hubs_er)]
                        else:
                            seed_idx = int(rng.integers(0, n_cur))

                    alc, dur, c_glob, alc_mot = self._simular_ic_rodada(
                        n_cur,
                        adj_cur,
                        None,
                        seed_idx,
                        beta,
                        rng,
                        mask_cur,
                        n_mot_cur,
                        ponderado=False,
                    )
                    alc_arr[r] = alc
                    dur_arr[r] = float(dur)
                    glob_arr[r] = c_glob
                    mot_arr[r] = alc_mot

                linhas_ic_curvas.append({
                    "Visao": visao,
                    "Cenario": cenario,
                    "Beta": beta,
                    "Alcance_Final_Pct_Media": float(np.mean(alc_arr)),
                    "Alcance_Final_Pct_Std": float(np.std(alc_arr)),
                    "Duracao_Media": float(np.mean(dur_arr)),
                    "Prob_Cascata_Global_Pct": float(np.mean(glob_arr)),
                    "Alcance_Motores_Pct_Media": _nanmean_seguro(mot_arr),
                })

        return linhas_si_curvas, linhas_si_resumo, linhas_ic_curvas

    def _executar_ic_ponderada_real(self) -> pd.DataFrame:
        """Executa a variante ponderada da Cascata Independente em Combinada (Direcionada)."""
        nos_real, mapa_real, adj_real, adj_w_real, _ = self._preparar_adjacencia_indexada(
            self.g_comb_dir
        )
        n_real = len(nos_real)
        motor_mask_real = np.array(
            [no in set(self.nos_motores) for no in nos_real], dtype=bool
        )
        n_motores_real = int(motor_mask_real.sum())

        hubs_idx_real = [mapa_real[h] for h in self.nos_hubs if h in mapa_real]
        sens_idx_real = [mapa_real[s] for s in self.nos_sensoriais if s in mapa_real]

        linhas: List[Dict[str, Any]] = []
        cenarios_pond = [
            (CENARIO_REAL_HUBS, hubs_idx_real, True),
            (CENARIO_REAL_SENS, sens_idx_real, False),
        ]

        for c_idx, (cenario, pool_sementes, rotativo) in enumerate(cenarios_pond):
            for b_idx, beta in enumerate(self.betas_ic):
                rng = np.random.default_rng(self.seed_base + 9000 + c_idx * 50 + b_idx)
                alc_arr = np.zeros(self.n_runs_ic, dtype=float)
                dur_arr = np.zeros(self.n_runs_ic, dtype=float)
                glob_arr = np.zeros(self.n_runs_ic, dtype=float)
                mot_arr = np.full(self.n_runs_ic, np.nan, dtype=float)

                for r in range(self.n_runs_ic):
                    if rotativo:
                        seed_idx = pool_sementes[r % len(pool_sementes)]
                    else:
                        seed_idx = int(pool_sementes[int(rng.integers(0, len(pool_sementes)))])

                    alc, dur, c_glob, alc_mot = self._simular_ic_rodada(
                        n_real,
                        adj_real,
                        adj_w_real,
                        seed_idx,
                        beta,
                        rng,
                        motor_mask_real,
                        n_motores_real,
                        ponderado=True,
                    )
                    alc_arr[r] = alc
                    dur_arr[r] = float(dur)
                    glob_arr[r] = c_glob
                    mot_arr[r] = alc_mot

                linhas.append({
                    "Visao": VISAO_COMB_DIR,
                    "Cenario": cenario,
                    "Beta": beta,
                    "Alcance_Final_Pct_Media": float(np.mean(alc_arr)),
                    "Alcance_Final_Pct_Std": float(np.std(alc_arr)),
                    "Duracao_Media": float(np.mean(dur_arr)),
                    "Prob_Cascata_Global_Pct": float(np.mean(glob_arr)),
                    "Alcance_Motores_Pct_Media": _nanmean_seguro(mot_arr),
                })

        return pd.DataFrame(linhas)

    # ------------------------------------------------------------------
    # CONSOLIDAÇÃO, DIAGNÓSTICO E RELATÓRIO
    # ------------------------------------------------------------------

    @staticmethod
    def _buscar_valor(
        df: pd.DataFrame, filtros: Dict[str, Any], coluna: str
    ) -> float:
        """Extrai com segurança um escalar float de um DataFrame filtrado."""
        mask = pd.Series(True, index=df.index)
        for col, val in filtros.items():
            if isinstance(val, float):
                mask &= np.isclose(df[col].to_numpy(dtype=float), val, atol=1e-9)
            else:
                mask &= df[col] == val
        sub = df.loc[mask, coluna]
        if sub.empty:
            return float("nan")
        return float(sub.iloc[0])

    def _montar_conclusao_textual(
        self,
        est_u: Dict[str, Any],
        est_d: Dict[str, Any],
        si_u: Dict[str, Any],
        ic_u: Dict[str, Any],
        ic_pond: Dict[str, Any],
    ) -> str:
        """Sintetiza em linguagem natural a resposta sobre a facilidade de difusão na rede."""
        l_real = est_u["L_real"]
        l_er = est_u["L_er_media"]
        d_real = est_u["D_real"]
        e_real = est_u["E_glob_real"]
        e_er = est_u["E_glob_er_media"]
        lc_real = est_u["lambda_c_real"]
        lc_er = est_u["lambda_c_er_media"]
        pct3 = est_u["pct_pares_ate_3_saltos_real"]
        l_sm = est_d["L_sensorial_para_motor"]
        alc_sm = est_d["alcance_sensorial_para_motor_pct"]

        t50_hubs = si_u["hubs_t50"]
        t50_sens = si_u["sensorial_t50"]
        t50_er = si_u["er_aleatorio_t50"]
        alc_hubs = ic_u["hubs_alcance_pct"]
        alc_sens = ic_u["sensorial_alcance_pct"]
        alc_er = ic_u["er_aleatorio_alcance_pct"]
        alc_pond_hubs = ic_pond["hubs_alcance_pct"]

        str_t50_hubs = f"{t50_hubs:.1f}" if t50_hubs is not None and not math.isnan(t50_hubs) else "N/A"
        str_t50_sens = f"{t50_sens:.1f}" if t50_sens is not None and not math.isnan(t50_sens) else "N/A"
        str_t50_er = f"{t50_er:.1f}" if t50_er is not None and not math.isnan(t50_er) else "N/A"

        return (
            f"Sim — a difusão de informação no conectoma do C. elegans é estruturalmente fácil e "
            f"fortemente acelerada pelos hubs do Rich-Club. Na projeção combinada não-direcionada, "
            f"o caminho mínimo médio é curto (L = {l_real:.2f} vs. {l_er:.2f} na rede Erdős-Rényi, "
            f"diâmetro D = {d_real:.0f}), com eficiência global E_glob = {e_real:.3f} (ER: {e_er:.3f}) "
            f"e {pct3:.1f}% de todos os pares de neurônios separados por no máximo 3 saltos. "
            f"Pela heterogeneidade de grau, o limiar epidêmico teórico cai para lambda_c = {lc_real:.4f} "
            f"(contra {lc_er:.4f} em ER), facilitando a sustentação de sinais mesmo com baixa transmissibilidade. "
            f"No canal biológico direcionado, sinais partindo de neurônios Sensoriais alcançam {alc_sm:.1f}% "
            f"dos Motores em média em {l_sm:.2f} saltos. Nas simulações dinâmicas (beta = {self.beta_ic_referencia:.2f}), "
            f"a propagação SI a partir de Hubs atinge 50% da rede em t50 = {str_t50_hubs} passos "
            f"(Sensorial: {str_t50_sens}; ER aleatório: {str_t50_er}), enquanto na Cascata Independente "
            f"os Hubs alcançam {alc_hubs:.1f}% da rede (Sensorial: {alc_sens:.1f}%; ER aleatório: {alc_er:.1f}%), "
            f"subindo para {alc_pond_hubs:.1f}% ao incorporar a multiplicidade sináptica ponderada."
        )

    def executar_analise_completa(self) -> Dict[str, Any]:
        """Executa os Pilares 1 e 2 e consolida DataFrames e dicionário JSON."""
        df_estrutural, df_distancias, detalhes_est = self._executar_analise_estrutural()

        si_c_u, si_r_u, ic_c_u = self._executar_simulacoes_visao(
            VISAO_COMB_UNDIR, self.g_comb_undir, directed=False, offset_seed=1000
        )
        si_c_d, si_r_d, ic_c_d = self._executar_simulacoes_visao(
            VISAO_COMB_DIR, self.g_comb_dir, directed=True, offset_seed=2000
        )

        df_si_curvas = pd.DataFrame(si_c_u + si_c_d)
        df_si_resumo = pd.DataFrame(si_r_u + si_r_d)
        df_ic_curvas = pd.DataFrame(ic_c_u + ic_c_d)
        df_ic_ponderada = self._executar_ic_ponderada_real()

        est_u = detalhes_est[VISAO_COMB_UNDIR]
        est_d = detalhes_est[VISAO_COMB_DIR]
        est_q = detalhes_est[VISAO_QUIM_DIR]

        si_u_json = {
            "hubs_t50": self._buscar_valor(
                df_si_resumo, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_HUBS}, "t50_Media"
            ),
            "hubs_auc": self._buscar_valor(
                df_si_resumo, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_HUBS}, "AUC_SI_Media"
            ),
            "sensorial_t50": self._buscar_valor(
                df_si_resumo, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_SENS}, "t50_Media"
            ),
            "sensorial_auc": self._buscar_valor(
                df_si_resumo, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_SENS}, "AUC_SI_Media"
            ),
            "aleatorio_real_t50": self._buscar_valor(
                df_si_resumo, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_ALEAT}, "t50_Media"
            ),
            "er_hubs_t50": self._buscar_valor(
                df_si_resumo, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_ER_HUBS}, "t50_Media"
            ),
            "er_aleatorio_t50": self._buscar_valor(
                df_si_resumo, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_ER_ALEAT}, "t50_Media"
            ),
        }
        si_d_json = {
            "hubs_t_motor_50": self._buscar_valor(
                df_si_resumo,
                {"Visao": VISAO_COMB_DIR, "Cenario": CENARIO_REAL_HUBS},
                "t_motor_50_Media",
            ),
            "sensorial_t_motor_50": self._buscar_valor(
                df_si_resumo,
                {"Visao": VISAO_COMB_DIR, "Cenario": CENARIO_REAL_SENS},
                "t_motor_50_Media",
            ),
        }

        b_ref = self.beta_ic_referencia
        ic_u_json = {
            "hubs_alcance_pct": self._buscar_valor(
                df_ic_curvas,
                {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_HUBS, "Beta": b_ref},
                "Alcance_Final_Pct_Media",
            ),
            "sensorial_alcance_pct": self._buscar_valor(
                df_ic_curvas,
                {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_SENS, "Beta": b_ref},
                "Alcance_Final_Pct_Media",
            ),
            "aleatorio_real_alcance_pct": self._buscar_valor(
                df_ic_curvas,
                {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_ALEAT, "Beta": b_ref},
                "Alcance_Final_Pct_Media",
            ),
            "er_hubs_alcance_pct": self._buscar_valor(
                df_ic_curvas,
                {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_ER_HUBS, "Beta": b_ref},
                "Alcance_Final_Pct_Media",
            ),
            "er_aleatorio_alcance_pct": self._buscar_valor(
                df_ic_curvas,
                {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_ER_ALEAT, "Beta": b_ref},
                "Alcance_Final_Pct_Media",
            ),
        }
        ic_pond_json = {
            "hubs_alcance_pct": self._buscar_valor(
                df_ic_ponderada,
                {"Visao": VISAO_COMB_DIR, "Cenario": CENARIO_REAL_HUBS, "Beta": b_ref},
                "Alcance_Final_Pct_Media",
            ),
            "sensorial_alcance_pct": self._buscar_valor(
                df_ic_ponderada,
                {"Visao": VISAO_COMB_DIR, "Cenario": CENARIO_REAL_SENS, "Beta": b_ref},
                "Alcance_Final_Pct_Media",
            ),
        }

        conclusao = self._montar_conclusao_textual(
            est_u, est_d, si_u_json, ic_u_json, ic_pond_json
        )

        resumo_json_bruto = {
            "metadados": {
                "N": self.g_comb_undir.number_of_nodes(),
                "n_sensoriais": len(self.nos_sensoriais),
                "n_motores": len(self.nos_motores),
                "membros_hubs": list(self.nos_hubs),
                "beta_si_padrao": self.beta_si_padrao,
                "beta_ic_referencia": self.beta_ic_referencia,
            },
            "estrutural": {
                "combinada_undir": {
                    "L_real": est_u["L_real"],
                    "L_er_media": est_u["L_er_media"],
                    "D_real": est_u["D_real"],
                    "D_er_media": est_u["D_er_media"],
                    "E_glob_real": est_u["E_glob_real"],
                    "E_glob_er_media": est_u["E_glob_er_media"],
                    "lambda_c_real": est_u["lambda_c_real"],
                    "lambda_c_er_media": est_u["lambda_c_er_media"],
                    "pct_pares_ate_3_saltos_real": est_u["pct_pares_ate_3_saltos_real"],
                    "pct_pares_ate_3_saltos_er": est_u["pct_pares_ate_3_saltos_er"],
                },
                "combinada_dir": {
                    "L_real": est_d["L_real"],
                    "L_er_media": est_d["L_er_media"],
                    "D_real": est_d["D_real"],
                    "D_er_media": est_d["D_er_media"],
                    "E_glob_real": est_d["E_glob_real"],
                    "E_glob_er_media": est_d["E_glob_er_media"],
                    "L_sensorial_para_motor": est_d["L_sensorial_para_motor"],
                    "alcance_sensorial_para_motor_pct": est_d["alcance_sensorial_para_motor_pct"],
                    "L_hubs_para_motor": est_d["L_hubs_para_motor"],
                },
                "quimica_dir": {
                    "L_real": est_q["L_real"],
                    "D_real": est_q["D_real"],
                    "E_glob_real": est_q["E_glob_real"],
                    "L_sensorial_para_motor": est_q["L_sensorial_para_motor"],
                },
            },
            "simulacao_si": {
                "combinada_undir": si_u_json,
                "combinada_dir": si_d_json,
            },
            "simulacao_ic": {
                "beta_referencia": b_ref,
                "combinada_undir": ic_u_json,
                "combinada_dir_ponderada": ic_pond_json,
            },
            "conclusao_textual": conclusao,
        }

        return {
            "df_estrutural": df_estrutural,
            "df_distancias": df_distancias,
            "df_si_curvas": df_si_curvas,
            "df_si_resumo": df_si_resumo,
            "df_ic_curvas": df_ic_curvas,
            "df_ic_ponderada": df_ic_ponderada,
            "resumo_json": _sanitizar_json(resumo_json_bruto),
        }

    def exibir_relatorio_console(self, resultado: Dict[str, Any]) -> None:
        """Imprime no console o relatório estrutural, dinâmico e o diagnóstico final."""
        df_est = resultado["df_estrutural"]
        df_si = resultado["df_si_resumo"]
        df_ic = resultado["df_ic_curvas"]
        df_ic_p = resultado["df_ic_ponderada"]
        res = resultado["resumo_json"]

        print("\n=== FACILIDADE DE DIFUSÃO DE INFORMAÇÃO — PILAR 1 (MÉTRICAS ESTRUTURAIS) ===")
        cols_est = [
            "Visao", "N", "M", "L_Real", "L_ER_Media",
            "D_Real", "D_ER_Media", "E_Glob_Real", "E_Glob_ER_Media",
            "Lambda_c_Real", "Lambda_c_ER_Media",
        ]
        print(df_est[cols_est].to_string(index=False))

        est_d = res["estrutural"]["combinada_dir"]
        print(
            f"\nCanal Biológico (Combinada Direcionada): "
            f"L(Sensorial -> Motor) = {est_d['L_sensorial_para_motor']:.2f} "
            f"(alcance {est_d['alcance_sensorial_para_motor_pct']:.1f}%) | "
            f"L(Hubs -> Motor) = {est_d['L_hubs_para_motor']:.2f}"
        )

        print(f"\n=== PILAR 2 — SIMULAÇÃO MODELO SI (beta = {self.beta_si_padrao:.2f}, T = {self.max_steps_si}) ===")
        cols_si = [
            "Visao", "Cenario", "t50_Media", "t80_Media",
            "Frac_Atingiu_50_Pct", "AUC_SI_Media", "Infectados_Final_Pct", "t_motor_50_Media",
        ]
        print(df_si[cols_si].to_string(index=False))

        b_ref = self.beta_ic_referencia
        df_ic_ref = df_ic[np.isclose(df_ic["Beta"].to_numpy(dtype=float), b_ref, atol=1e-9)]
        print(f"\n=== PILAR 2 — CASCATA INDEPENDENTE / SIR (beta_ref = {b_ref:.2f}) ===")
        cols_ic = [
            "Visao", "Cenario", "Alcance_Final_Pct_Media",
            "Duracao_Media", "Prob_Cascata_Global_Pct", "Alcance_Motores_Pct_Media",
        ]
        print(df_ic_ref[cols_ic].to_string(index=False))

        df_ic_p_ref = df_ic_p[np.isclose(df_ic_p["Beta"].to_numpy(dtype=float), b_ref, atol=1e-9)]
        print(f"\n--- Complemento Biológico: Cascata Independente Ponderada (beta = {b_ref:.2f}) ---")
        print(df_ic_p_ref[cols_ic].to_string(index=False))

        print("\n=== DIAGNÓSTICO DE DIFUSÃO DE INFORMAÇÃO ===")
        print(res["conclusao_textual"])
