"""
Módulo de análise de Navegabilidade (Roteamento Guloso / Greedy Routing)
e Assortatividade multidimensional do conectoma de C. elegans.

Implementa:
  - 4 protocolos locais de roteamento (Espacial via soma_pos, Grau/Hubs,
    Híbrido Espacial+Grau e Baseline de Caminhada Aleatória);
  - Métricas de Taxa de Sucesso (S_R), Stretch (tau), Eficiência de Navegação
    (E_nav / E_R), mediação pelo núcleo Rich-Club e estratificação por fluxo funcional;
  - Modelos nulos espacial (permutação de soma_pos) e topológico (double_edge_swap);
  - Assortatividade de grau (não-direcionada, ponderada, direcionada), curva k_nn(k),
    homofilia funcional/anatômica e correlação espacial de arestas.

Utiliza apenas biblioteca padrão do Python, NumPy, Pandas e NetworkX (sem SciPy).
"""

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


ESTRATEGIAS_ORDEM = [
    "Espacial (soma_pos)",
    "Grau (Hubs)",
    "Híbrido (Espacial + Grau)",
    "Caminhada Aleatória",
]


def _para_nativo(obj: Any) -> Any:
    """Converte recursivamente tipos NumPy/NaN para tipos nativos serializáveis em JSON."""
    if isinstance(obj, dict):
        return {str(k): _para_nativo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_para_nativo(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_para_nativo(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return obj


def _sanitizar_float(valor: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """Garante float finito válido; retorna default caso seja NaN/Inf/None."""
    if valor is None:
        return default
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _interpretar_assortatividade(valor_r: float) -> str:
    """Classifica qualitativamente o coeficiente de assortatividade r."""
    if valor_r >= 0.25:
        return "Fortemente Assortativo / Homofílico"
    if valor_r >= 0.05:
        return "Moderadamente Assortativo"
    if valor_r > -0.05:
        return "Neutro (Sem Correlação Relevante)"
    if valor_r > -0.25:
        return "Moderadamente Disassortativo"
    return "Fortemente Disassortativo"


class NavigationAssortativityAnalyzer:
    """
    Analisa a navegabilidade local via Roteamento Guloso e o perfil completo
    de assortatividade estrutural, funcional e espacial do conectoma.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        membros_rich_club: Optional[Set[str]] = None,
        n_runs_null: int = 20,
        n_pares_nulo: int = 5000,
        max_hops_fator: int = 3,
        alpha_hibrido: float = 0.55,
        seed_base: int = 42,
    ) -> None:
        self.graph = graph
        self.n_runs_null = max(1, int(n_runs_null))
        self.n_pares_nulo = max(100, int(n_pares_nulo))
        self.max_hops_fator = max(1, int(max_hops_fator))
        self.alpha_hibrido = float(min(max(alpha_hibrido, 0.0), 1.0))
        self.seed_base = int(seed_base)

        # 1. Sanitização de soma_pos em [0.0, 1.0]
        self.soma_pos_map, self.n_soma_pos_ausentes = self._sanitizar_soma_pos()

        # 2. Projeções do multigrafo preservando atributos dos nós
        self.g_comb_undir = self._projetar_grafo_undir(camadas={"Quimica", "Eletrica"})
        self.g_comb_dir = self._projetar_grafo_dir(camadas={"Quimica", "Eletrica"})
        self.g_quim_dir = self._projetar_grafo_dir(camadas={"Quimica"})
        self.g_elet_undir = self._projetar_grafo_undir(camadas={"Eletrica"})

        # 3. Núcleo Rich-Club de referência
        if membros_rich_club is None or len(membros_rich_club) == 0:
            top_nos = sorted(
                self.g_comb_undir.degree(),
                key=lambda item: (-item[1], str(item[0])),
            )[:6]
            self.membros_rich_club: Set[str] = {n for n, _ in top_nos}
        else:
            self.membros_rich_club = set(membros_rich_club) & set(self.graph.nodes)

    def _sanitizar_soma_pos(self) -> Tuple[Dict[str, float], int]:
        """Sanitiza soma_pos para floats em [0, 1], imputando ausentes pela mediana."""
        valores_validos: List[float] = []
        brutos: Dict[str, Optional[float]] = {}

        for no, dados in self.graph.nodes(data=True):
            raw = dados.get("soma_pos", None)
            val_ok: Optional[float] = None
            if raw is not None:
                try:
                    fval = float(raw)
                    if not math.isnan(fval) and not math.isinf(fval) and 0.0 <= fval <= 1.0:
                        val_ok = fval
                except (TypeError, ValueError):
                    val_ok = None
            brutos[no] = val_ok
            if val_ok is not None:
                valores_validos.append(val_ok)

        mediana_soma = float(np.median(valores_validos)) if valores_validos else 0.5
        mediana_soma = min(max(mediana_soma, 0.0), 1.0)

        soma_map: Dict[str, float] = {}
        n_ausentes = 0
        for no, val in brutos.items():
            if val is None:
                soma_map[no] = mediana_soma
                n_ausentes += 1
            else:
                soma_map[no] = min(max(val, 0.0), 1.0)

        return soma_map, n_ausentes

    def _atributos_no(self, no: str) -> Dict[str, Any]:
        """Extrai os atributos relevantes de um vértice com soma_pos sanitizado."""
        dados = self.graph.nodes[no]
        return {
            "tipo_funcional": dados.get("tipo_funcional", "N/A"),
            "soma_pos": self.soma_pos_map[no],
            "regiao_soma": dados.get("regiao_soma", "N/A"),
            "span": dados.get("span", "N/A"),
        }

    def _projetar_grafo_undir(self, camadas: Set[str]) -> nx.Graph:
        """Projeta as camadas selecionadas em grafo simples não-direcionado sem auto-laços."""
        g = nx.Graph()
        for no in self.graph.nodes:
            g.add_node(no, **self._atributos_no(no))

        pesos_eletricos: Dict[Tuple[str, str], float] = {}
        pesos_quimicos: Dict[Tuple[str, str], float] = {}

        for u, v, chave, dados in self.graph.edges(keys=True, data=True):
            if u == v:
                continue
            tipo = dados.get("synapse_type", chave)
            if tipo not in camadas:
                continue
            peso = float(dados.get("weight", 1.0))
            if peso <= 0.0:
                continue
            par = (u, v) if u <= v else (v, u)
            if tipo == "Eletrica":
                pesos_eletricos[par] = max(pesos_eletricos.get(par, 0.0), peso)
            else:
                pesos_quimicos[par] = pesos_quimicos.get(par, 0.0) + peso

        todos_pares = set(pesos_eletricos.keys()) | set(pesos_quimicos.keys())
        for u, v in todos_pares:
            peso_total = pesos_eletricos.get((u, v), 0.0) + pesos_quimicos.get((u, v), 0.0)
            g.add_edge(u, v, weight=peso_total)

        return g

    def _projetar_grafo_dir(self, camadas: Set[str]) -> nx.DiGraph:
        """Projeta as camadas selecionadas em DiGraph sem auto-laços (Elétrica simétrica)."""
        g = nx.DiGraph()
        for no in self.graph.nodes:
            g.add_node(no, **self._atributos_no(no))

        pesos_eletricos: Dict[Tuple[str, str], float] = {}

        for u, v, chave, dados in self.graph.edges(keys=True, data=True):
            if u == v:
                continue
            tipo = dados.get("synapse_type", chave)
            if tipo not in camadas:
                continue
            peso = float(dados.get("weight", 1.0))
            if peso <= 0.0:
                continue

            if tipo == "Eletrica":
                par = (u, v) if u <= v else (v, u)
                pesos_eletricos[par] = max(pesos_eletricos.get(par, 0.0), peso)
            else:
                atual = g[u][v]["weight"] if g.has_edge(u, v) else 0.0
                g.add_edge(u, v, weight=atual + peso)

        for (u, v), peso in pesos_eletricos.items():
            atual_uv = g[u][v]["weight"] if g.has_edge(u, v) else 0.0
            g.add_edge(u, v, weight=atual_uv + peso)
            atual_vu = g[v][u]["weight"] if g.has_edge(v, u) else 0.0
            g.add_edge(v, u, weight=atual_vu + peso)

        return g

    # ------------------------------------------------------------------
    # Motor de Roteamento Guloso (Indexado por Inteiros)
    # ------------------------------------------------------------------
    def _preparar_estruturas_roteamento(
        self,
        g: nx.Graph,
        soma_pos_override: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Converte o grafo em estruturas indexadas por inteiros para roteamento O(1)."""
        nos = sorted(g.nodes(), key=lambda x: str(x))
        n = len(nos)
        no_para_idx = {no: i for i, no in enumerate(nos)}
        eh_dir = g.is_directed()

        mapa_pos = soma_pos_override if soma_pos_override is not None else self.soma_pos_map
        x_pos = [float(mapa_pos[no]) for no in nos]
        tipos = [str(g.nodes[no].get("tipo_funcional", "N/A")) for no in nos]
        is_rc = [no in self.membros_rich_club for no in nos]

        if eh_dir:
            graus = [int(g.out_degree(no)) for no in nos]
        else:
            graus = [int(g.degree(no)) for no in nos]

        k_max = max(graus) if graus else 0
        graus_norm = [float(k) / float(k_max) if k_max > 0 else 0.0 for k in graus]

        adj_list: List[List[int]] = [[] for _ in range(n)]
        adj_set: List[Set[int]] = [set() for _ in range(n)]
        adj_by_deg: List[List[int]] = [[] for _ in range(n)]

        for i, no in enumerate(nos):
            vizinhos = [no_para_idx[v] for v in g.neighbors(no) if v != no]
            adj_list[i] = vizinhos
            adj_set[i] = set(vizinhos)
            adj_by_deg[i] = sorted(vizinhos, key=lambda idx: (-graus[idx], idx))

        # Distâncias mínimas reais entre todos os pares alcançáveis
        pares_alc: List[Tuple[int, int, int]] = []
        max_dist = 0
        soma_inv_sp = 0.0

        for src, dists in nx.all_pairs_shortest_path_length(g):
            s_idx = no_para_idx[src]
            for dst, d_sp in dists.items():
                if dst == src or d_sp < 1:
                    continue
                t_idx = no_para_idx[dst]
                pares_alc.append((s_idx, t_idx, int(d_sp)))
                if d_sp > max_dist:
                    max_dist = int(d_sp)
                soma_inv_sp += 1.0 / float(d_sp)

        diametro_gcc = max(1, max_dist)
        max_hops = max(25, self.max_hops_fator * diametro_gcc)
        total_pares_possiveis = n * (n - 1)
        frac_alcancavel = (
            float(len(pares_alc)) / float(total_pares_possiveis)
            if total_pares_possiveis > 0
            else 0.0
        )
        e_glob = (
            soma_inv_sp / float(total_pares_possiveis)
            if total_pares_possiveis > 0
            else 0.0
        )

        return {
            "n": n,
            "nos": nos,
            "x_pos": x_pos,
            "tipos": tipos,
            "is_rc": is_rc,
            "graus": graus,
            "graus_norm": graus_norm,
            "k_max": k_max,
            "adj_list": adj_list,
            "adj_set": adj_set,
            "adj_by_deg": adj_by_deg,
            "pares_alc": pares_alc,
            "diametro_gcc": diametro_gcc,
            "max_hops": max_hops,
            "frac_alcancavel": frac_alcancavel,
            "e_glob": e_glob,
        }

    def _simular_estrategia(
        self,
        est: Dict[str, Any],
        estrategia: str,
        pares: List[Tuple[int, int, int]],
        seed: int = 42,
        coletar_fluxo: bool = False,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Executa uma estratégia de roteamento local sobre os pares informados."""
        n = est["n"]
        x_pos = est["x_pos"]
        tipos = est["tipos"]
        is_rc = est["is_rc"]
        graus = est["graus"]
        graus_norm = est["graus_norm"]
        adj_list = est["adj_list"]
        adj_set = est["adj_set"]
        adj_by_deg = est["adj_by_deg"]
        max_hops = est["max_hops"]
        alpha = self.alpha_hibrido
        um_menos_alpha = 1.0 - alpha

        # Gerador determinístico para caminhada aleatória (derivado de np.random.default_rng)
        np_rng = np.random.default_rng(seed)
        py_rng = random.Random(int(np_rng.integers(0, 2**32 - 1)))

        n_pares = len(pares)
        stretches: List[float] = []
        hops_sucesso: List[int] = []
        soma_sp = 0
        n_sucesso = 0
        n_otimo = 0
        soma_inv_greedy = 0.0
        n_sucesso_d2 = 0
        n_via_rc = 0

        fluxo_acum: Dict[Tuple[str, str], List[Any]] = {}

        for s, t, d_sp in pares:
            soma_sp += d_sp
            u = s
            visitados = {s}
            passos = 0
            sucesso = False
            via_rc = False
            x_t = x_pos[t]

            if estrategia == "Espacial (soma_pos)":
                while passos < max_hops:
                    if t in adj_set[u]:
                        passos += 1
                        sucesso = True
                        break
                    viz = adj_list[u]
                    if not viz:
                        break
                    melhor_v = -1
                    melhor_dist = 2.0
                    melhor_deg = -1
                    for v in viz:
                        d_v = abs(x_pos[v] - x_t)
                        k_v = graus[v]
                        if (
                            d_v < melhor_dist
                            or (d_v == melhor_dist and (k_v > melhor_deg or (k_v == melhor_deg and v < melhor_v)))
                        ):
                            melhor_dist = d_v
                            melhor_deg = k_v
                            melhor_v = v
                    if melhor_v < 0 or melhor_v in visitados:
                        break
                    u = melhor_v
                    visitados.add(u)
                    passos += 1
                    if is_rc[u]:
                        via_rc = True

            elif estrategia == "Grau (Hubs)":
                while passos < max_hops:
                    if t in adj_set[u]:
                        passos += 1
                        sucesso = True
                        break
                    proximo = -1
                    for v in adj_by_deg[u]:
                        if v not in visitados:
                            proximo = v
                            break
                    if proximo < 0:
                        break
                    u = proximo
                    visitados.add(u)
                    passos += 1
                    if is_rc[u]:
                        via_rc = True

            elif estrategia == "Híbrido (Espacial + Grau)":
                while passos < max_hops:
                    if t in adj_set[u]:
                        passos += 1
                        sucesso = True
                        break
                    viz = adj_list[u]
                    if not viz:
                        break
                    melhor_v = -1
                    melhor_custo = 1e9
                    melhor_deg = -1
                    for v in viz:
                        if v in visitados:
                            continue
                        custo = alpha * abs(x_pos[v] - x_t) - um_menos_alpha * graus_norm[v]
                        k_v = graus[v]
                        if (
                            custo < melhor_custo
                            or (custo == melhor_custo and (k_v > melhor_deg or (k_v == melhor_deg and v < melhor_v)))
                        ):
                            melhor_custo = custo
                            melhor_deg = k_v
                            melhor_v = v
                    if melhor_v < 0:
                        break
                    u = melhor_v
                    visitados.add(u)
                    passos += 1
                    if is_rc[u]:
                        via_rc = True

            elif estrategia == "Caminhada Aleatória":
                while passos < max_hops:
                    if t in adj_set[u]:
                        passos += 1
                        sucesso = True
                        break
                    candidatos = [v for v in adj_list[u] if v not in visitados]
                    if not candidatos:
                        break
                    u = candidatos[py_rng.randrange(len(candidatos))]
                    visitados.add(u)
                    passos += 1
                    if is_rc[u]:
                        via_rc = True
            else:
                raise ValueError(f"Estratégia de roteamento desconhecida: {estrategia}")

            if sucesso:
                # Invariante matemático: caminho guloso nunca pode ser menor que o caminho mínimo
                if passos < d_sp:
                    raise AssertionError(
                        f"Violação de invariante de stretch em '{estrategia}': "
                        f"h_greedy={passos} < d_sp={d_sp} para par ({s}, {t})."
                    )
                tau = float(passos) / float(d_sp)
                n_sucesso += 1
                stretches.append(tau)
                hops_sucesso.append(passos)
                soma_inv_greedy += 1.0 / float(passos)
                if passos == d_sp:
                    n_otimo += 1
                if d_sp >= 2:
                    n_sucesso_d2 += 1
                    if via_rc:
                        n_via_rc += 1

            if coletar_fluxo:
                chave_fluxo = (tipos[s], tipos[t])
                bucket = fluxo_acum.get(chave_fluxo)
                if bucket is None:
                    bucket = [0, 0, []]  # [n_pares, n_sucesso, stretches]
                    fluxo_acum[chave_fluxo] = bucket
                bucket[0] += 1
                if sucesso:
                    bucket[1] += 1
                    bucket[2].append(float(passos) / float(d_sp))

        taxa_sucesso_pct = (100.0 * n_sucesso / n_pares) if n_pares > 0 else 0.0
        hops_sp_medio = (float(soma_sp) / float(n_pares)) if n_pares > 0 else float("nan")

        if stretches:
            arr_tau = np.asarray(stretches, dtype=float)
            stretch_medio = float(np.mean(arr_tau))
            stretch_mediana = float(np.median(arr_tau))
            stretch_p90 = float(np.percentile(arr_tau, 90))
            frac_otimo_pct = 100.0 * float(n_otimo) / float(n_sucesso)
            hops_greedy_medio = float(np.mean(hops_sucesso))
        else:
            stretch_medio = float("nan")
            stretch_mediana = float("nan")
            stretch_p90 = float("nan")
            frac_otimo_pct = 0.0
            hops_greedy_medio = float("nan")

        # Eficiência de navegação normalizada por N*(N-1) (escalada se for amostra)
        total_pares_possiveis = n * (n - 1)
        if n_pares == len(est["pares_alc"]):
            e_nav = (
                soma_inv_greedy / float(total_pares_possiveis)
                if total_pares_possiveis > 0
                else 0.0
            )
            e_glob = est["e_glob"]
        else:
            soma_inv_sp_amostra = sum(1.0 / float(d) for _, _, d in pares)
            fator_escala = (
                est["frac_alcancavel"] / float(n_pares) if n_pares > 0 else 0.0
            )
            e_nav = soma_inv_greedy * fator_escala
            e_glob = soma_inv_sp_amostra * fator_escala

        razao_eficiencia = (e_nav / e_glob) if e_glob > 0.0 else 0.0
        razao_eficiencia = min(max(razao_eficiencia, 0.0), 1.0)
        pct_via_rc = (
            100.0 * float(n_via_rc) / float(n_sucesso_d2)
            if n_sucesso_d2 > 0
            else 0.0
        )

        metricas = {
            "N_Pares_Alcancaveis": int(n_pares),
            "Fracao_Alcancavel": float(est["frac_alcancavel"]),
            "Taxa_Sucesso_Pct": float(taxa_sucesso_pct),
            "Stretch_Medio": float(stretch_medio),
            "Stretch_Mediana": float(stretch_mediana),
            "Stretch_P90": float(stretch_p90),
            "Frac_Otimo_Pct": float(frac_otimo_pct),
            "Hops_Greedy_Medio": float(hops_greedy_medio),
            "Hops_SP_Medio": float(hops_sp_medio),
            "Eficiencia_Nav": float(e_nav),
            "Eficiencia_Global": float(e_glob),
            "Razao_Eficiencia": float(razao_eficiencia),
            "Pct_Rotas_Via_RichClub": float(pct_via_rc),
        }

        linhas_fluxo: List[Dict[str, Any]] = []
        if coletar_fluxo:
            for (t_orig, t_dest), (np_f, ns_f, st_f) in sorted(fluxo_acum.items()):
                linhas_fluxo.append(
                    {
                        "Estrategia": estrategia,
                        "Tipo_Origem": t_orig,
                        "Tipo_Destino": t_dest,
                        "N_Pares": int(np_f),
                        "N_Sucesso": int(ns_f),
                        "Taxa_Sucesso_Pct": (100.0 * ns_f / np_f) if np_f > 0 else 0.0,
                        "Stretch_Medio": float(np.mean(st_f)) if st_f else float("nan"),
                        "Stretch_Mediana": float(np.median(st_f)) if st_f else float("nan"),
                    }
                )

        return metricas, linhas_fluxo

    def _amostrar_pares(
        self, pares: List[Tuple[int, int, int]], n_amostra: int, seed: int
    ) -> List[Tuple[int, int, int]]:
        """Amostra sem reposição até n_amostra pares alcançáveis de forma determinística."""
        total = len(pares)
        if total <= n_amostra:
            return pares
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=n_amostra, replace=False)
        return [pares[int(idx)] for idx in indices]

    def _executar_modelos_nulos(
        self, est_undir: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executa os dois modelos nulos na rede Combinada (Não-Direcionada):
          1. Nulo Espacial (permutação de soma_pos mantendo topologia exata);
          2. Nulo Topológico (double_edge_swap preservando sequência de graus),
             aproveitando o mesmo ensemble para r_k nulo e curva k_nn(k) nula.
        """
        nos = est_undir["nos"]
        x_orig = np.asarray([self.soma_pos_map[no] for no in nos], dtype=float)
        arestas_undir = list(self.g_comb_undir.edges())
        m_arestas = len(arestas_undir)

        # 1. Nulo Espacial (permutação de soma_pos)
        sr_espacial_nulo: List[float] = []
        dist_media_espacial_nulo: List[float] = []

        for i in range(self.n_runs_null):
            seed_i = self.seed_base + i
            rng_i = np.random.default_rng(seed_i)
            x_perm = rng_i.permutation(x_orig)
            mapa_perm = {no: float(x_perm[idx]) for idx, no in enumerate(nos)}

            if arestas_undir:
                dists_e = [abs(mapa_perm[u] - mapa_perm[v]) for u, v in arestas_undir]
                dist_media_espacial_nulo.append(float(np.mean(dists_e)))

            est_perm = dict(est_undir)
            est_perm["x_pos"] = [float(x_perm[idx]) for idx in range(len(nos))]
            pares_amostra = self._amostrar_pares(
                est_undir["pares_alc"], self.n_pares_nulo, seed=seed_i + 1000
            )
            met, _ = self._simular_estrategia(
                est_perm, "Espacial (soma_pos)", pares_amostra, seed=seed_i + 2000
            )
            sr_espacial_nulo.append(met["Taxa_Sucesso_Pct"])

        # 2. Nulo Topológico (double_edge_swap preservando graus)
        sr_rewiring: Dict[str, List[float]] = {e: [] for e in ESTRATEGIAS_ORDEM}
        r_grau_nulo_lista: List[float] = []
        knn_nulo_acum: Dict[int, List[float]] = {}

        for i in range(self.n_runs_null):
            seed_i = self.seed_base + 500 + i
            g_rew = self.g_comb_undir.copy()
            if m_arestas >= 4:
                try:
                    nx.double_edge_swap(
                        g_rew,
                        nswap=2 * m_arestas,
                        max_tries=20 * m_arestas,
                        seed=seed_i,
                    )
                except nx.NetworkXAlgorithmError:
                    pass

            # Assortatividade de grau e k_nn(k) na rede recableada
            r_rew = _sanitizar_float(nx.degree_assortativity_coefficient(g_rew), default=None)
            if r_rew is not None:
                r_grau_nulo_lista.append(r_rew)

            knn_rew = nx.average_degree_connectivity(g_rew)
            for k_val, knn_val in knn_rew.items():
                if k_val > 0 and knn_val is not None and not math.isnan(float(knn_val)):
                    knn_nulo_acum.setdefault(int(k_val), []).append(float(knn_val))

            est_rew = self._preparar_estruturas_roteamento(g_rew)
            pares_amostra = self._amostrar_pares(
                est_rew["pares_alc"], self.n_pares_nulo, seed=seed_i + 3000
            )
            for idx_e, estrategia in enumerate(ESTRATEGIAS_ORDEM):
                met, _ = self._simular_estrategia(
                    est_rew,
                    estrategia,
                    pares_amostra,
                    seed=seed_i + 4000 + idx_e * 97,
                )
                sr_rewiring[estrategia].append(met["Taxa_Sucesso_Pct"])

        return {
            "nulo_espacial_sr": sr_espacial_nulo,
            "dist_media_nula": float(np.mean(dist_media_espacial_nulo))
            if dist_media_espacial_nulo
            else 0.0,
            "nulo_rewiring_sr": sr_rewiring,
            "r_grau_nulo_lista": r_grau_nulo_lista,
            "knn_nulo_acum": knn_nulo_acum,
        }

    # ------------------------------------------------------------------
    # Bloco de Assortatividade
    # ------------------------------------------------------------------
    def _calcular_correlacao_espacial(
        self, g: nx.Graph
    ) -> Tuple[float, float]:
        """
        Calcula a correlação de Pearson de soma_pos nas arestas (simétrica em grafos
        não-direcionados) e a distância física média |x_u - x_v| das arestas.
        """
        if g.number_of_edges() == 0:
            return 0.0, 0.0

        xs: List[float] = []
        ys: List[float] = []
        dists: List[float] = []
        eh_dir = g.is_directed()

        for u, v in g.edges():
            if u == v:
                continue
            xu = self.soma_pos_map[u]
            xv = self.soma_pos_map[v]
            dists.append(abs(xu - xv))
            xs.append(xu)
            ys.append(xv)
            if not eh_dir:
                xs.append(xv)
                ys.append(xu)

        if len(xs) < 2:
            return 0.0, float(np.mean(dists)) if dists else 0.0

        arr_x = np.asarray(xs, dtype=float)
        arr_y = np.asarray(ys, dtype=float)
        dx = arr_x - np.mean(arr_x)
        dy = arr_y - np.mean(arr_y)
        denom = math.sqrt(float(np.sum(dx * dx) * np.sum(dy * dy)))
        r_esp = float(np.sum(dx * dy) / denom) if denom > 1e-12 else 0.0
        r_esp = max(-1.0, min(1.0, _sanitizar_float(r_esp, 0.0) or 0.0))
        dist_media = float(np.mean(dists)) if dists else 0.0
        return r_esp, dist_media

    def _assortatividade_atributo_segura(
        self, g: nx.Graph, atributo: str
    ) -> float:
        """Calcula assortatividade de atributo discreto ignorando nós com 'N/A' ou None."""
        nos_validos = []
        valores_distintos = set()
        for no, dados in g.nodes(data=True):
            val = dados.get(atributo, None)
            if val is None:
                continue
            val_str = str(val).strip()
            if val_str in {"", "N/A", "nan", "None"}:
                continue
            nos_validos.append(no)
            valores_distintos.add(val_str)

        if len(nos_validos) < 2 or len(valores_distintos) < 2:
            return 0.0

        sub = g.subgraph(nos_validos)
        if sub.number_of_edges() == 0:
            return 0.0

        try:
            r_val = nx.attribute_assortativity_coefficient(sub, atributo)
        except Exception:
            return 0.0

        r_clean = _sanitizar_float(r_val, 0.0) or 0.0
        return max(-1.0, min(1.0, r_clean))

    def _assortatividade_grau_segura(
        self,
        g: nx.Graph,
        x: str = "out",
        y: str = "in",
        weight: Optional[str] = None,
    ) -> float:
        """Wrapper seguro contra NaN/exceções para nx.degree_assortativity_coefficient."""
        if g.number_of_edges() < 2:
            return 0.0
        try:
            if g.is_directed():
                r_val = nx.degree_assortativity_coefficient(g, x=x, y=y, weight=weight)
            else:
                r_val = nx.degree_assortativity_coefficient(g, weight=weight)
        except Exception:
            return 0.0
        r_clean = _sanitizar_float(r_val, 0.0) or 0.0
        return max(-1.0, min(1.0, r_clean))

    def calcular_assortatividade(
        self, nulos: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Executa o pacote completo de Assortatividade + curva k_nn(k)."""
        # 1. Grau: Combinada, Elétrica e Química Direcionada
        r_comb = self._assortatividade_grau_segura(self.g_comb_undir)
        r_comb_w = self._assortatividade_grau_segura(self.g_comb_undir, weight="weight")
        r_elet = self._assortatividade_grau_segura(self.g_elet_undir)
        r_elet_w = self._assortatividade_grau_segura(self.g_elet_undir, weight="weight")

        r_q_out_in = self._assortatividade_grau_segura(self.g_quim_dir, x="out", y="in")
        r_q_out_out = self._assortatividade_grau_segura(self.g_quim_dir, x="out", y="out")
        r_q_in_in = self._assortatividade_grau_segura(self.g_quim_dir, x="in", y="in")
        r_q_in_out = self._assortatividade_grau_segura(self.g_quim_dir, x="in", y="out")

        r_nulo_lista = nulos["r_grau_nulo_lista"]
        r_nulo_media = float(np.mean(r_nulo_lista)) if r_nulo_lista else 0.0
        r_nulo_std = float(np.std(r_nulo_lista, ddof=1)) if len(r_nulo_lista) > 1 else 0.0
        z_r_grau = (
            (r_comb - r_nulo_media) / r_nulo_std if r_nulo_std > 1e-12 else 0.0
        )

        # 2. Curva k_nn(k) + inclinação log-log beta_knn
        knn_real_map = nx.average_degree_connectivity(self.g_comb_undir)
        knn_nulo_acum = nulos["knn_nulo_acum"]

        linhas_knn: List[Dict[str, Any]] = []
        ks_fit: List[float] = []
        knns_fit: List[float] = []

        for k_val in sorted(knn_real_map.keys()):
            if k_val <= 0:
                continue
            knn_r = _sanitizar_float(knn_real_map[k_val], default=None)
            if knn_r is None or knn_r <= 0.0:
                continue
            vals_n = knn_nulo_acum.get(int(k_val), [])
            knn_n_med = float(np.mean(vals_n)) if vals_n else float("nan")
            knn_n_std = float(np.std(vals_n, ddof=1)) if len(vals_n) > 1 else 0.0
            linhas_knn.append(
                {
                    "k": int(k_val),
                    "knn_real": float(knn_r),
                    "knn_nulo_media": knn_n_med,
                    "knn_nulo_std": knn_n_std,
                }
            )
            ks_fit.append(float(k_val))
            knns_fit.append(float(knn_r))

        df_knn = pd.DataFrame(linhas_knn)
        if len(ks_fit) >= 2:
            slope, intercept = np.polyfit(np.log(ks_fit), np.log(knns_fit), 1)
            beta_knn = _sanitizar_float(slope, 0.0) or 0.0
            intercept_knn = _sanitizar_float(intercept, 0.0) or 0.0
        else:
            beta_knn = 0.0
            intercept_knn = 0.0

        # 3. Atributos discretos (tipo_funcional, regiao_soma, span) nas 3 camadas
        camadas_attr = [
            ("Combinada", self.g_comb_undir),
            ("Quimica", self.g_quim_dir),
            ("Eletrica", self.g_elet_undir),
        ]
        attr_resultados: Dict[Tuple[str, str], float] = {}
        for nome_cam, g_cam in camadas_attr:
            for attr in ("tipo_funcional", "regiao_soma", "span"):
                attr_resultados[(attr, nome_cam)] = self._assortatividade_atributo_segura(
                    g_cam, attr
                )

        # 4. Correlação espacial de arestas (soma_pos) e distância média
        r_esp_comb, dist_media_comb = self._calcular_correlacao_espacial(self.g_comb_undir)
        r_esp_quim, _ = self._calcular_correlacao_espacial(self.g_quim_dir)
        r_esp_elet, _ = self._calcular_correlacao_espacial(self.g_elet_undir)
        dist_media_nula = float(nulos["dist_media_nula"])

        # Montar tabela consolidada df_assortatividade
        linhas_assort: List[Dict[str, Any]] = [
            {
                "Metrica": "Grau (r_k)",
                "Camada": "Combinada (Não-Direcionada)",
                "Valor_r": r_comb,
                "Referencia_Nula": r_nulo_media,
                "Std_Nula": r_nulo_std,
                "Z_Score": z_r_grau,
                "Interpretacao": _interpretar_assortatividade(r_comb),
            },
            {
                "Metrica": "Grau Ponderado (r_k^w)",
                "Camada": "Combinada (Não-Direcionada)",
                "Valor_r": r_comb_w,
                "Referencia_Nula": np.nan,
                "Std_Nula": np.nan,
                "Z_Score": np.nan,
                "Interpretacao": _interpretar_assortatividade(r_comb_w),
            },
            {
                "Metrica": "Grau (r_k)",
                "Camada": "Eletrica (Não-Direcionada)",
                "Valor_r": r_elet,
                "Referencia_Nula": np.nan,
                "Std_Nula": np.nan,
                "Z_Score": np.nan,
                "Interpretacao": _interpretar_assortatividade(r_elet),
            },
            {
                "Metrica": "Grau Ponderado (r_k^w)",
                "Camada": "Eletrica (Não-Direcionada)",
                "Valor_r": r_elet_w,
                "Referencia_Nula": np.nan,
                "Std_Nula": np.nan,
                "Z_Score": np.nan,
                "Interpretacao": _interpretar_assortatividade(r_elet_w),
            },
            {
                "Metrica": "Grau Direcionado (out -> in)",
                "Camada": "Quimica (Direcionada)",
                "Valor_r": r_q_out_in,
                "Referencia_Nula": np.nan,
                "Std_Nula": np.nan,
                "Z_Score": np.nan,
                "Interpretacao": _interpretar_assortatividade(r_q_out_in),
            },
            {
                "Metrica": "Grau Direcionado (out -> out)",
                "Camada": "Quimica (Direcionada)",
                "Valor_r": r_q_out_out,
                "Referencia_Nula": np.nan,
                "Std_Nula": np.nan,
                "Z_Score": np.nan,
                "Interpretacao": _interpretar_assortatividade(r_q_out_out),
            },
            {
                "Metrica": "Grau Direcionado (in -> in)",
                "Camada": "Quimica (Direcionada)",
                "Valor_r": r_q_in_in,
                "Referencia_Nula": np.nan,
                "Std_Nula": np.nan,
                "Z_Score": np.nan,
                "Interpretacao": _interpretar_assortatividade(r_q_in_in),
            },
            {
                "Metrica": "Grau Direcionado (in -> out)",
                "Camada": "Quimica (Direcionada)",
                "Valor_r": r_q_in_out,
                "Referencia_Nula": np.nan,
                "Std_Nula": np.nan,
                "Z_Score": np.nan,
                "Interpretacao": _interpretar_assortatividade(r_q_in_out),
            },
        ]

        nomes_attr_legiveis = {
            "tipo_funcional": "Homofilia Funcional (tipo_funcional)",
            "regiao_soma": "Homofilia Anatômica (regiao_soma)",
            "span": "Homofilia de Alcance (span)",
        }
        for attr in ("tipo_funcional", "regiao_soma", "span"):
            for nome_cam in ("Combinada", "Quimica", "Eletrica"):
                val_r = attr_resultados[(attr, nome_cam)]
                linhas_assort.append(
                    {
                        "Metrica": nomes_attr_legiveis[attr],
                        "Camada": nome_cam,
                        "Valor_r": val_r,
                        "Referencia_Nula": np.nan,
                        "Std_Nula": np.nan,
                        "Z_Score": np.nan,
                        "Interpretacao": _interpretar_assortatividade(val_r),
                    }
                )

        for nome_cam, val_r in (
            ("Combinada", r_esp_comb),
            ("Quimica", r_esp_quim),
            ("Eletrica", r_esp_elet),
        ):
            linhas_assort.append(
                {
                    "Metrica": "Correlação Espacial (soma_pos)",
                    "Camada": nome_cam,
                    "Valor_r": val_r,
                    "Referencia_Nula": 0.0 if nome_cam == "Combinada" else np.nan,
                    "Std_Nula": np.nan,
                    "Z_Score": np.nan,
                    "Interpretacao": _interpretar_assortatividade(val_r),
                }
            )

        df_assortatividade = pd.DataFrame(linhas_assort)

        resumo_assort = {
            "r_grau_combinada": float(r_comb),
            "r_grau_combinada_ponderada": float(r_comb_w),
            "r_grau_nulo_media": float(r_nulo_media),
            "r_grau_nulo_std": float(r_nulo_std),
            "z_score_r_grau": float(z_r_grau),
            "beta_knn": float(beta_knn),
            "intercept_knn": float(intercept_knn),
            "r_quimica_out_in": float(r_q_out_in),
            "r_quimica_out_out": float(r_q_out_out),
            "r_quimica_in_in": float(r_q_in_in),
            "r_quimica_in_out": float(r_q_in_out),
            "r_eletrica_grau": float(r_elet),
            "r_eletrica_grau_ponderada": float(r_elet_w),
            "r_homofilia_tipo_funcional": float(attr_resultados[("tipo_funcional", "Combinada")]),
            "r_homofilia_regiao_soma": float(attr_resultados[("regiao_soma", "Combinada")]),
            "r_homofilia_span": float(attr_resultados[("span", "Combinada")]),
            "r_espacial_soma_pos": float(r_esp_comb),
            "distancia_fisica_media_aresta": float(dist_media_comb),
            "distancia_fisica_media_nula": float(dist_media_nula),
        }

        return df_assortatividade, df_knn, resumo_assort

    # ------------------------------------------------------------------
    # Orquestração Completa e Relatório no Console
    # ------------------------------------------------------------------
    def _gerar_conclusao_textual(
        self,
        df_nav_undir: pd.DataFrame,
        resumo_assort: Dict[str, Any],
    ) -> str:
        """Gera veredito interpretativo respondendo se é fácil navegar no conectoma."""
        idx_melhor = df_nav_undir["Taxa_Sucesso_Pct"].idxmax()
        melhor = df_nav_undir.loc[idx_melhor]
        esp = df_nav_undir[df_nav_undir["Estrategia"] == "Espacial (soma_pos)"].iloc[0]

        sr_melhor = float(melhor["Taxa_Sucesso_Pct"])
        tau_melhor = float(melhor["Stretch_Medio"])
        er_melhor = float(melhor["Razao_Eficiencia"])
        sr_esp = float(esp["Taxa_Sucesso_Pct"])
        r_k = float(resumo_assort["r_grau_combinada"])
        r_esp = float(resumo_assort["r_espacial_soma_pos"])

        if sr_melhor >= 85.0 and tau_melhor <= 1.6:
            nivel = "Sim — a rede é altamente navegável por regras puramente locais."
        elif sr_melhor >= 65.0:
            nivel = "Moderadamente fácil — o roteamento local atinge alta taxa de entrega com alongamento controlado."
        else:
            nivel = "Navegação local desafiadora — exige combinação de heurísticas ou conhecimento topológico."

        perfil_grau = (
            "disassortativo por grau"
            if r_k < -0.05
            else ("assortativo por grau" if r_k > 0.05 else "neutro em assortatividade de grau")
        )

        return (
            f"{nivel} A melhor estratégia local na camada combinada foi '{melhor['Estrategia']}' "
            f"(S_R = {sr_melhor:.1f}%, stretch médio τ = {tau_melhor:.2f}, razão de eficiência E_R = {er_melhor:.2f}), "
            f"enquanto o roteamento puramente espacial via soma_pos obteve S_R = {sr_esp:.1f}% "
            f"(sustentado pela forte correlação espacial das sinapses r_espacial = {r_esp:.3f}). "
            f"Topologicamente, o conectoma é {perfil_grau} (r_k = {r_k:.3f}), favorecendo "
            f"a concentração de rotas através de hubs de alto grau (Rich-Club)."
        )

    def executar_analise_completa(self) -> Dict[str, Any]:
        """Executa roteamento guloso, modelos nulos, assortatividade e consolidação."""
        est_undir = self._preparar_estruturas_roteamento(self.g_comb_undir)
        est_dir = self._preparar_estruturas_roteamento(self.g_comb_dir)

        nulos = self._executar_modelos_nulos(est_undir)

        linhas_nav: List[Dict[str, Any]] = []
        linhas_fluxo_todas: List[Dict[str, Any]] = []

        # Visão 1: Combinada (Não-Direcionada)
        for idx_e, estrategia in enumerate(ESTRATEGIAS_ORDEM):
            met, linhas_fluxo = self._simular_estrategia(
                est_undir,
                estrategia,
                est_undir["pares_alc"],
                seed=self.seed_base + 100 + idx_e * 31,
                coletar_fluxo=True,
            )
            linhas_fluxo_todas.extend(linhas_fluxo)

            if estrategia == "Espacial (soma_pos)":
                nulo_tipo = "Nulo_Espacial"
                vals_nulo = nulos["nulo_espacial_sr"]
            else:
                nulo_tipo = "Nulo_Rewiring"
                vals_nulo = nulos["nulo_rewiring_sr"][estrategia]

            nulo_media = float(np.mean(vals_nulo)) if vals_nulo else float("nan")
            nulo_std = float(np.std(vals_nulo, ddof=1)) if len(vals_nulo) > 1 else 0.0
            z_score = (
                (met["Taxa_Sucesso_Pct"] - nulo_media) / nulo_std
                if nulo_std > 1e-12
                else 0.0
            )

            linha = {
                "Visao": "Combinada (Não-Direcionada)",
                "Estrategia": estrategia,
                **met,
                "Nulo_Tipo": nulo_tipo,
                "Nulo_Sucesso_Media": nulo_media,
                "Nulo_Sucesso_Std": nulo_std,
                "Z_Score_Sucesso": z_score,
            }
            linhas_nav.append(linha)

        # Visão 2: Combinada (Direcionada)
        for idx_e, estrategia in enumerate(ESTRATEGIAS_ORDEM):
            met, _ = self._simular_estrategia(
                est_dir,
                estrategia,
                est_dir["pares_alc"],
                seed=self.seed_base + 200 + idx_e * 31,
                coletar_fluxo=False,
            )
            linha = {
                "Visao": "Combinada (Direcionada)",
                "Estrategia": estrategia,
                **met,
                "Nulo_Tipo": "N/A",
                "Nulo_Sucesso_Media": np.nan,
                "Nulo_Sucesso_Std": np.nan,
                "Z_Score_Sucesso": np.nan,
            }
            linhas_nav.append(linha)

        colunas_nav = [
            "Visao",
            "Estrategia",
            "N_Pares_Alcancaveis",
            "Fracao_Alcancavel",
            "Taxa_Sucesso_Pct",
            "Stretch_Medio",
            "Stretch_Mediana",
            "Stretch_P90",
            "Frac_Otimo_Pct",
            "Hops_Greedy_Medio",
            "Hops_SP_Medio",
            "Eficiencia_Nav",
            "Eficiencia_Global",
            "Razao_Eficiencia",
            "Pct_Rotas_Via_RichClub",
            "Nulo_Tipo",
            "Nulo_Sucesso_Media",
            "Nulo_Sucesso_Std",
            "Z_Score_Sucesso",
        ]
        df_navegacao = pd.DataFrame(linhas_nav)[colunas_nav]
        df_nav_fluxo = pd.DataFrame(linhas_fluxo_todas)

        # Assortatividade + k_nn(k)
        df_assortatividade, df_knn, resumo_assort = self.calcular_assortatividade(nulos)

        # Montar resumo JSON conforme contrato do plano
        df_undir = df_navegacao[df_navegacao["Visao"] == "Combinada (Não-Direcionada)"]
        linha_esp = df_undir[df_undir["Estrategia"] == "Espacial (soma_pos)"].iloc[0]
        linha_grau = df_undir[df_undir["Estrategia"] == "Grau (Hubs)"].iloc[0]
        linha_hib = df_undir[df_undir["Estrategia"] == "Híbrido (Espacial + Grau)"].iloc[0]

        df_gulosas = df_undir[df_undir["Estrategia"] != "Caminhada Aleatória"]
        melhor_estrategia_undir = str(
            df_gulosas.loc[df_gulosas["Taxa_Sucesso_Pct"].idxmax(), "Estrategia"]
        )

        def _buscar_fluxo_sm(nome_est: str) -> Tuple[float, float]:
            sub = df_nav_fluxo[
                (df_nav_fluxo["Estrategia"] == nome_est)
                & (df_nav_fluxo["Tipo_Origem"] == "Sensorial")
                & (df_nav_fluxo["Tipo_Destino"] == "Motor")
            ]
            if sub.empty:
                return 0.0, 1.0
            row = sub.iloc[0]
            return (
                _sanitizar_float(row["Taxa_Sucesso_Pct"], 0.0) or 0.0,
                _sanitizar_float(row["Stretch_Medio"], 1.0) or 1.0,
            )

        sm_esp_sr, sm_esp_tau = _buscar_fluxo_sm("Espacial (soma_pos)")
        sm_grau_sr, sm_grau_tau = _buscar_fluxo_sm("Grau (Hubs)")
        sm_hib_sr, sm_hib_tau = _buscar_fluxo_sm("Híbrido (Espacial + Grau)")

        conclusao = self._gerar_conclusao_textual(df_undir, resumo_assort)

        resumo_json = _para_nativo(
            {
                "metadados": {
                    "N": int(self.g_comb_undir.number_of_nodes()),
                    "M_combinada": int(self.g_comb_undir.number_of_edges()),
                    "n_soma_pos_ausentes": int(self.n_soma_pos_ausentes),
                    "membros_rich_club": sorted(self.membros_rich_club),
                },
                "navegacao": {
                    "melhor_estrategia_undir": melhor_estrategia_undir,
                    "taxa_sucesso_espacial_pct": float(linha_esp["Taxa_Sucesso_Pct"]),
                    "stretch_medio_espacial": float(linha_esp["Stretch_Medio"]),
                    "taxa_sucesso_grau_pct": float(linha_grau["Taxa_Sucesso_Pct"]),
                    "stretch_medio_grau": float(linha_grau["Stretch_Medio"]),
                    "taxa_sucesso_hibrido_pct": float(linha_hib["Taxa_Sucesso_Pct"]),
                    "stretch_medio_hibrido": float(linha_hib["Stretch_Medio"]),
                    "sensorial_para_motor": {
                        "espacial_sucesso_pct": float(sm_esp_sr),
                        "espacial_stretch_medio": float(sm_esp_tau),
                        "grau_sucesso_pct": float(sm_grau_sr),
                        "grau_stretch_medio": float(sm_grau_tau),
                        "hibrido_sucesso_pct": float(sm_hib_sr),
                        "hibrido_stretch_medio": float(sm_hib_tau),
                    },
                },
                "assortatividade": resumo_assort,
                "conclusao_textual": conclusao,
            }
        )

        return {
            "df_navegacao": df_navegacao,
            "df_nav_fluxo": df_nav_fluxo,
            "df_assortatividade": df_assortatividade,
            "df_knn": df_knn,
            "resumo_json": resumo_json,
        }

    def exibir_relatorio_console(self, resultado: Dict[str, Any]) -> None:
        """Imprime tabelas formatadas e veredito conciso no console."""
        df_nav = resultado["df_navegacao"]
        df_assort = resultado["df_assortatividade"]
        resumo = resultado["resumo_json"]

        print("\n=== NAVEGABILIDADE POR ROTEAMENTO GULOSO (GREEDY ROUTING) ===")
        cols_exibir = [
            "Visao",
            "Estrategia",
            "Taxa_Sucesso_Pct",
            "Stretch_Medio",
            "Stretch_P90",
            "Frac_Otimo_Pct",
            "Razao_Eficiencia",
            "Pct_Rotas_Via_RichClub",
            "Nulo_Sucesso_Media",
            "Z_Score_Sucesso",
        ]
        print(df_nav[cols_exibir].to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        sm = resumo["navegacao"]["sensorial_para_motor"]
        print("\nCanal Biológico Sensorial -> Motor (Combinada Não-Direcionada):")
        print(
            f"  Espacial: S_R={sm['espacial_sucesso_pct']:.1f}% (τ={sm['espacial_stretch_medio']:.2f}) | "
            f"Grau: S_R={sm['grau_sucesso_pct']:.1f}% (τ={sm['grau_stretch_medio']:.2f}) | "
            f"Híbrido: S_R={sm['hibrido_sucesso_pct']:.1f}% (τ={sm['hibrido_stretch_medio']:.2f})"
        )

        print("\n=== ASSORTATIVIDADE MULTIDIMENSIONAL DO CONECTOMA ===")
        print(df_assort.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        assort = resumo["assortatividade"]
        print(
            f"\nCurva k_nn(k): inclinação log-log β_knn = {assort['beta_knn']:.3f} | "
            f"Distância média de aresta = {assort['distancia_fisica_media_aresta']:.3f} "
            f"(Nulo espacial = {assort['distancia_fisica_media_nula']:.3f})"
        )
        print(f"\nVeredito: {resumo['conclusao_textual']}")
