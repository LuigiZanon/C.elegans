"""
Módulo de análise da capacidade de preservação e retenção temporal de informação
no conectoma de C. elegans por meio de quatro pilares estruturais:
  1. Reciprocidade na camada química (topológica, ponderada e por classe funcional);
  2. Maior Componente Fortemente Conexa (LSCC) e Decomposição Bow-Tie;
  3. Modularidade (Q), comunidades estruturais, retenção intra-módulo e NMI;
  4. Densidade exata de ciclos curtos de retroalimentação (C2, C3) e censo de tríades.

Todas as métricas principais são validadas contra modelos nulos direcionados que
preservam exatamente os graus de entrada e saída (k_in, k_out) de cada neurônio.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from config import ORDEM_CLASSES_PADRAO


SETORES_BOWTIE = [
    "IN (Entrada / Feeders)",
    "SCC (Núcleo Recorrente)",
    "OUT (Saída / Sinks)",
    "Outros (Isolados/Tubes)",
]


class InformationPreservationAnalyzer:
    """Quantifica a facilidade de preservação de informação no conectoma."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        membros_rich_club: Optional[Set[str]] = None,
        n_runs_null: int = 30,
        swaps_fator: int = 3,
        louvain_resolution: float = 1.0,
        seed_base: int = 42,
    ) -> None:
        self.graph = graph
        self.n_runs_null = max(1, int(n_runs_null))
        self.swaps_fator = max(1, int(swaps_fator))
        self.louvain_resolution = float(louvain_resolution)
        self.seed_base = int(seed_base)

        self.g_quim_dir = self._projetar_quimica_direcionada()
        self.g_comb_dir = self._projetar_combinada_direcionada()
        self.g_comb_undir = self._projetar_combinada_nao_direcionada()

        if membros_rich_club is None:
            graus_ordenados = sorted(
                self.g_comb_undir.degree(),
                key=lambda item: (-item[1], item[0]),
            )
            self.membros_rich_club: Set[str] = {no for no, _ in graus_ordenados[:6]}
        else:
            self.membros_rich_club = set(membros_rich_club)

        classes_presentes = {
            dados.get("tipo_funcional", "Desconhecido")
            for _, dados in self.g_quim_dir.nodes(data=True)
        }
        extras = sorted(c for c in classes_presentes if c not in ORDEM_CLASSES_PADRAO)
        self.ordem_classes: List[str] = [
            c for c in ORDEM_CLASSES_PADRAO if c in classes_presentes
        ] + extras

    # ------------------------------------------------------------------
    # Projeções base (livres de auto-laços, preservando atributos)
    # ------------------------------------------------------------------
    def _copiar_nos_com_atributos(self, destino: nx.Graph) -> None:
        for no, dados in self.graph.nodes(data=True):
            destino.add_node(
                no,
                tipo_funcional=dados.get("tipo_funcional", "Desconhecido"),
                regiao_soma=dados.get("regiao_soma", "N/A"),
                soma_pos=dados.get("soma_pos", 0.0),
            )

    def _projetar_quimica_direcionada(self) -> nx.DiGraph:
        g = nx.DiGraph()
        self._copiar_nos_com_atributos(g)
        for u, v, chave, dados in self.graph.edges(keys=True, data=True):
            if u == v:
                continue
            tipo = dados.get("synapse_type", chave)
            if tipo != "Quimica":
                continue
            peso = float(dados.get("weight", 1.0))
            if peso <= 0.0:
                continue
            if g.has_edge(u, v):
                g[u][v]["weight"] += peso
            else:
                g.add_edge(u, v, weight=peso)
        return g

    def _projetar_combinada_direcionada(self) -> nx.DiGraph:
        g = nx.DiGraph()
        self._copiar_nos_com_atributos(g)
        pesos_eletricos: Dict[Tuple[str, str], float] = {}

        for u, v, chave, dados in self.graph.edges(keys=True, data=True):
            if u == v:
                continue
            tipo = dados.get("synapse_type", chave)
            peso = float(dados.get("weight", 1.0))
            if peso <= 0.0:
                continue
            if tipo == "Quimica":
                if g.has_edge(u, v):
                    g[u][v]["weight"] += peso
                else:
                    g.add_edge(u, v, weight=peso)
            elif tipo == "Eletrica":
                par = (u, v) if u <= v else (v, u)
                pesos_eletricos[par] = max(pesos_eletricos.get(par, 0.0), peso)

        for (u, v), peso in pesos_eletricos.items():
            for a, b in ((u, v), (v, u)):
                if g.has_edge(a, b):
                    g[a][b]["weight"] += peso
                else:
                    g.add_edge(a, b, weight=peso)
        return g

    def _projetar_combinada_nao_direcionada(self) -> nx.Graph:
        g = nx.Graph()
        self._copiar_nos_com_atributos(g)
        pesos_eletricos: Dict[Tuple[str, str], float] = {}

        for u, v, chave, dados in self.graph.edges(keys=True, data=True):
            if u == v:
                continue
            tipo = dados.get("synapse_type", chave)
            peso = float(dados.get("weight", 1.0))
            if peso <= 0.0:
                continue
            if tipo == "Quimica":
                if g.has_edge(u, v):
                    g[u][v]["weight"] += peso
                else:
                    g.add_edge(u, v, weight=peso)
            elif tipo == "Eletrica":
                par = (u, v) if u <= v else (v, u)
                pesos_eletricos[par] = max(pesos_eletricos.get(par, 0.0), peso)

        for (u, v), peso in pesos_eletricos.items():
            if g.has_edge(u, v):
                g[u][v]["weight"] += peso
            else:
                g.add_edge(u, v, weight=peso)
        return g

    # ------------------------------------------------------------------
    # Utilitários estatísticos, NMI e Modelo Nulo Direcionado
    # ------------------------------------------------------------------
    @staticmethod
    def _z_score(valor_real: float, media_nulo: float, std_nulo: float) -> float:
        if std_nulo > 1e-12:
            return float((valor_real - media_nulo) / std_nulo)
        return 0.0

    @staticmethod
    def _razao_segura(valor_real: float, media_nulo: float) -> float:
        if abs(media_nulo) > 1e-12:
            return float(valor_real / media_nulo)
        return 0.0

    @staticmethod
    def _media_std(valores: Sequence[float]) -> Tuple[float, float]:
        arr = np.asarray(valores, dtype=float)
        if arr.size == 0:
            return 0.0, 0.0
        media = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        return media, std

    @classmethod
    def _converter_para_nativo(cls, obj: Any) -> Any:
        """Converte recursivamente tipos NumPy para tipos nativos serializáveis em JSON."""
        if isinstance(obj, dict):
            return {str(k): cls._converter_para_nativo(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [cls._converter_para_nativo(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            return 0.0 if (math.isnan(val) or math.isinf(val)) else val
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return [cls._converter_para_nativo(v) for v in obj.tolist()]
        if isinstance(obj, float):
            return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
        return obj

    @staticmethod
    def _gerar_grafo_nulo_direcionado(
        arestas: List[Tuple[str, str, float]],
        nos: List[str],
        n_swaps: int,
        rng: np.random.Generator,
    ) -> nx.DiGraph:
        """
        Gera uma rede nula direcionada via Directed Double-Edge Swap com
        embaralhamento aleatório de pesos, preservando exatamente k_in e k_out
        de todos os vértices.
        """
        g_null = nx.DiGraph()
        g_null.add_nodes_from(nos)

        m = len(arestas)
        if m == 0:
            return g_null
        if m == 1:
            u, v, w = arestas[0]
            g_null.add_edge(u, v, weight=float(w))
            return g_null

        origens = [u for u, _, _ in arestas]
        destinos = [v for _, v, _ in arestas]
        pesos = [float(w) for _, _, w in arestas]
        conjunto_arestas = set(zip(origens, destinos))

        for _ in range(n_swaps):
            i, j = rng.integers(0, m, size=2)
            if i == j:
                continue
            u1, v1 = origens[i], destinos[i]
            u2, v2 = origens[j], destinos[j]

            if u1 == v2 or u2 == v1 or u1 == u2 or v1 == v2:
                continue
            if (u1, v2) in conjunto_arestas or (u2, v1) in conjunto_arestas:
                continue

            conjunto_arestas.remove((u1, v1))
            conjunto_arestas.remove((u2, v2))
            conjunto_arestas.add((u1, v2))
            conjunto_arestas.add((u2, v1))

            destinos[i] = v2
            destinos[j] = v1

        pesos_embaralhados = rng.permutation(pesos).tolist()
        g_null.add_weighted_edges_from(zip(origens, destinos, pesos_embaralhados))
        return g_null

    @staticmethod
    def _calcular_nmi(labels_a: List[str], labels_b: List[str]) -> float:
        """Calcula Informação Mútua Normalizada (NMI) em NumPy puro."""
        n = len(labels_a)
        if n == 0 or len(labels_b) != n:
            return 0.0

        _, idx_a = np.unique(labels_a, return_inverse=True)
        _, idx_b = np.unique(labels_b, return_inverse=True)
        n_a = int(idx_a.max()) + 1
        n_b = int(idx_b.max()) + 1

        contingencia = np.zeros((n_a, n_b), dtype=np.float64)
        np.add.at(contingencia, (idx_a, idx_b), 1.0)

        a_marg = contingencia.sum(axis=1)
        b_marg = contingencia.sum(axis=0)

        p_a = a_marg[a_marg > 0] / n
        p_b = b_marg[b_marg > 0] / n
        h_a = float(-np.sum(p_a * np.log(p_a)))
        h_b = float(-np.sum(p_b * np.log(p_b)))

        if (h_a + h_b) <= 1e-12:
            return 0.0

        nz = contingencia > 0
        esperado = np.outer(a_marg, b_marg)
        mi = float(np.sum((contingencia[nz] / n) * np.log((n * contingencia[nz]) / esperado[nz])))
        nmi = (2.0 * mi) / (h_a + h_b)
        return float(min(1.0, max(0.0, nmi)))

    @staticmethod
    def _detectar_comunidades(
        g: nx.Graph,
        weight: Optional[str],
        resolution: float,
        seed: int,
    ) -> List[Set[str]]:
        """Detecta comunidades via Louvain nativo do NetworkX com fallback seguro."""
        if g.number_of_nodes() == 0:
            return []
        if g.number_of_edges() == 0:
            return [{n} for n in g.nodes()]

        try:
            comunidades = nx.community.louvain_communities(
                g, weight=weight, resolution=resolution, seed=seed
            )
            return [set(c) for c in comunidades]
        except Exception:
            try:
                comunidades = nx.community.greedy_modularity_communities(
                    g, weight=weight, resolution=resolution
                )
            except TypeError:
                comunidades = nx.community.greedy_modularity_communities(g, weight=weight)
            return [set(c) for c in comunidades]

    @staticmethod
    def _calcular_modularidade_segura(
        g: nx.Graph,
        comunidades: List[Set[str]],
        weight: Optional[str],
    ) -> float:
        if g.number_of_edges() == 0 or not comunidades:
            return 0.0
        return float(nx.community.modularity(g, comunidades, weight=weight))

    @staticmethod
    def _retencao_intra_modulo(g: nx.Graph, comunidades: List[Set[str]]) -> float:
        mapa_com: Dict[str, int] = {}
        for idx, com in enumerate(comunidades):
            for no in com:
                mapa_com[no] = idx

        peso_total = 0.0
        peso_intra = 0.0
        for u, v, dados in g.edges(data=True):
            w = float(dados.get("weight", 1.0))
            peso_total += w
            if mapa_com.get(u) == mapa_com.get(v):
                peso_intra += w

        if peso_total <= 0.0:
            return 0.0
        return float((peso_intra / peso_total) * 100.0)

    @staticmethod
    def _calcular_reciprocidade_grafo(g: nx.DiGraph) -> Tuple[float, float]:
        """Calcula reciprocidade topológica (R) e ponderada (R_w, Squartini et al., 2013)."""
        if g.number_of_edges() == 0:
            return 0.0, 0.0

        r_topo = float(nx.overall_reciprocity(g))
        soma_total = 0.0
        soma_rec = 0.0
        for u, v, dados in g.edges(data=True):
            w_uv = float(dados.get("weight", 1.0))
            soma_total += w_uv
            if g.has_edge(v, u):
                w_vu = float(g[v][u].get("weight", 1.0))
                soma_rec += min(w_uv, w_vu)

        r_pond = float(soma_rec / soma_total) if soma_total > 0.0 else 0.0
        return r_topo, r_pond

    @staticmethod
    def _contar_ciclos_curtos_matriz(
        g: nx.DiGraph,
        nos_ordenados: List[str],
    ) -> Tuple[int, int, np.ndarray, np.ndarray]:
        """
        Conta exatamente ciclos de comprimento 2 e 3 via álgebra matricial inteira
        (A^2 e A^3 com diagonal nula e dtype=int64).
        """
        n = len(nos_ordenados)
        if n == 0:
            vazio = np.zeros(0, dtype=np.int64)
            return 0, 0, vazio, vazio

        idx_no = {no: i for i, no in enumerate(nos_ordenados)}
        a_mat = np.zeros((n, n), dtype=np.int64)
        for u, v in g.edges():
            if u != v:
                a_mat[idx_no[u], idx_no[v]] = 1
        np.fill_diagonal(a_mat, 0)

        a2 = a_mat @ a_mat
        a3 = a2 @ a_mat

        c2_por_no = np.diag(a2).astype(np.int64)
        c3_por_no = np.diag(a3).astype(np.int64)

        n_c2 = int(np.sum(c2_por_no) // 2)
        n_c3 = int(np.sum(c3_por_no) // 3)
        return n_c2, n_c3, c2_por_no, c3_por_no

    # ------------------------------------------------------------------
    # Pilar 1: Reciprocidade na Camada Química (e comparação Combinada)
    # ------------------------------------------------------------------
    def calcular_reciprocidade(
        self,
        null_r_topo: Sequence[float],
        null_r_pond: Sequence[float],
        null_comb_r_topo: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        r_quim, rw_quim = self._calcular_reciprocidade_grafo(self.g_quim_dir)
        r_comb, rw_comb = self._calcular_reciprocidade_grafo(self.g_comb_dir)

        med_r, std_r = self._media_std(null_r_topo)
        med_rw, std_rw = self._media_std(null_r_pond)
        z_r = self._z_score(r_quim, med_r, std_r)
        z_rw = self._z_score(rw_quim, med_rw, std_rw)

        if null_comb_r_topo:
            med_rc, std_rc = self._media_std(null_comb_r_topo)
            z_rc = self._z_score(r_comb, med_rc, std_rc)
        else:
            med_rc, std_rc, z_rc = med_r, std_r, self._z_score(r_comb, med_r, std_r)

        # Matriz de reciprocidade por classe funcional na camada Química
        cont_tot = pd.DataFrame(0, index=self.ordem_classes, columns=self.ordem_classes, dtype=int)
        cont_rec = pd.DataFrame(0, index=self.ordem_classes, columns=self.ordem_classes, dtype=int)

        pares_rec_intra = 0
        pares_rec_inter = 0
        vistos_pares: Set[Tuple[str, str]] = set()

        for u, v in self.g_quim_dir.edges():
            cls_u = self.g_quim_dir.nodes[u].get("tipo_funcional", "Desconhecido")
            cls_v = self.g_quim_dir.nodes[v].get("tipo_funcional", "Desconhecido")
            if cls_u in cont_tot.index and cls_v in cont_tot.columns:
                cont_tot.loc[cls_u, cls_v] += 1
                if self.g_quim_dir.has_edge(v, u):
                    cont_rec.loc[cls_u, cls_v] += 1
                    par = (u, v) if u < v else (v, u)
                    if par not in vistos_pares:
                        vistos_pares.add(par)
                        if cls_u == cls_v:
                            pares_rec_intra += 1
                        else:
                            pares_rec_inter += 1

        mat_rec = pd.DataFrame(0.0, index=self.ordem_classes, columns=self.ordem_classes, dtype=float)
        for a in self.ordem_classes:
            for b in self.ordem_classes:
                tot = int(cont_tot.loc[a, b])
                mat_rec.loc[a, b] = float(cont_rec.loc[a, b] / tot) if tot > 0 else 0.0

        return {
            "quimica_topologica": r_quim,
            "quimica_ponderada": rw_quim,
            "quimica_nulo_topologica_media": med_r,
            "quimica_nulo_topologica_std": std_r,
            "z_score_reciprocidade": z_r,
            "quimica_nulo_ponderada_media": med_rw,
            "quimica_nulo_ponderada_std": std_rw,
            "z_score_reciprocidade_ponderada": z_rw,
            "combinada_topologica": r_comb,
            "combinada_ponderada": rw_comb,
            "combinada_nulo_topologica_media": med_rc,
            "combinada_nulo_topologica_std": std_rc,
            "z_score_combinada_topologica": z_rc,
            "pares_reciprocos_intra_classe": pares_rec_intra,
            "pares_reciprocos_inter_classe": pares_rec_inter,
            "mat_reciprocidade": mat_rec,
        }

    # ------------------------------------------------------------------
    # Pilar 2: Maior Componente Fortemente Conexa (SCC) e Bow-Tie
    # ------------------------------------------------------------------
    def calcular_scc_e_bowtie(self, null_lscc_sizes: Sequence[float]) -> Dict[str, Any]:
        n_nos = self.g_quim_dir.number_of_nodes()

        sccs_quim = sorted(
            nx.strongly_connected_components(self.g_quim_dir),
            key=len,
            reverse=True,
        )
        lscc_quim: Set[str] = set(sccs_quim[0]) if sccs_quim else set()
        n_sccs_quim_total = len(sccs_quim)
        n_sccs_quim_nt = sum(1 for c in sccs_quim if len(c) >= 2)

        sccs_comb = sorted(
            nx.strongly_connected_components(self.g_comb_dir),
            key=len,
            reverse=True,
        )
        lscc_comb: Set[str] = set(sccs_comb[0]) if sccs_comb else set()

        rc_na_lscc = len(self.membros_rich_club & lscc_quim)
        rc_na_lscc_pct = (
            (rc_na_lscc / len(self.membros_rich_club)) * 100.0
            if self.membros_rich_club
            else 0.0
        )

        # Decomposição Bow-Tie exata na camada Química
        anc: Set[str] = set()
        desc: Set[str] = set()
        if lscc_quim:
            representante = next(iter(lscc_quim))
            anc = nx.ancestors(self.g_quim_dir, representante)
            desc = nx.descendants(self.g_quim_dir, representante)

        setor_por_no: Dict[str, str] = {}
        contagens_bowtie: Dict[str, int] = {s: 0 for s in SETORES_BOWTIE}
        contagem_setor_classe: Dict[str, Dict[str, int]] = {
            s: {c: 0 for c in self.ordem_classes} for s in SETORES_BOWTIE
        }

        for no, dados in self.g_quim_dir.nodes(data=True):
            if no in lscc_quim:
                setor = "SCC (Núcleo Recorrente)"
            elif no in anc:
                setor = "IN (Entrada / Feeders)"
            elif no in desc:
                setor = "OUT (Saída / Sinks)"
            else:
                setor = "Outros (Isolados/Tubes)"

            setor_por_no[no] = setor
            contagens_bowtie[setor] += 1
            cls = dados.get("tipo_funcional", "Desconhecido")
            if cls in contagem_setor_classe[setor]:
                contagem_setor_classe[setor][cls] += 1

        linhas_bowtie = []
        for setor in SETORES_BOWTIE:
            linha: Dict[str, Any] = {"Setor_BowTie": setor}
            for cls in self.ordem_classes:
                linha[cls] = contagem_setor_classe[setor][cls]
            tot = contagens_bowtie[setor]
            linha["Total"] = tot
            linha["Fracao_Pct"] = round((tot / n_nos) * 100.0, 2) if n_nos > 0 else 0.0
            linhas_bowtie.append(linha)
        df_bowtie_classes = pd.DataFrame(linhas_bowtie)

        med_lscc, std_lscc = self._media_std(null_lscc_sizes)
        z_lscc = self._z_score(float(len(lscc_quim)), med_lscc, std_lscc)

        return {
            "quimica_lscc_tamanho": len(lscc_quim),
            "quimica_lscc_fracao_pct": (len(lscc_quim) / n_nos) * 100.0 if n_nos > 0 else 0.0,
            "quimica_n_sccs_total": n_sccs_quim_total,
            "quimica_n_sccs_nao_triviais": n_sccs_quim_nt,
            "quimica_lscc_nulo_media": med_lscc,
            "quimica_lscc_nulo_std": std_lscc,
            "quimica_lscc_z_score": z_lscc,
            "combinada_lscc_tamanho": len(lscc_comb),
            "combinada_lscc_fracao_pct": (len(lscc_comb) / n_nos) * 100.0 if n_nos > 0 else 0.0,
            "rich_club_na_lscc_quimica_qtd": rc_na_lscc,
            "rich_club_total": len(self.membros_rich_club),
            "rich_club_na_lscc_quimica_pct": rc_na_lscc_pct,
            "bowtie_contagens": {
                "SCC (Núcleo Recorrente)": contagens_bowtie["SCC (Núcleo Recorrente)"],
                "IN (Entrada / Feeders)": contagens_bowtie["IN (Entrada / Feeders)"],
                "OUT (Saída / Sinks)": contagens_bowtie["OUT (Saída / Sinks)"],
                "Outros (Isolados/Tubes)": contagens_bowtie["Outros (Isolados/Tubes)"],
            },
            "df_bowtie_classes": df_bowtie_classes,
            "setor_por_no": setor_por_no,
        }

    # ------------------------------------------------------------------
    # Pilar 3: Modularidade (Q), Comunidades, Retenção Local e NMI
    # ------------------------------------------------------------------
    def calcular_modularidade_e_comunidades(
        self,
        null_q_quim: Sequence[float],
    ) -> Dict[str, Any]:
        # 1. Louvain na camada Química (ponderada e não-ponderada)
        coms_quim_w = self._detectar_comunidades(
            self.g_quim_dir, weight="weight", resolution=self.louvain_resolution, seed=self.seed_base
        )
        q_quim_w = self._calcular_modularidade_segura(self.g_quim_dir, coms_quim_w, weight="weight")

        coms_quim_u = self._detectar_comunidades(
            self.g_quim_dir, weight=None, resolution=self.louvain_resolution, seed=self.seed_base
        )
        q_quim_u = self._calcular_modularidade_segura(self.g_quim_dir, coms_quim_u, weight=None)

        # 2. Louvain na camada Combinada não-direcionada (ponderada e não-ponderada)
        coms_comb_w = self._detectar_comunidades(
            self.g_comb_undir, weight="weight", resolution=self.louvain_resolution, seed=self.seed_base
        )
        q_comb_w = self._calcular_modularidade_segura(self.g_comb_undir, coms_comb_w, weight="weight")

        coms_comb_u = self._detectar_comunidades(
            self.g_comb_undir, weight=None, resolution=self.louvain_resolution, seed=self.seed_base
        )
        q_comb_u = self._calcular_modularidade_segura(self.g_comb_undir, coms_comb_u, weight=None)

        # 3. Partição Biológica Funcional (tipo_funcional)
        grupos_func: Dict[str, Set[str]] = {}
        for no, dados in self.g_quim_dir.nodes(data=True):
            cls = str(dados.get("tipo_funcional", "Desconhecido"))
            grupos_func.setdefault(cls, set()).add(no)
        part_func = list(grupos_func.values())
        q_func_quim = self._calcular_modularidade_segura(self.g_quim_dir, part_func, weight="weight")
        q_func_comb = self._calcular_modularidade_segura(self.g_comb_undir, part_func, weight="weight")

        # 4. Partição Anatômica (regiao_soma, filtrando 'N/A')
        grupos_anat: Dict[str, Set[str]] = {}
        nos_anat: List[str] = []
        for no, dados in self.g_quim_dir.nodes(data=True):
            reg = str(dados.get("regiao_soma", "N/A")).strip()
            if reg and reg.upper() != "N/A" and reg.lower() != "nan":
                grupos_anat.setdefault(reg, set()).add(no)
                nos_anat.append(no)
        part_anat = list(grupos_anat.values())
        sub_quim_anat = self.g_quim_dir.subgraph(nos_anat)
        sub_comb_anat = self.g_comb_undir.subgraph(nos_anat)
        q_anat_quim = self._calcular_modularidade_segura(sub_quim_anat, part_anat, weight="weight")
        q_anat_comb = self._calcular_modularidade_segura(sub_comb_anat, part_anat, weight="weight")

        # 5. Retenção sináptica intra-módulo
        ret_quim_pct = self._retencao_intra_modulo(self.g_quim_dir, coms_quim_w)
        ret_comb_pct = self._retencao_intra_modulo(self.g_comb_undir, coms_comb_w)
        ret_func_pct = self._retencao_intra_modulo(self.g_quim_dir, part_func)
        ret_anat_pct = self._retencao_intra_modulo(sub_quim_anat, part_anat)

        # 6. NMI entre comunidades Louvain (Química ponderada) e Biologia
        mapa_louvain: Dict[str, str] = {}
        for idx, com in enumerate(coms_quim_w):
            for no in com:
                mapa_louvain[no] = f"C{idx}"

        nos_todos = list(self.g_quim_dir.nodes())
        labels_louvain_todos = [mapa_louvain[no] for no in nos_todos]
        labels_func_todos = [str(self.g_quim_dir.nodes[no].get("tipo_funcional", "Desconhecido")) for no in nos_todos]
        nmi_func = self._calcular_nmi(labels_louvain_todos, labels_func_todos)

        labels_louvain_anat = [mapa_louvain[no] for no in nos_anat]
        labels_anat = [str(self.g_quim_dir.nodes[no].get("regiao_soma", "N/A")).strip() for no in nos_anat]
        nmi_anat = self._calcular_nmi(labels_louvain_anat, labels_anat)

        # 7. Validação contra modelo nulo direcionado (Química)
        med_q, std_q = self._media_std(null_q_quim)
        z_q = self._z_score(q_quim_w, med_q, std_q)

        df_modularidade = pd.DataFrame([
            {
                "Particao": "Louvain (Química)",
                "Camada": "Quimica",
                "Q_Ponderado": q_quim_w,
                "Q_Nao_Ponderado": q_quim_u,
                "N_Comunidades": len(coms_quim_w),
                "Retencao_Intra_Pct": ret_quim_pct,
                "Nulo_Media": med_q,
                "Nulo_Std": std_q,
                "Z_Score": z_q,
            },
            {
                "Particao": "Louvain (Combinada)",
                "Camada": "Combinada",
                "Q_Ponderado": q_comb_w,
                "Q_Nao_Ponderado": q_comb_u,
                "N_Comunidades": len(coms_comb_w),
                "Retencao_Intra_Pct": ret_comb_pct,
                "Nulo_Media": np.nan,
                "Nulo_Std": np.nan,
                "Z_Score": np.nan,
            },
            {
                "Particao": "Partição Funcional",
                "Camada": "Quimica",
                "Q_Ponderado": q_func_quim,
                "Q_Nao_Ponderado": self._calcular_modularidade_segura(self.g_quim_dir, part_func, weight=None),
                "N_Comunidades": len(part_func),
                "Retencao_Intra_Pct": ret_func_pct,
                "Nulo_Media": np.nan,
                "Nulo_Std": np.nan,
                "Z_Score": np.nan,
            },
            {
                "Particao": "Partição Anatômica",
                "Camada": "Quimica",
                "Q_Ponderado": q_anat_quim,
                "Q_Nao_Ponderado": self._calcular_modularidade_segura(sub_quim_anat, part_anat, weight=None),
                "N_Comunidades": len(part_anat),
                "Retencao_Intra_Pct": ret_anat_pct,
                "Nulo_Media": np.nan,
                "Nulo_Std": np.nan,
                "Z_Score": np.nan,
            },
            {
                "Particao": "Modelo Nulo (Química)",
                "Camada": "Quimica",
                "Q_Ponderado": med_q,
                "Q_Nao_Ponderado": np.nan,
                "N_Comunidades": np.nan,
                "Retencao_Intra_Pct": np.nan,
                "Nulo_Media": med_q,
                "Nulo_Std": std_q,
                "Z_Score": 0.0,
            },
        ])

        return {
            "Q_quimica_louvain_ponderada": q_quim_w,
            "Q_quimica_louvain_nao_ponderada": q_quim_u,
            "Q_quimica_louvain_n_comunidades": len(coms_quim_w),
            "Q_quimica_nulo_media": med_q,
            "Q_quimica_nulo_std": std_q,
            "Q_quimica_z_score": z_q,
            "Q_combinada_louvain_ponderada": q_comb_w,
            "Q_combinada_louvain_nao_ponderada": q_comb_u,
            "Q_combinada_louvain_n_comunidades": len(coms_comb_w),
            "Q_funcional_quimica": q_func_quim,
            "Q_funcional_combinada": q_func_comb,
            "Q_anatomica_quimica": q_anat_quim,
            "Q_anatomica_combinada": q_anat_comb,
            "retencao_intra_modulo_quimica_pct": ret_quim_pct,
            "retencao_intra_modulo_combinada_pct": ret_comb_pct,
            "nmi_louvain_vs_funcional": nmi_func,
            "nmi_louvain_vs_anatomico": nmi_anat,
            "df_modularidade": df_modularidade,
        }

    # ------------------------------------------------------------------
    # Pilar 4: Densidade de Ciclos Curtos (C2, C3) e Censo de Tríades
    # ------------------------------------------------------------------
    def calcular_ciclos_retroalimentacao(
        self,
        nos_ordenados: List[str],
        null_c2: Sequence[float],
        null_c3: Sequence[float],
        null_triades: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        n = len(nos_ordenados)
        n_c2, n_c3, c2_por_no, c3_por_no = self._contar_ciclos_curtos_matriz(
            self.g_quim_dir, nos_ordenados
        )

        rho_c2 = float((2.0 * n_c2) / (n * (n - 1))) if n >= 2 else 0.0
        rho_c3 = float((3.0 * n_c3) / (n * (n - 1) * (n - 2))) if n >= 3 else 0.0
        n_triplas = float((n * (n - 1) * (n - 2)) / 6.0) if n >= 3 else 1.0

        # Participação nodal por classe funcional e Rich-Club vs. Periferia
        df_nos_ciclos = pd.DataFrame({
            "Neuronio": nos_ordenados,
            "Tipo_Funcional": [
                self.g_quim_dir.nodes[no].get("tipo_funcional", "Desconhecido")
                for no in nos_ordenados
            ],
            "Rich_Club": [no in self.membros_rich_club for no in nos_ordenados],
            "C2_No": c2_por_no,
            "C3_No": c3_por_no,
        })

        linhas_classe = []
        for cls in self.ordem_classes:
            sub = df_nos_ciclos[df_nos_ciclos["Tipo_Funcional"] == cls]
            linhas_classe.append({
                "Grupo": cls,
                "Categoria": "Classe Funcional",
                "N_Neuronios": int(len(sub)),
                "C2_Media": float(sub["C2_No"].mean()) if len(sub) > 0 else 0.0,
                "C2_Soma": int(sub["C2_No"].sum()) if len(sub) > 0 else 0,
                "C3_Media": float(sub["C3_No"].mean()) if len(sub) > 0 else 0.0,
                "C3_Soma": int(sub["C3_No"].sum()) if len(sub) > 0 else 0,
            })

        sub_rc = df_nos_ciclos[df_nos_ciclos["Rich_Club"]]
        sub_perif = df_nos_ciclos[~df_nos_ciclos["Rich_Club"]]
        media_c2_rc = float(sub_rc["C2_No"].mean()) if len(sub_rc) > 0 else 0.0
        media_c2_perif = float(sub_perif["C2_No"].mean()) if len(sub_perif) > 0 else 0.0
        media_c3_rc = float(sub_rc["C3_No"].mean()) if len(sub_rc) > 0 else 0.0
        media_c3_perif = float(sub_perif["C3_No"].mean()) if len(sub_perif) > 0 else 0.0

        linhas_classe.append({
            "Grupo": "Núcleo Rich-Club",
            "Categoria": "Rich-Club vs. Periferia",
            "N_Neuronios": int(len(sub_rc)),
            "C2_Media": media_c2_rc,
            "C2_Soma": int(sub_rc["C2_No"].sum()) if len(sub_rc) > 0 else 0,
            "C3_Media": media_c3_rc,
            "C3_Soma": int(sub_rc["C3_No"].sum()) if len(sub_rc) > 0 else 0,
        })
        linhas_classe.append({
            "Grupo": "Periferia (Demais)",
            "Categoria": "Rich-Club vs. Periferia",
            "N_Neuronios": int(len(sub_perif)),
            "C2_Media": media_c2_perif,
            "C2_Soma": int(sub_perif["C2_No"].sum()) if len(sub_perif) > 0 else 0,
            "C3_Media": media_c3_perif,
            "C3_Soma": int(sub_perif["C3_No"].sum()) if len(sub_perif) > 0 else 0,
        })
        df_ciclos_por_classe = pd.DataFrame(linhas_classe)

        # Censo de tríades direcionadas
        censo = nx.triadic_census(self.g_quim_dir)
        t_030t = int(censo.get("030T", 0))
        t_030c = int(censo.get("030C", 0))
        t_120c = int(censo.get("120C", 0))
        t_210 = int(censo.get("210", 0))
        t_300 = int(censo.get("300", 0))

        total_fechadas = sum(
            int(censo.get(k, 0)) for k in ("030T", "030C", "120D", "120U", "120C", "210", "300")
        )
        total_recorrentes = t_030c + t_120c + t_210 + t_300
        fracao_recorrentes_pct = (
            (total_recorrentes / total_fechadas) * 100.0 if total_fechadas > 0 else 0.0
        )
        razao_030c_030t = float(t_030c / max(t_030t, 1))

        # Validação contra modelo nulo direcionado
        med_c2, std_c2 = self._media_std(null_c2)
        z_c2 = self._z_score(float(n_c2), med_c2, std_c2)

        med_c3, std_c3 = self._media_std(null_c3)
        z_c3 = self._z_score(float(n_c3), med_c3, std_c3)

        especificacoes = [
            ("2-Ciclos (C2)", "Retroalimentação Direta", n_c2, rho_c2, med_c2, std_c2, z_c2),
            ("3-Ciclos (C3)", "Retroalimentação Cíclica", n_c3, rho_c3, med_c3, std_c3, z_c3),
        ]

        triades_meta = [
            ("030T", "030T (Feedforward)", "Feedforward Acíclico", t_030t),
            ("030C", "030C (3-Ciclo Puro)", "Retroalimentação Cíclica", t_030c),
            ("120C", "120C (Ciclo + Díade)", "Retroalimentação Cíclica", t_120c),
            ("210", "210 (2 Díades Recíprocas)", "Retroalimentação Intensa", t_210),
            ("300", "300 (Clique Bidirecional)", "Retroalimentação Máxima", t_300),
        ]

        z_triades: Dict[str, float] = {}
        med_triades: Dict[str, float] = {}
        std_triades: Dict[str, float] = {}
        for chave, rotulo, tipo_circ, cont_real in triades_meta:
            med_t, std_t = self._media_std(null_triades.get(chave, []))
            z_t = self._z_score(float(cont_real), med_t, std_t)
            z_triades[chave] = z_t
            med_triades[chave] = med_t
            std_triades[chave] = std_t
            dens_t = float(cont_real / n_triplas) if n_triplas > 0 else 0.0
            especificacoes.append((rotulo, tipo_circ, cont_real, dens_t, med_t, std_t, z_t))

        linhas_motifs = []
        for estrutura, tipo_circ, cont_real, dens_real, med_n, std_n, z_val in especificacoes:
            linhas_motifs.append({
                "Estrutura": estrutura,
                "Tipo_Circuito": tipo_circ,
                "Contagem_Real": int(cont_real),
                "Densidade_Real": float(dens_real),
                "Nulo_Media": float(med_n),
                "Nulo_Std": float(std_n),
                "Razao_Real_Nulo": self._razao_segura(float(cont_real), med_n),
                "Z_Score": float(z_val),
            })
        df_ciclos_motifs = pd.DataFrame(linhas_motifs)

        return {
            "n_ciclos_2": n_c2,
            "densidade_ciclos_2": rho_c2,
            "nulo_ciclos_2_media": med_c2,
            "nulo_ciclos_2_std": std_c2,
            "z_score_ciclos_2": z_c2,
            "n_ciclos_3": n_c3,
            "densidade_ciclos_3": rho_c3,
            "nulo_ciclos_3_media": med_c3,
            "nulo_ciclos_3_std": std_c3,
            "z_score_ciclos_3": z_c3,
            "triade_030T_feedforward": t_030t,
            "triade_030C_feedback": t_030c,
            "triade_120C": t_120c,
            "triade_210": t_210,
            "triade_300": t_300,
            "total_triades_fechadas": total_fechadas,
            "total_triades_recorrentes": total_recorrentes,
            "fracao_triades_recorrentes_pct": fracao_recorrentes_pct,
            "razao_030C_030T": razao_030c_030t,
            "media_c2_rich_club": media_c2_rc,
            "media_c2_periferia": media_c2_perif,
            "media_c3_rich_club": media_c3_rc,
            "media_c3_periferia": media_c3_perif,
            "z_triades": z_triades,
            "med_triades": med_triades,
            "std_triades": std_triades,
            "df_ciclos_motifs": df_ciclos_motifs,
            "df_ciclos_por_classe": df_ciclos_por_classe,
        }

    # ------------------------------------------------------------------
    # Execução Unificada, Veredito e Relatório no Console
    # ------------------------------------------------------------------
    def _construir_conclusao_textual(
        self,
        rec: Dict[str, Any],
        scc: Dict[str, Any],
        mod: Dict[str, Any],
        cic: Dict[str, Any],
    ) -> str:
        evidencias_positivas = 0
        if rec["z_score_reciprocidade"] > 2.0:
            evidencias_positivas += 1
        if scc["quimica_lscc_fracao_pct"] >= 60.0:
            evidencias_positivas += 1
        if mod["Q_quimica_z_score"] > 2.0 and mod["Q_quimica_louvain_ponderada"] >= 0.30:
            evidencias_positivas += 1
        if cic["z_score_ciclos_2"] > 2.0 or cic["z_score_ciclos_3"] > 2.0:
            evidencias_positivas += 1

        if evidencias_positivas >= 3:
            veredito = (
                "SIM — a preservação e retenção temporal de informação é estruturalmente "
                "facilitada no conectoma do C. elegans."
            )
        elif evidencias_positivas == 2:
            veredito = (
                "PARCIALMENTE SIM — o conectoma oferece suporte moderado à preservação de "
                "informação, equilibrando circuitos recorrentes e vias feedforward."
            )
        else:
            veredito = (
                "LIMITADA — a rede química exibe predomínio de fluxo acíclico/feedforward "
                "com suporte recorrente restrito frente ao modelo nulo."
            )

        detalhes = (
            f" (1) Reciprocidade química topológica R={rec['quimica_topologica']*100:.1f}% "
            f"(Z={rec['z_score_reciprocidade']:+.2f} vs. nulo) e ponderada R_w={rec['quimica_ponderada']*100:.1f}% "
            f"(Z={rec['z_score_reciprocidade_ponderada']:+.2f}), saltando para R={rec['combinada_topologica']*100:.1f}% "
            f"com junções elétricas; "
            f"(2) Grande núcleo fortemente conexo (LSCC química com {scc['quimica_lscc_tamanho']} neurônios, "
            f"{scc['quimica_lscc_fracao_pct']:.1f}% da rede, abrigando {scc['rich_club_na_lscc_quimica_pct']:.1f}% "
            f"do Rich-Club, e {scc['combinada_lscc_fracao_pct']:.1f}% na camada combinada); "
            f"(3) Alta modularidade direcionada ponderada Q={mod['Q_quimica_louvain_ponderada']:.3f} "
            f"(Z={mod['Q_quimica_z_score']:+.2f}), retendo {mod['retencao_intra_modulo_quimica_pct']:.1f}% do "
            f"volume sináptico químico dentro dos módulos; "
            f"(4) Enriquecimento de laços curtos de retroalimentação (C2={cic['n_ciclos_2']}, "
            f"Z={cic['z_score_ciclos_2']:+.2f}; C3={cic['n_ciclos_3']}, Z={cic['z_score_ciclos_3']:+.2f}; "
            f"tríades 210 Z={cic['z_triades']['210']:+.2f} e 300 Z={cic['z_triades']['300']:+.2f}), "
            f"com participação média em 3-ciclos de {cic['media_c3_rich_club']:.1f} no Rich-Club "
            f"vs. {cic['media_c3_periferia']:.1f} na periferia."
        )
        return veredito + detalhes

    def executar_analise_completa(self) -> Dict[str, Any]:
        """
        Executa os 4 pilares de preservação de informação compartilhando um único
        conjunto determinístico de redes nulas direcionadas da camada Química.
        """
        nos_ordenados = sorted(self.g_quim_dir.nodes())
        arestas_quim = [
            (u, v, float(d.get("weight", 1.0)))
            for u, v, d in self.g_quim_dir.edges(data=True)
        ]
        m_quim = len(arestas_quim)
        n_swaps_quim = max(1500, self.swaps_fator * m_quim)

        arestas_comb = [
            (u, v, float(d.get("weight", 1.0)))
            for u, v, d in self.g_comb_dir.edges(data=True)
        ]
        n_swaps_comb = max(1500, self.swaps_fator * len(arestas_comb))

        null_r_topo: List[float] = []
        null_r_pond: List[float] = []
        null_comb_r_topo: List[float] = []
        null_lscc_sizes: List[float] = []
        null_q_quim: List[float] = []
        null_c2: List[float] = []
        null_c3: List[float] = []
        chaves_triades = ("030T", "030C", "120C", "210", "300")
        null_triades: Dict[str, List[float]] = {k: [] for k in chaves_triades}

        for idx in range(self.n_runs_null):
            seed_i = self.seed_base + idx
            rng_i = np.random.default_rng(seed_i)
            g_null = self._gerar_grafo_nulo_direcionado(
                arestas_quim, nos_ordenados, n_swaps_quim, rng_i
            )

            # Pilar 1 (nulo Química)
            r_n, rw_n = self._calcular_reciprocidade_grafo(g_null)
            null_r_topo.append(r_n)
            null_r_pond.append(rw_n)

            # Nulo Combinada apenas para referência de reciprocidade topológica
            rng_c = np.random.default_rng(seed_i + 10000)
            g_null_comb = self._gerar_grafo_nulo_direcionado(
                arestas_comb, nos_ordenados, n_swaps_comb, rng_c
            )
            rc_n, _ = self._calcular_reciprocidade_grafo(g_null_comb)
            null_comb_r_topo.append(rc_n)

            # Pilar 2 (nulo LSCC Química)
            sccs_n = list(nx.strongly_connected_components(g_null))
            lscc_n = max((len(c) for c in sccs_n), default=0)
            null_lscc_sizes.append(float(lscc_n))

            # Pilar 3 (nulo Modularidade Louvain Química)
            coms_n = self._detectar_comunidades(
                g_null, weight="weight", resolution=self.louvain_resolution, seed=seed_i
            )
            q_n = self._calcular_modularidade_segura(g_null, coms_n, weight="weight")
            null_q_quim.append(q_n)

            # Pilar 4 (nulo Ciclos C2, C3 e Tríades Química)
            c2_n, c3_n, _, _ = self._contar_ciclos_curtos_matriz(g_null, nos_ordenados)
            null_c2.append(float(c2_n))
            null_c3.append(float(c3_n))
            censo_n = nx.triadic_census(g_null)
            for chave in chaves_triades:
                null_triades[chave].append(float(censo_n.get(chave, 0)))

        res_rec = self.calcular_reciprocidade(null_r_topo, null_r_pond, null_comb_r_topo)
        res_scc = self.calcular_scc_e_bowtie(null_lscc_sizes)
        res_mod = self.calcular_modularidade_e_comunidades(null_q_quim)
        res_cic = self.calcular_ciclos_retroalimentacao(
            nos_ordenados, null_c2, null_c3, null_triades
        )

        conclusao = self._construir_conclusao_textual(res_rec, res_scc, res_mod, res_cic)

        # Tabela resumo unificada (schema 4.1)
        linhas_resumo = [
            {
                "Pilar": "1. Reciprocidade",
                "Metrica": "Reciprocidade Topológica (R)",
                "Camada": "Quimica",
                "Valor_Real": res_rec["quimica_topologica"],
                "Nulo_Media": res_rec["quimica_nulo_topologica_media"],
                "Nulo_Std": res_rec["quimica_nulo_topologica_std"],
                "Razao_Real_Nulo": self._razao_segura(
                    res_rec["quimica_topologica"], res_rec["quimica_nulo_topologica_media"]
                ),
                "Z_Score": res_rec["z_score_reciprocidade"],
            },
            {
                "Pilar": "1. Reciprocidade",
                "Metrica": "Reciprocidade Ponderada (R_w)",
                "Camada": "Quimica",
                "Valor_Real": res_rec["quimica_ponderada"],
                "Nulo_Media": res_rec["quimica_nulo_ponderada_media"],
                "Nulo_Std": res_rec["quimica_nulo_ponderada_std"],
                "Razao_Real_Nulo": self._razao_segura(
                    res_rec["quimica_ponderada"], res_rec["quimica_nulo_ponderada_media"]
                ),
                "Z_Score": res_rec["z_score_reciprocidade_ponderada"],
            },
            {
                "Pilar": "1. Reciprocidade",
                "Metrica": "Reciprocidade Topológica (R)",
                "Camada": "Combinada",
                "Valor_Real": res_rec["combinada_topologica"],
                "Nulo_Media": res_rec["combinada_nulo_topologica_media"],
                "Nulo_Std": res_rec["combinada_nulo_topologica_std"],
                "Razao_Real_Nulo": self._razao_segura(
                    res_rec["combinada_topologica"], res_rec["combinada_nulo_topologica_media"]
                ),
                "Z_Score": res_rec["z_score_combinada_topologica"],
            },
            {
                "Pilar": "1. Reciprocidade",
                "Metrica": "Reciprocidade Ponderada (R_w)",
                "Camada": "Combinada",
                "Valor_Real": res_rec["combinada_ponderada"],
                "Nulo_Media": np.nan,
                "Nulo_Std": np.nan,
                "Razao_Real_Nulo": np.nan,
                "Z_Score": np.nan,
            },
            {
                "Pilar": "2. SCC e Bow-Tie",
                "Metrica": "Tamanho LSCC (|V_LSCC|)",
                "Camada": "Quimica",
                "Valor_Real": float(res_scc["quimica_lscc_tamanho"]),
                "Nulo_Media": res_scc["quimica_lscc_nulo_media"],
                "Nulo_Std": res_scc["quimica_lscc_nulo_std"],
                "Razao_Real_Nulo": self._razao_segura(
                    float(res_scc["quimica_lscc_tamanho"]), res_scc["quimica_lscc_nulo_media"]
                ),
                "Z_Score": res_scc["quimica_lscc_z_score"],
            },
            {
                "Pilar": "2. SCC e Bow-Tie",
                "Metrica": "Tamanho LSCC (|V_LSCC|)",
                "Camada": "Combinada",
                "Valor_Real": float(res_scc["combinada_lscc_tamanho"]),
                "Nulo_Media": np.nan,
                "Nulo_Std": np.nan,
                "Razao_Real_Nulo": np.nan,
                "Z_Score": np.nan,
            },
            {
                "Pilar": "3. Modularidade",
                "Metrica": "Modularidade Louvain Ponderada (Q)",
                "Camada": "Quimica",
                "Valor_Real": res_mod["Q_quimica_louvain_ponderada"],
                "Nulo_Media": res_mod["Q_quimica_nulo_media"],
                "Nulo_Std": res_mod["Q_quimica_nulo_std"],
                "Razao_Real_Nulo": self._razao_segura(
                    res_mod["Q_quimica_louvain_ponderada"], res_mod["Q_quimica_nulo_media"]
                ),
                "Z_Score": res_mod["Q_quimica_z_score"],
            },
            {
                "Pilar": "3. Modularidade",
                "Metrica": "Modularidade Louvain Ponderada (Q)",
                "Camada": "Combinada",
                "Valor_Real": res_mod["Q_combinada_louvain_ponderada"],
                "Nulo_Media": np.nan,
                "Nulo_Std": np.nan,
                "Razao_Real_Nulo": np.nan,
                "Z_Score": np.nan,
            },
            {
                "Pilar": "3. Modularidade",
                "Metrica": "Retenção Intra-Módulo (%)",
                "Camada": "Quimica",
                "Valor_Real": res_mod["retencao_intra_modulo_quimica_pct"],
                "Nulo_Media": np.nan,
                "Nulo_Std": np.nan,
                "Razao_Real_Nulo": np.nan,
                "Z_Score": np.nan,
            },
            {
                "Pilar": "4. Ciclos Curtos",
                "Metrica": "2-Ciclos Direcionados (N_C2)",
                "Camada": "Quimica",
                "Valor_Real": float(res_cic["n_ciclos_2"]),
                "Nulo_Media": res_cic["nulo_ciclos_2_media"],
                "Nulo_Std": res_cic["nulo_ciclos_2_std"],
                "Razao_Real_Nulo": self._razao_segura(
                    float(res_cic["n_ciclos_2"]), res_cic["nulo_ciclos_2_media"]
                ),
                "Z_Score": res_cic["z_score_ciclos_2"],
            },
            {
                "Pilar": "4. Ciclos Curtos",
                "Metrica": "3-Ciclos Direcionados (N_C3)",
                "Camada": "Quimica",
                "Valor_Real": float(res_cic["n_ciclos_3"]),
                "Nulo_Media": res_cic["nulo_ciclos_3_media"],
                "Nulo_Std": res_cic["nulo_ciclos_3_std"],
                "Razao_Real_Nulo": self._razao_segura(
                    float(res_cic["n_ciclos_3"]), res_cic["nulo_ciclos_3_media"]
                ),
                "Z_Score": res_cic["z_score_ciclos_3"],
            },
            {
                "Pilar": "4. Ciclos Curtos",
                "Metrica": "Tríades 210 (2 Díades Recíprocas)",
                "Camada": "Quimica",
                "Valor_Real": float(res_cic["triade_210"]),
                "Nulo_Media": res_cic["med_triades"]["210"],
                "Nulo_Std": res_cic["std_triades"]["210"],
                "Razao_Real_Nulo": self._razao_segura(
                    float(res_cic["triade_210"]), res_cic["med_triades"]["210"]
                ),
                "Z_Score": res_cic["z_triades"]["210"],
            },
            {
                "Pilar": "4. Ciclos Curtos",
                "Metrica": "Tríades 300 (Clique Bidirecional)",
                "Camada": "Quimica",
                "Valor_Real": float(res_cic["triade_300"]),
                "Nulo_Media": res_cic["med_triades"]["300"],
                "Nulo_Std": res_cic["std_triades"]["300"],
                "Razao_Real_Nulo": self._razao_segura(
                    float(res_cic["triade_300"]), res_cic["med_triades"]["300"]
                ),
                "Z_Score": res_cic["z_triades"]["300"],
            },
        ]
        df_resumo = pd.DataFrame(linhas_resumo)

        resumo_json = self._converter_para_nativo({
            "metadados": {
                "N": self.g_quim_dir.number_of_nodes(),
                "M_quimica": self.g_quim_dir.number_of_edges(),
                "M_combinada_dir": self.g_comb_dir.number_of_edges(),
                "n_runs_null": self.n_runs_null,
            },
            "reciprocidade": {
                "quimica_topologica": res_rec["quimica_topologica"],
                "quimica_ponderada": res_rec["quimica_ponderada"],
                "quimica_nulo_topologica_media": res_rec["quimica_nulo_topologica_media"],
                "quimica_nulo_topologica_std": res_rec["quimica_nulo_topologica_std"],
                "z_score_reciprocidade": res_rec["z_score_reciprocidade"],
                "quimica_nulo_ponderada_media": res_rec["quimica_nulo_ponderada_media"],
                "quimica_nulo_ponderada_std": res_rec["quimica_nulo_ponderada_std"],
                "z_score_reciprocidade_ponderada": res_rec["z_score_reciprocidade_ponderada"],
                "combinada_topologica": res_rec["combinada_topologica"],
                "combinada_ponderada": res_rec["combinada_ponderada"],
                "combinada_nulo_topologica_media": res_rec["combinada_nulo_topologica_media"],
                "combinada_nulo_topologica_std": res_rec["combinada_nulo_topologica_std"],
                "z_score_combinada_topologica": res_rec["z_score_combinada_topologica"],
                "pares_reciprocos_intra_classe": res_rec["pares_reciprocos_intra_classe"],
                "pares_reciprocos_inter_classe": res_rec["pares_reciprocos_inter_classe"],
            },
            "scc_bowtie": {
                "quimica_lscc_tamanho": res_scc["quimica_lscc_tamanho"],
                "quimica_lscc_fracao_pct": res_scc["quimica_lscc_fracao_pct"],
                "quimica_n_sccs_nao_triviais": res_scc["quimica_n_sccs_nao_triviais"],
                "quimica_lscc_nulo_media": res_scc["quimica_lscc_nulo_media"],
                "quimica_lscc_nulo_std": res_scc["quimica_lscc_nulo_std"],
                "quimica_lscc_z_score": res_scc["quimica_lscc_z_score"],
                "combinada_lscc_tamanho": res_scc["combinada_lscc_tamanho"],
                "combinada_lscc_fracao_pct": res_scc["combinada_lscc_fracao_pct"],
                "rich_club_na_lscc_quimica_qtd": res_scc["rich_club_na_lscc_quimica_qtd"],
                "rich_club_total": res_scc["rich_club_total"],
                "rich_club_na_lscc_quimica_pct": res_scc["rich_club_na_lscc_quimica_pct"],
                "bowtie_contagens": res_scc["bowtie_contagens"],
            },
            "modularidade": {
                "Q_quimica_louvain_ponderada": res_mod["Q_quimica_louvain_ponderada"],
                "Q_quimica_louvain_n_comunidades": res_mod["Q_quimica_louvain_n_comunidades"],
                "Q_quimica_nulo_media": res_mod["Q_quimica_nulo_media"],
                "Q_quimica_nulo_std": res_mod["Q_quimica_nulo_std"],
                "Q_quimica_z_score": res_mod["Q_quimica_z_score"],
                "Q_combinada_louvain_ponderada": res_mod["Q_combinada_louvain_ponderada"],
                "Q_combinada_louvain_n_comunidades": res_mod["Q_combinada_louvain_n_comunidades"],
                "Q_funcional_quimica": res_mod["Q_funcional_quimica"],
                "Q_anatomica_quimica": res_mod["Q_anatomica_quimica"],
                "retencao_intra_modulo_quimica_pct": res_mod["retencao_intra_modulo_quimica_pct"],
                "retencao_intra_modulo_combinada_pct": res_mod["retencao_intra_modulo_combinada_pct"],
                "nmi_louvain_vs_funcional": res_mod["nmi_louvain_vs_funcional"],
                "nmi_louvain_vs_anatomico": res_mod["nmi_louvain_vs_anatomico"],
            },
            "ciclos_curtos": {
                "n_ciclos_2": res_cic["n_ciclos_2"],
                "densidade_ciclos_2": res_cic["densidade_ciclos_2"],
                "z_score_ciclos_2": res_cic["z_score_ciclos_2"],
                "n_ciclos_3": res_cic["n_ciclos_3"],
                "densidade_ciclos_3": res_cic["densidade_ciclos_3"],
                "z_score_ciclos_3": res_cic["z_score_ciclos_3"],
                "triade_030T_feedforward": res_cic["triade_030T_feedforward"],
                "triade_030C_feedback": res_cic["triade_030C_feedback"],
                "triade_120C": res_cic["triade_120C"],
                "triade_210": res_cic["triade_210"],
                "triade_300": res_cic["triade_300"],
                "fracao_triades_recorrentes_pct": res_cic["fracao_triades_recorrentes_pct"],
                "razao_030C_030T": res_cic["razao_030C_030T"],
                "media_c2_rich_club": res_cic["media_c2_rich_club"],
                "media_c2_periferia": res_cic["media_c2_periferia"],
                "media_c3_rich_club": res_cic["media_c3_rich_club"],
                "media_c3_periferia": res_cic["media_c3_periferia"],
            },
            "conclusao_textual": conclusao,
        })

        return {
            "df_resumo": df_resumo,
            "mat_reciprocidade": res_rec["mat_reciprocidade"],
            "df_bowtie_classes": res_scc["df_bowtie_classes"],
            "df_modularidade": res_mod["df_modularidade"],
            "df_ciclos_motifs": res_cic["df_ciclos_motifs"],
            "df_ciclos_por_classe": res_cic["df_ciclos_por_classe"],
            "resumo_json": resumo_json,
        }

    def exibir_relatorio_console(self, resultado: Dict[str, Any]) -> None:
        """Imprime relatório estruturado dos 4 pilares de preservação de informação."""
        res = resultado["resumo_json"]
        rec = res["reciprocidade"]
        scc = res["scc_bowtie"]
        mod = res["modularidade"]
        cic = res["ciclos_curtos"]

        print("\n" + "=" * 78)
        print("PRESERVAÇÃO E RETENÇÃO DE INFORMAÇÃO NO CONECTOMA (C. ELEGANS)")
        print("=" * 78)
        print(
            f"Grafo base: N={res['metadados']['N']} neurônios | "
            f"M_Química={res['metadados']['M_quimica']} | "
            f"M_Combinada_Dir={res['metadados']['M_combinada_dir']} | "
            f"Modelos nulos direcionados: {res['metadados']['n_runs_null']}"
        )

        print("\n[Pilar 1] Reciprocidade na Camada Química vs. Combinada:")
        print(
            f"  • Química Topológica (R):  {rec['quimica_topologica']*100:5.2f}% "
            f"(Nulo: {rec['quimica_nulo_topologica_media']*100:5.2f}% ± {rec['quimica_nulo_topologica_std']*100:4.2f}%, "
            f"Z = {rec['z_score_reciprocidade']:+.2f})"
        )
        print(
            f"  • Química Ponderada (R_w): {rec['quimica_ponderada']*100:5.2f}% "
            f"(Nulo: {rec['quimica_nulo_ponderada_media']*100:5.2f}% ± {rec['quimica_nulo_ponderada_std']*100:4.2f}%, "
            f"Z = {rec['z_score_reciprocidade_ponderada']:+.2f})"
        )
        print(
            f"  • Combinada (Química + Elétrica): R={rec['combinada_topologica']*100:5.2f}% | "
            f"R_w={rec['combinada_ponderada']*100:5.2f}%"
        )
        print(
            f"  • Pares recíprocos químicos: {rec['pares_reciprocos_intra_classe']} intra-classe | "
            f"{rec['pares_reciprocos_inter_classe']} inter-classe"
        )

        print("\n[Pilar 2] Maior Componente Fortemente Conexa (LSCC) e Decomposição Bow-Tie:")
        print(
            f"  • LSCC Química:   {scc['quimica_lscc_tamanho']} neurônios "
            f"({scc['quimica_lscc_fracao_pct']:.1f}% | Nulo: {scc['quimica_lscc_nulo_media']:.1f}, "
            f"Z = {scc['quimica_lscc_z_score']:+.2f}) | SCCs não-triviais: {scc['quimica_n_sccs_nao_triviais']}"
        )
        print(
            f"  • LSCC Combinada: {scc['combinada_lscc_tamanho']} neurônios "
            f"({scc['combinada_lscc_fracao_pct']:.1f}%) | "
            f"Rich-Club na LSCC Química: {scc['rich_club_na_lscc_quimica_qtd']}/{scc['rich_club_total']} "
            f"({scc['rich_club_na_lscc_quimica_pct']:.1f}%)"
        )
        print(resultado["df_bowtie_classes"].to_string(index=False))

        print("\n[Pilar 3] Modularidade (Q), Retenção Intra-Módulo e Alinhamento Biológico:")
        print(
            f"  • Louvain Química (Ponderada): Q={mod['Q_quimica_louvain_ponderada']:.4f} "
            f"({mod['Q_quimica_louvain_n_comunidades']} módulos | "
            f"Nulo: {mod['Q_quimica_nulo_media']:.4f}, Z = {mod['Q_quimica_z_score']:+.2f}) | "
            f"Retenção intra-módulo: {mod['retencao_intra_modulo_quimica_pct']:.1f}%"
        )
        print(
            f"  • Louvain Combinada (Ponderada): Q={mod['Q_combinada_louvain_ponderada']:.4f} "
            f"({mod['Q_combinada_louvain_n_comunidades']} módulos) | "
            f"Retenção intra-módulo: {mod['retencao_intra_modulo_combinada_pct']:.1f}%"
        )
        print(
            f"  • Q Funcional (Química)={mod['Q_funcional_quimica']:.4f} | "
            f"Q Anatômica (Química)={mod['Q_anatomica_quimica']:.4f} | "
            f"NMI(Louvain, Funcional)={mod['nmi_louvain_vs_funcional']:.3f} | "
            f"NMI(Louvain, Anatômico)={mod['nmi_louvain_vs_anatomico']:.3f}"
        )

        print("\n[Pilar 4] Ciclos Curtos de Retroalimentação (C2, C3) e Censo de Tríades:")
        print(resultado["df_ciclos_motifs"].to_string(index=False))
        print(
            f"  • Fração de tríades fechadas recorrentes: {cic['fracao_triades_recorrentes_pct']:.1f}% | "
            f"Razão 030C/030T: {cic['razao_030C_030T']:.3f}"
        )
        print(
            f"  • Participação média em 3-ciclos: Rich-Club = {cic['media_c3_rich_club']:.1f} "
            f"vs. Periferia = {cic['media_c3_periferia']:.1f}"
        )

        print("\n[Veredito Final — Preservação de Informação]")
        print(f"  {res['conclusao_textual']}")
        print("=" * 78)
