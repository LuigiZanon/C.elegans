"""
Módulo de análise estatística, fluxo entre classes, centralidades, cabeamento e Rich-Club Ponderado.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from config import ORDEM_CLASSES_PADRAO
from graph_utils import gerar_nulo_direcionado, projetar_grafo


JUSTIFICATIVAS_BIOLOGICAS_TOP: Dict[str, str] = {
    "AVAL": "Interneurônio de comando primário de marcha retrógrada e fuga; principal hub integrador do conectoma.",
    "AVAR": "Homólogo contralateral direito de AVAL; maior proximidade global e sinergia bilateral na reversão de marcha.",
    "AVBR": "Interneurônio de comando da locomoção anterógrada; forma com AVAL/AVAR o oscilador central de movimento.",
    "PVCR": "Interneurônio de comando posterior condutor de locomoção anterógrada após estímulos na cauda.",
    "AVBL": "Par contralateral esquerdo de AVBR na coordenação da locomoção anterógrada contínua.",
    "PVCL": "Par contralateral esquerdo de PVCR no reflexo de avanço por toque posterior.",
    "VD01": "Motoneurônio ventral participante da transmissão do padrão ondulatório motor.",
    "VA08": "Motoneurônio do cordão ventral acoplado eletricamente aos comandos de reversão.",
}


class ConnectomeAnalyzer:
    """Realiza análises de fluxo entre classes, centralidades, cabeamento e Rich-Club Ponderado."""

    def __init__(self, graph: nx.MultiDiGraph, agrupar_comando: bool = False):
        self.graph = graph
        self.agrupar_comando = agrupar_comando
        self.df_fluxo: pd.DataFrame = self._extrair_dataframe_fluxo()
        self._cache_centralidades: Dict[str, pd.DataFrame] = {}

    def _distancia_soma(self, u: str, v: str) -> float:
        """Distância física longitudinal |soma_pos(u) - soma_pos(v)|, limitada a [0, 1]."""
        pos_u = self.graph.nodes[u].get("soma_pos")
        pos_v = self.graph.nodes[v].get("soma_pos")
        du = min(max(float(pos_u if pos_u is not None else 0.5), 0.0), 1.0)
        dv = min(max(float(pos_v if pos_v is not None else 0.5), 0.0), 1.0)
        return abs(du - dv)

    def _extrair_dataframe_fluxo(self) -> pd.DataFrame:
        """Extrai todas as multi-arestas com classes, camada, peso e custo de cabeamento."""
        registros = []
        for u, v, camada, dados in self.graph.edges(keys=True, data=True):
            peso = dados.get("weight", 1)
            distancia = self._distancia_soma(u, v)
            registros.append({
                "Neuronio_Origem": u,
                "Neuronio_Destino": v,
                "Origem": self.graph.nodes[u].get("tipo_funcional", "Desconhecido"),
                "Destino": self.graph.nodes[v].get("tipo_funcional", "Desconhecido"),
                "Camada": camada,
                "Peso_Sinaptico": peso,
                "Distancia_Soma": distancia,
                "Custo_Cabeamento": float(peso) * distancia,
                "Arestas": 1,
            })
        df = pd.DataFrame(registros)
        if self.agrupar_comando and not df.empty:
            df["Origem"] = df["Origem"].replace({"Interneurônio (Comando)": "Interneurônio"})
            df["Destino"] = df["Destino"].replace({"Interneurônio (Comando)": "Interneurônio"})
        return df

    def obter_ordem_classes(self) -> List[str]:
        """Retorna a lista de classes presentes ordenada segundo a hierarquia biológica."""
        if self.df_fluxo.empty:
            return []
        presentes = set(self.df_fluxo["Origem"].unique()) | set(self.df_fluxo["Destino"].unique())
        return [c for c in ORDEM_CLASSES_PADRAO if c in presentes]

    def calcular_custo_cabeamento(self, camada: Optional[str] = None) -> pd.DataFrame:
        """Retorna o custo de cabeamento espacial por multi-aresta (peso × distância entre somas)."""
        colunas = [
            "Neuronio_Origem", "Neuronio_Destino", "Origem", "Destino",
            "Camada", "Peso_Sinaptico", "Distancia_Soma", "Custo_Cabeamento",
        ]
        if self.df_fluxo.empty:
            return pd.DataFrame(columns=colunas)
        df = self.df_fluxo if camada is None else self.df_fluxo[self.df_fluxo["Camada"] == camada]
        return df[colunas].reset_index(drop=True)

    def avaliar_custo_cabeamento_nulo(
        self,
        membros_rich_club: Optional[Set[str]] = None,
        n_runs_null: int = 50,
        seed_base: int = 42,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compara o custo de cabeamento espacial real contra permutações aleatórias de soma_pos."""
        s_efetiva = int(seed if seed is not None else seed_base)
        nos = list(self.graph.nodes())
        idx_map = {n: i for i, n in enumerate(nos)}

        raw_pos = [self.graph.nodes[n].get("soma_pos") for n in nos]
        validos = [
            min(max(float(p), 0.0), 1.0)
            for p in raw_pos
            if p is not None and not (isinstance(p, float) and np.isnan(p))
        ]
        mediana = float(np.median(validos)) if validos else 0.5
        pos = np.array([
            min(max(float(p), 0.0), 1.0)
            if (p is not None and not (isinstance(p, float) and np.isnan(p))) else mediana
            for p in raw_pos
        ], dtype=float)

        u_idx, v_idx, weights, camadas = [], [], [], []
        for u, v, cam, d in self.graph.edges(keys=True, data=True):
            u_idx.append(idx_map[u])
            v_idx.append(idx_map[v])
            weights.append(float(d.get("weight", 1.0)))
            camadas.append(cam)

        u_arr = np.asarray(u_idx, dtype=int)
        v_arr = np.asarray(v_idx, dtype=int)
        w_arr = np.asarray(weights, dtype=float)
        cam_arr = np.asarray(camadas, dtype=object)

        dist_real = np.abs(pos[u_arr] - pos[v_arr])
        cost_real = w_arr * dist_real

        rc_set = set(membros_rich_club or set())
        u_in_rc = np.array([nos[i] in rc_set for i in u_arr], dtype=bool)
        v_in_rc = np.array([nos[i] in rc_set for i in v_arr], dtype=bool)
        grupos_rc = {
            "Internas ao Rich-Club": dist_real[u_in_rc & v_in_rc].tolist(),
            "Feeder / Ponte Rich-Club": dist_real[u_in_rc ^ v_in_rc].tolist(),
            "Periféricas": dist_real[(~u_in_rc) & (~v_in_rc)].tolist(),
        }

        rng = np.random.default_rng(s_efetiva)
        mascaras = {
            "Quimica": cam_arr == "Quimica",
            "Eletrica": cam_arr == "Eletrica",
            "Combinada": np.ones(len(w_arr), dtype=bool),
        }
        custos_null: Dict[str, List[float]] = {k: [] for k in mascaras}
        dist_null_amostras: List[np.ndarray] = []

        for r in range(int(n_runs_null)):
            perm = rng.permutation(pos)
            d_n = np.abs(perm[u_arr] - perm[v_arr])
            c_n = w_arr * d_n
            if r < 10:
                dist_null_amostras.append(d_n)
            for k, mask in mascaras.items():
                custos_null[k].append(float(c_n[mask].sum()))

        linhas_resumo: List[Dict[str, Any]] = []
        resumo_json: Dict[str, Any] = {}
        for camada, mask in mascaras.items():
            w_tot = float(w_arr[mask].sum())
            c_tot = float(cost_real[mask].sum())
            d_pond = c_tot / w_tot if w_tot > 0 else 0.0
            d_med = float(dist_real[mask].mean()) if mask.any() else 0.0
            n_mu = float(np.mean(custos_null[camada]))
            n_sd = float(np.std(custos_null[camada]))
            econ = (1.0 - c_tot / n_mu) * 100.0 if n_mu > 0 else 0.0
            z_sc = (c_tot - n_mu) / n_sd if n_sd > 1e-12 else 0.0

            linhas_resumo.append({
                "Camada": camada,
                "Custo_Real": c_tot,
                "Custo_Nulo_Media": n_mu,
                "Custo_Nulo_Std": n_sd,
                "Economia_Cabeamento_Pct": econ,
                "Z_Score_Custo": z_sc,
                "Comprimento_Medio_Ponderado": d_pond,
                "Comprimento_Medio_Nao_Ponderado": d_med,
            })
            resumo_json[camada.lower()] = {
                "custo_real": c_tot,
                "custo_nulo_media": n_mu,
                "custo_nulo_std": n_sd,
                "economia_cabeamento_pct": econ,
                "z_score_custo": z_sc,
                "comprimento_medio_ponderado": d_pond,
                "comprimento_medio_nao_ponderado": d_med,
            }

        resumo_json["grupos_rich_club"] = {
            g: {"n_arestas": len(vals), "comprimento_medio": float(np.mean(vals)) if vals else 0.0}
            for g, vals in grupos_rc.items()
        }
        return {
            "df_custo_nulo": pd.DataFrame(linhas_resumo),
            "dist_comprimentos_real": dist_real.tolist(),
            "dist_comprimentos_nulo": np.concatenate(dist_null_amostras).tolist() if dist_null_amostras else [],
            "comprimentos_por_grupo_rc": grupos_rc,
            "resumo_json": resumo_json,
        }

    def criar_matriz_fluxo(self, camada: str, metrica: str = "Peso_Sinaptico") -> pd.DataFrame:
        """Gera a matriz origem -> destino para uma camada ('Quimica' ou 'Eletrica')."""
        ordem = self.obter_ordem_classes()
        sub = self.df_fluxo[self.df_fluxo["Camada"] == camada]
        matriz = pd.pivot_table(
            sub, values=metrica, index="Origem", columns="Destino", aggfunc="sum", fill_value=0
        )
        return matriz.reindex(index=ordem, columns=ordem, fill_value=0)

    def obter_matrizes_quimica_eletrica(
        self, metrica: str = "Peso_Sinaptico"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retorna simultaneamente as matrizes das camadas Química e Elétrica."""
        return (
            self.criar_matriz_fluxo("Quimica", metrica=metrica),
            self.criar_matriz_fluxo("Eletrica", metrica=metrica),
        )

    def _calcular_tabela_centralidades(self, camada: str) -> pd.DataFrame:
        """Calcula e armazena em cache as centralidades base para a camada informada."""
        if camada in self._cache_centralidades:
            return self._cache_centralidades[camada].copy()

        projetado = projetar_grafo(
            self.graph, camada=camada, direcionado=True, adicionar_distancia=True, preservar_atributos_nos=False
        )
        if projetado.number_of_edges() == 0:
            interm = {n: 0.0 for n in projetado}
            pr = {n: 0.0 for n in projetado}
            prox = {n: 0.0 for n in projetado}
        else:
            interm = nx.betweenness_centrality(projetado, weight="distance", normalized=True)
            pr = nx.pagerank(projetado, weight="weight")
            prox = nx.closeness_centrality(projetado, distance="distance", wf_improved=True)

        fe = dict(projetado.in_degree(weight="weight"))
        fs = dict(projetado.out_degree(weight="weight"))
        registros = [
            {
                "Neuronio": n,
                "Tipo": self.graph.nodes[n].get("tipo_funcional", "Desconhecido"),
                "Grau_Total": int(projetado.degree(n)),
                "Grau_Entrada": int(projetado.in_degree(n)),
                "Grau_Saida": int(projetado.out_degree(n)),
                "Grau_Ponderado": fe[n] + fs[n],
                "Forca_Entrada": fe[n],
                "Forca_Saida": fs[n],
                "Intermediacao": interm[n],
                "PageRank": pr[n],
                "Proximidade": prox[n],
            }
            for n in projetado.nodes
        ]
        df = pd.DataFrame(registros)
        self._cache_centralidades[camada] = df
        return df.copy()

    def ranking_estratificado_por_tipo(self, camada: str = "Combinada") -> pd.DataFrame:
        """Ranqueia os neurônios dentro de cada tipo funcional a partir do cache de centralidades."""
        colunas = [
            "Tipo", "Neuronio", "Posicao_No_Tipo", "E_Chefe",
            "Grau_Ponderado", "Forca_Entrada", "Forca_Saida",
            "PageRank", "Intermediacao", "Proximidade",
            "Pct_Grau_Ponderado", "Pct_PageRank", "Pct_Intermediacao", "Pct_Proximidade",
            "Score_Chefe",
        ]
        res = self._calcular_tabela_centralidades(camada)
        if res.empty:
            return pd.DataFrame(columns=colunas)

        for col in ("Grau_Ponderado", "PageRank", "Intermediacao", "Proximidade"):
            res[f"Pct_{col}"] = res.groupby("Tipo")[col].rank(pct=True, method="average")
        res["Score_Chefe"] = res[["Pct_Grau_Ponderado", "Pct_PageRank", "Pct_Intermediacao"]].mean(axis=1)

        ordem_tipos = self.obter_ordem_classes()
        extras = sorted(set(res["Tipo"]) - set(ordem_tipos))
        res["Tipo"] = pd.Categorical(res["Tipo"], categories=ordem_tipos + extras)
        ord_df = res.sort_values(
            ["Tipo", "Score_Chefe", "Forca_Saida", "Neuronio"],
            ascending=[True, False, False, True],
        )
        ord_df["Posicao_No_Tipo"] = ord_df.groupby("Tipo", observed=False).cumcount() + 1
        ord_df["E_Chefe"] = ord_df["Posicao_No_Tipo"] == 1
        return ord_df[colunas].reset_index(drop=True)

    def resumo_chefes_por_tipo(self, camada: str = "Combinada") -> pd.DataFrame:
        """Retorna uma linha por tipo funcional com o chefe composto e os líderes por métrica."""
        colunas = [
            "Tipo", "Chefe", "Score_Chefe",
            "Lider_Grau_Ponderado", "Lider_PageRank", "Lider_Intermediacao", "Lider_Proximidade",
        ]
        ranking = self.ranking_estratificado_por_tipo(camada=camada)
        if ranking.empty:
            return pd.DataFrame(columns=colunas)

        resumo = ranking[ranking["E_Chefe"]][["Tipo", "Neuronio", "Score_Chefe"]].rename(
            columns={"Neuronio": "Chefe"}
        ).copy()
        resumo["Tipo"] = resumo["Tipo"].astype(str)
        for col in ("Grau_Ponderado", "PageRank", "Intermediacao", "Proximidade"):
            maximo = ranking.groupby("Tipo", observed=False)[col].transform("max")
            lideres = ranking[ranking[col] == maximo].drop_duplicates("Tipo", keep="first")
            resumo[f"Lider_{col}"] = resumo["Tipo"].map(dict(zip(lideres["Tipo"].astype(str), lideres["Neuronio"])))
        return resumo[colunas].reset_index(drop=True)

    def ranking_global_unificado(self, camada: str = "Combinada") -> pd.DataFrame:
        """Calcula o ranking global unificado agregando os 4 pilares de centralidade."""
        colunas = [
            "Posicao_Global", "Neuronio", "Tipo", "Score_Global",
            "Grau_Ponderado", "Forca_Entrada", "Forca_Saida",
            "Intermediacao", "Proximidade", "PageRank",
            "Pct_Grau_Ponderado", "Pct_Intermediacao", "Pct_Proximidade", "Pct_PageRank",
            "Justificativa_Biologica",
        ]
        res = self._calcular_tabela_centralidades(camada)
        if res.empty:
            return pd.DataFrame(columns=colunas)

        for col in ("Grau_Ponderado", "Intermediacao", "Proximidade", "PageRank"):
            res[f"Pct_{col}"] = res[col].rank(pct=True, method="average")
        res["Score_Global"] = res[
            ["Pct_Grau_Ponderado", "Pct_Intermediacao", "Pct_Proximidade", "Pct_PageRank"]
        ].mean(axis=1)
        res["Justificativa_Biologica"] = [
            JUSTIFICATIVAS_BIOLOGICAS_TOP.get(n, f"Neurônio {n} ({t}): integração funcional.")
            for n, t in zip(res["Neuronio"], res["Tipo"])
        ]
        ord_df = res.sort_values(
            ["Score_Global", "Forca_Saida", "Grau_Ponderado", "Neuronio"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        ord_df["Posicao_Global"] = ord_df.index + 1
        return ord_df[colunas]

    def obter_top_vertices_globais(self, top_n: int = 3, camada: str = "Combinada") -> pd.DataFrame:
        """Retorna as top_n primeiras linhas do ranking global unificado."""
        return self.ranking_global_unificado(camada=camada).head(top_n).copy()

    def calcular_rich_club_ponderado(
        self,
        camada: str = "Combinada",
        n_iter_nulo: int = 50,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, List[str], int]:
        """Calcula a curva do Coeficiente Rich-Club Ponderado (Opsahl et al., 2008)."""
        projetado = projetar_grafo(
            self.graph, camada=camada, direcionado=True, adicionar_distancia=False, preservar_atributos_nos=False
        )
        graus = dict(projetado.degree())
        nos = list(projetado.nodes())
        arestas_orig = [(u, v, float(d["weight"])) for u, v, d in projetado.edges(data=True)]
        if not arestas_orig:
            return pd.DataFrame(), [], 0

        soma_acum = np.cumsum(np.sort([w for _, _, w in arestas_orig])[::-1])
        limiares_k = [k for k in sorted(set(graus.values())) if sum(1 for g in graus.values() if g > k) >= 3]
        conjuntos_nos = {k: {n for n, g in graus.items() if g > k} for k in limiares_k}

        dados_k = {}
        for k in limiares_k:
            club = conjuntos_nos[k]
            n_club = len(club)
            pesos_int = [w for u, v, w in arestas_orig if u in club and v in club]
            e_k = len(pesos_int)
            w_k = float(sum(pesos_int))
            dados_k[k] = {
                "k": k,
                "Num_Nos": n_club,
                "Arestas_Club": e_k,
                "Peso_Club": w_k,
                "Phi_Topologico": e_k / (n_club * (n_club - 1)) if n_club > 1 else 0.0,
                "Phi_w": w_k / float(soma_acum[e_k - 1]) if e_k > 0 else 0.0,
            }

        rng = np.random.default_rng(seed)
        acum_nulo = {k: [] for k in limiares_k}
        n_swaps = max(len(arestas_orig) * 2, 1000)

        for _ in range(n_iter_nulo):
            g_null = gerar_nulo_direcionado(arestas_orig, nos, n_swaps=n_swaps, rng=rng)
            arestas_nulas = [(u, v, float(d["weight"])) for u, v, d in g_null.edges(data=True)]
            for k in limiares_k:
                club = conjuntos_nos[k]
                pesos_int = [w for u, v, w in arestas_nulas if u in club and v in club]
                e_n = len(pesos_int)
                acum_nulo[k].append(float(sum(pesos_int)) / float(soma_acum[e_n - 1]) if e_n > 0 else 0.0)

        registros = []
        for k in limiares_k:
            item = dados_k[k]
            med = float(np.mean(acum_nulo[k])) if acum_nulo[k] else 0.0
            item["Phi_w_Null"] = med
            item["Rho_w"] = item["Phi_w"] / med if med > 0 else 0.0
            registros.append(item)

        df_rc = pd.DataFrame(registros)
        if df_rc.empty:
            return df_rc, [], 0

        cand = df_rc[(df_rc["Num_Nos"] >= 5) & (df_rc["Num_Nos"] <= 18) & (df_rc["Rho_w"] > 1.0)]
        idx_otimo = cand["Rho_w"].idxmax() if not cand.empty else df_rc["Rho_w"].idxmax()
        k_otimo = int(df_rc.loc[idx_otimo, "k"])
        membros_elite = sorted(conjuntos_nos[k_otimo], key=lambda n: graus[n], reverse=True)
        return df_rc, membros_elite, k_otimo
