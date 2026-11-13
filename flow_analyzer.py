"""
Módulo de análise estatística, resumo topológico, influência e Rich-Club Ponderado.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from config import ORDEM_CLASSES_PADRAO


class ConnectomeAnalyzer:
    """Realiza análises de fluxo entre classes, influência e Rich-Club Ponderado."""

    def __init__(self, graph: nx.MultiDiGraph, agrupar_comando: bool = False):
        self.graph = graph
        self.agrupar_comando = agrupar_comando
        self.df_fluxo: pd.DataFrame = self._extrair_dataframe_fluxo()

    def _extrair_dataframe_fluxo(self) -> pd.DataFrame:
        """Extrai todas as multi-arestas com suas classes de origem, destino, camada e peso."""
        registros = []
        for u, v, camada, dados in self.graph.edges(keys=True, data=True):
            tipo_u = self.graph.nodes[u].get("tipo_funcional", "Desconhecido")
            tipo_v = self.graph.nodes[v].get("tipo_funcional", "Desconhecido")
            peso = dados.get("weight", 1)

            registros.append({
                "Origem": tipo_u,
                "Destino": tipo_v,
                "Camada": camada,
                "Peso_Sinaptico": peso,
                "Arestas": 1
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

    def criar_matriz_fluxo(self, camada: str, metrica: str = "Peso_Sinaptico") -> pd.DataFrame:
        """Gera a matriz origem -> destino para uma camada ('Quimica' ou 'Eletrica')."""
        ordem = self.obter_ordem_classes()
        sub = self.df_fluxo[self.df_fluxo["Camada"] == camada]
        matriz = pd.pivot_table(
            sub,
            values=metrica,
            index="Origem",
            columns="Destino",
            aggfunc="sum",
            fill_value=0
        )
        return matriz.reindex(index=ordem, columns=ordem, fill_value=0)

    def obter_matrizes_quimica_eletrica(
        self, metrica: str = "Peso_Sinaptico"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Retorna simultaneamente as matrizes das camadas Química e Elétrica."""
        mat_quimica = self.criar_matriz_fluxo("Quimica", metrica=metrica)
        mat_eletrica = self.criar_matriz_fluxo("Eletrica", metrica=metrica)
        return mat_quimica, mat_eletrica

    def _projetar_camada_influencia(self, camada: str) -> nx.DiGraph:
        """Cria um grafo ponderado; junções elétricas são tratadas como bidirecionais."""
        camadas_validas = {"Quimica", "Eletrica", "Combinada"}
        if camada not in camadas_validas:
            raise ValueError(f"Camada inválida: {camada}. Use uma de {sorted(camadas_validas)}.")

        projetado = nx.DiGraph()
        projetado.add_nodes_from(self.graph.nodes)
        pesos_eletricos: Dict[Tuple[str, str], float] = {}

        for u, v, chave, dados in self.graph.edges(keys=True, data=True):
            tipo = dados.get("synapse_type", chave)
            peso = float(dados.get("weight", 1))
            if peso < 0:
                raise ValueError(f"Peso sináptico negativo na conexão {u} -> {v}: {peso}.")
            if peso == 0:
                continue

            if tipo == "Eletrica":
                if camada in {"Eletrica", "Combinada"}:
                    par = (u, v) if u <= v else (v, u)
                    pesos_eletricos[par] = max(pesos_eletricos.get(par, 0.0), peso)
            elif tipo == "Quimica" and camada in {"Quimica", "Combinada"}:
                peso_existente = projetado[u][v]["weight"] if projetado.has_edge(u, v) else 0.0
                projetado.add_edge(u, v, weight=peso_existente + peso)

        for (u, v), peso in pesos_eletricos.items():
            peso_uv = projetado[u][v]["weight"] if projetado.has_edge(u, v) else 0.0
            projetado.add_edge(u, v, weight=peso_uv + peso)
            if u != v:
                peso_vu = projetado[v][u]["weight"] if projetado.has_edge(v, u) else 0.0
                projetado.add_edge(v, u, weight=peso_vu + peso)

        for u, v, dados in projetado.edges(data=True):
            dados["distance"] = 1.0 / dados["weight"]

        return projetado

    def analisar_influencia(self, camada: str = "Quimica") -> pd.DataFrame:
        """
        Classifica neurônios por influência topológica na camada indicada.
        """
        projetado = self._projetar_camada_influencia(camada)
        if projetado.number_of_edges() == 0:
            intermediacao = {n: 0.0 for n in projetado}
            page_rank = {n: 0.0 for n in projetado}
        else:
            intermediacao = nx.betweenness_centrality(
                projetado, weight="distance", normalized=True
            )
            page_rank = nx.pagerank(projetado, weight="weight")

        forca_entrada = dict(projetado.in_degree(weight="weight"))
        forca_saida = dict(projetado.out_degree(weight="weight"))
        registros = []
        for neuronio in projetado.nodes:
            dados_no = self.graph.nodes[neuronio]
            registros.append({
                "Neuronio": neuronio,
                "Tipo": dados_no.get("tipo_funcional", "Desconhecido"),
                "Grau_Entrada": projetado.in_degree(neuronio),
                "Grau_Saida": projetado.out_degree(neuronio),
                "Forca_Entrada": forca_entrada[neuronio],
                "Forca_Saida": forca_saida[neuronio],
                "Intermediacao": intermediacao[neuronio],
                "PageRank": page_rank[neuronio],
            })

        colunas = [
            "Neuronio", "Tipo", "Grau_Entrada", "Grau_Saida",
            "Forca_Entrada", "Forca_Saida", "Intermediacao", "PageRank",
            "Score_Influencia", "Destaque_Classe"
        ]
        resultados = pd.DataFrame(registros)
        if resultados.empty:
            return pd.DataFrame(columns=colunas)

        metricas_ranking = ["Grau_Saida", "Forca_Saida", "Intermediacao", "PageRank"]
        percentis = resultados[metricas_ranking].rank(method="average", pct=True)
        resultados["Score_Influencia"] = percentis.mean(axis=1)

        ordenados = resultados.sort_values(
            ["Score_Influencia", "Forca_Saida", "Grau_Saida", "Neuronio"],
            ascending=[False, False, False, True]
        )
        vencedores = ordenados.groupby("Tipo", sort=False).head(1)["Neuronio"]
        resultados["Destaque_Classe"] = resultados["Neuronio"].isin(vencedores)

        return resultados[colunas].sort_values(
            ["Score_Influencia", "Forca_Saida", "Grau_Saida", "Neuronio"],
            ascending=[False, False, False, True]
        ).reset_index(drop=True)

    def ranking_estratificado_por_tipo(self, camada: str = "Combinada") -> pd.DataFrame:
        """
        Ranqueia os neurônios dentro de cada tipo funcional, usando métricas globais
        calculadas sobre o grafo projetado (grau ponderado, PageRank e intermediação).

        O "chefe" de cada tipo é o neurônio com o maior score composto (média dos três
        percentis), com desempate determinístico por força de saída e, por fim, nome.
        """
        projetado = self._projetar_camada_influencia(camada)

        if projetado.number_of_edges() == 0:
            intermediacao = {n: 0.0 for n in projetado}
            page_rank = {n: 0.0 for n in projetado}
        else:
            intermediacao = nx.betweenness_centrality(
                projetado, weight="distance", normalized=True
            )
            page_rank = nx.pagerank(projetado, weight="weight")

        forca_entrada = dict(projetado.in_degree(weight="weight"))
        forca_saida = dict(projetado.out_degree(weight="weight"))

        registros = []
        for neuronio in projetado.nodes:
            dados_no = self.graph.nodes[neuronio]
            fe = forca_entrada[neuronio]
            fs = forca_saida[neuronio]
            registros.append({
                "Neuronio": neuronio,
                "Tipo": dados_no.get("tipo_funcional", "Desconhecido"),
                "Grau_Ponderado": fe + fs,
                "Forca_Entrada": fe,
                "Forca_Saida": fs,
                "PageRank": page_rank[neuronio],
                "Intermediacao": intermediacao[neuronio],
            })

        colunas = [
            "Tipo", "Neuronio", "Posicao_No_Tipo", "E_Chefe",
            "Grau_Ponderado", "Forca_Entrada", "Forca_Saida",
            "PageRank", "Intermediacao",
            "Pct_Grau_Ponderado", "Pct_PageRank", "Pct_Intermediacao",
            "Score_Chefe"
        ]
        resultados = pd.DataFrame(registros)
        if resultados.empty:
            return pd.DataFrame(columns=colunas)

        # Percentis calculados dentro de cada tipo funcional (método average para empates).
        resultados["Pct_Grau_Ponderado"] = resultados.groupby("Tipo")["Grau_Ponderado"].rank(pct=True, method="average")
        resultados["Pct_PageRank"] = resultados.groupby("Tipo")["PageRank"].rank(pct=True, method="average")
        resultados["Pct_Intermediacao"] = resultados.groupby("Tipo")["Intermediacao"].rank(pct=True, method="average")

        # Score composto: média dos três percentis (garante um único chefe por tipo).
        resultados["Score_Chefe"] = resultados[
            ["Pct_Grau_Ponderado", "Pct_PageRank", "Pct_Intermediacao"]
        ].mean(axis=1)

        # Ordena os tipos pela hierarquia biológica padrão (extras em ordem alfabética).
        ordem_tipos = self.obter_ordem_classes()
        extras = sorted(set(resultados["Tipo"]) - set(ordem_tipos))
        resultados["Tipo"] = pd.Categorical(resultados["Tipo"], categories=ordem_tipos + extras)

        ordenados = resultados.sort_values(
            ["Tipo", "Score_Chefe", "Forca_Saida", "Neuronio"],
            ascending=[True, False, False, True]
        )
        ordenados["Posicao_No_Tipo"] = ordenados.groupby("Tipo").cumcount() + 1
        ordenados["E_Chefe"] = ordenados["Posicao_No_Tipo"] == 1

        return ordenados[colunas].reset_index(drop=True)

    def resumo_chefes_por_tipo(self, camada: str = "Combinada") -> pd.DataFrame:
        """
        Retorna uma linha por tipo funcional com o neurônio chefe (maior Score_Chefe)
        e o líder de cada métrica individual dentro do tipo.

        Em caso de empate em uma métrica individual, o desempate segue a mesma
        convenção do ranking composto (Score_Chefe, força de saída e nome).
        """
        colunas = [
            "Tipo", "Chefe", "Score_Chefe",
            "Lider_Grau_Ponderado", "Lider_PageRank", "Lider_Intermediacao"
        ]
        ranking = self.ranking_estratificado_por_tipo(camada=camada)
        if ranking.empty:
            return pd.DataFrame(columns=colunas)

        chefes = ranking[ranking["E_Chefe"]].copy()
        resumo = chefes[["Tipo", "Neuronio", "Score_Chefe"]].rename(columns={"Neuronio": "Chefe"})
        resumo["Tipo"] = resumo["Tipo"].astype(str)

        for coluna, rotulo in [
            ("Grau_Ponderado", "Lider_Grau_Ponderado"),
            ("PageRank", "Lider_PageRank"),
            ("Intermediacao", "Lider_Intermediacao"),
        ]:
            maximo = ranking.groupby("Tipo")[coluna].transform("max")
            lideres = ranking[ranking[coluna] == maximo].drop_duplicates("Tipo", keep="first")
            mapa = dict(zip(lideres["Tipo"].astype(str), lideres["Neuronio"]))
            resumo[rotulo] = resumo["Tipo"].map(mapa)

        return resumo[colunas].reset_index(drop=True)

    @staticmethod
    def _randomizar_arestas_direcionadas(
        arestas: List[Tuple[str, str, float]],
        n_swaps: int,
        rng: np.random.Generator
    ) -> List[Tuple[str, str, float]]:
        """
        Realiza trocas de arestas direcionadas (Directed Double-Edge Swap) preservando
        exatamente o grau de entrada e saída de todos os vértices e embaralhando os pesos.
        """
        m = len(arestas)
        if m < 2:
            return list(arestas)

        origens = [u for u, _, _ in arestas]
        destinos = [v for _, v, _ in arestas]
        pesos = [w for _, _, w in arestas]
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
        return list(zip(origens, destinos, pesos_embaralhados))

    def calcular_rich_club_ponderado(
        self,
        camada: str = "Combinada",
        n_iter_nulo: int = 50,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, List[str], int]:
        """
        Calcula a curva do Coeficiente Rich-Club Ponderado (Opsahl et al., 2008)
        e a versão normalizada frente a um modelo nulo que preserva graus.
        """
        projetado = self._projetar_camada_influencia(camada)
        graus = dict(projetado.degree())

        arestas_originais = [
            (u, v, float(d["weight"])) for u, v, d in projetado.edges(data=True)
        ]
        if not arestas_originais:
            return pd.DataFrame(), [], 0

        pesos_ordenados = np.sort([w for _, _, w in arestas_originais])[::-1]
        soma_acumulada_ranking = np.cumsum(pesos_ordenados)

        valores_k = sorted(set(graus.values()))
        limiares_k = [
            k for k in valores_k
            if sum(1 for g in graus.values() if g > k) >= 3
        ]

        conjuntos_nos = {
            k: {n for n, g in graus.items() if g > k}
            for k in limiares_k
        }

        dados_k = {}
        for k in limiares_k:
            club_nodes = conjuntos_nos[k]
            n_club = len(club_nodes)
            pesos_internos = [
                w for u, v, w in arestas_originais
                if u in club_nodes and v in club_nodes
            ]
            e_k = len(pesos_internos)
            w_k = float(sum(pesos_internos))

            phi_topo = e_k / (n_club * (n_club - 1)) if n_club > 1 else 0.0
            phi_w = w_k / float(soma_acumulada_ranking[e_k - 1]) if e_k > 0 else 0.0

            dados_k[k] = {
                "k": k,
                "Num_Nos": n_club,
                "Arestas_Club": e_k,
                "Peso_Club": w_k,
                "Phi_Topologico": phi_topo,
                "Phi_w": phi_w
            }

        rng = np.random.default_rng(seed)
        acumulador_nulo = {k: [] for k in limiares_k}
        n_swaps = max(len(arestas_originais) * 2, 1000)

        for _ in range(n_iter_nulo):
            arestas_nulas = self._randomizar_arestas_direcionadas(
                arestas_originais, n_swaps=n_swaps, rng=rng
            )
            for k in limiares_k:
                club_nodes = conjuntos_nos[k]
                pesos_int_nulo = [
                    w for u, v, w in arestas_nulas
                    if u in club_nodes and v in club_nodes
                ]
                e_nulo = len(pesos_int_nulo)
                if e_nulo > 0:
                    w_nulo = float(sum(pesos_int_nulo))
                    phi_w_nulo = w_nulo / float(soma_acumulada_ranking[e_nulo - 1])
                else:
                    phi_w_nulo = 0.0
                acumulador_nulo[k].append(phi_w_nulo)

        registros = []
        for k in limiares_k:
            item = dados_k[k]
            media_nula = float(np.mean(acumulador_nulo[k])) if acumulador_nulo[k] else 0.0
            rho_w = item["Phi_w"] / media_nula if media_nula > 0 else 0.0
            item["Phi_w_Null"] = media_nula
            item["Rho_w"] = rho_w
            registros.append(item)

        df_rc = pd.DataFrame(registros)
        if df_rc.empty:
            return df_rc, [], 0

        candidatos_elite = df_rc[(df_rc["Num_Nos"] >= 5) & (df_rc["Num_Nos"] <= 18) & (df_rc["Rho_w"] > 1.0)]
        if not candidatos_elite.empty:
            idx_otimo = candidatos_elite["Rho_w"].idxmax()
        else:
            idx_otimo = df_rc["Rho_w"].idxmax()

        k_otimo = int(df_rc.loc[idx_otimo, "k"])
        membros_elite = sorted(
            conjuntos_nos[k_otimo],
            key=lambda n: graus[n],
            reverse=True
        )
        return df_rc, membros_elite, k_otimo

    def exibir_resumo_console(self, neuronio_exemplo: str = "AVAL") -> None:
        """Imprime estatísticas gerais, contagem por tipo e matrizes no terminal."""
        print(f"Total de Neurônios (Vértices): {self.graph.number_of_nodes()}")
        print(f"Total de Conexões (Multi-arestas): {self.graph.number_of_edges()}")

        contagem_tipos = pd.Series(
            nx.get_node_attributes(self.graph, "tipo_funcional")
        ).value_counts()
        print("\n--- Distribuição dos Tipos de Neurônios ---")
        print(contagem_tipos)

        if neuronio_exemplo in self.graph.nodes:
            print(f"\n--- Atributos vinculados ao neurônio {neuronio_exemplo} ---")
            print(self.graph.nodes[neuronio_exemplo])

        mat_quimica, mat_eletrica = self.obter_matrizes_quimica_eletrica()
        print("\n=== MATRIZ DE FLUXO: SINAPSES QUÍMICAS (Peso Total: Origem -> Destino) ===")
        print(mat_quimica, "\n")

        print("=== MATRIZ DE ACOPLAMENTO: JUNÇÕES ELÉTRICAS (Peso Total: Origem <-> Destino) ===")
        print(mat_eletrica, "\n")