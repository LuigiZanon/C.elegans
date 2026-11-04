"""
Módulo de análise estatística, resumo topológico e matrizes de transição sináptica.
"""

from typing import Dict, List, Tuple
import pandas as pd
import networkx as nx
from config import ORDEM_CLASSES_PADRAO


class ConnectomeAnalyzer:
    """Realiza análises de fluxo entre classes e de influência dos neurônios."""

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

        O escore é a média dos percentis de grau de saída, força de saída,
        intermediação e PageRank. Os pesos químicos são direcionais; junções
        elétricas são projetadas como relações bidirecionais.
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
