"""
Módulo de análise estatística, resumo topológico e matrizes de transição sináptica.
"""

from typing import List, Tuple
import pandas as pd
import networkx as nx
from config import ORDEM_CLASSES_PADRAO


class ConnectomeAnalyzer:
    """Realiza extrações tabulares, matrizes de fluxo entre classes e inspeções no conectoma."""

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
