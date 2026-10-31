"""
Módulo responsável pela leitura das planilhas do WormAtlas e construção do MultiDiGraph.
"""

import os
from typing import Optional
import pandas as pd
import networkx as nx
from config import (
    PREFIXOS_SENSORIAIS,
    PREFIXOS_MOTORES,
    PREFIXOS_FARINGEOS,
    INTERNEURONIOS_COMANDO
)


class ConnectomeBuilder:
    """Carrega os dados do WormAtlas e estrutura o conectoma como um MultiDiGraph."""

    def __init__(self, connect_path: str = "NeuronConnect.xls", type_path: Optional[str] = None):
        self.connect_path = connect_path
        if type_path is None:
            self.type_path = "NeuronType.xls" if os.path.exists("NeuronType.xls") else "NeuronType.xlsx"
        else:
            self.type_path = type_path

        self.df_conn: Optional[pd.DataFrame] = None
        self.df_type: Optional[pd.DataFrame] = None
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

    @staticmethod
    def classificar_funcao(nome_neuronio: str, tem_nmj: bool = False) -> str:
        """Retorna a classe funcional canônica do neurônio a partir da sua sigla e presença de NMJ."""
        if nome_neuronio in INTERNEURONIOS_COMANDO:
            return "Interneurônio (Comando)"
        if nome_neuronio.startswith(PREFIXOS_FARINGEOS):
            return "Faríngeo"
        if nome_neuronio.startswith(PREFIXOS_SENSORIAIS):
            return "Sensorial"
        if nome_neuronio.startswith(PREFIXOS_MOTORES) or tem_nmj:
            return "Motor"
        return "Interneurônio"

    def carregar_planilhas(self) -> None:
        """Lê e padroniza os arquivos Excel de conectividade e tipos neuronais."""
        self.df_conn = pd.read_excel(self.connect_path)
        self.df_type = pd.read_excel(self.type_path)

        self.df_conn.columns = [c.strip() for c in self.df_conn.columns]
        self.df_type.columns = [c.strip() for c in self.df_type.columns]

        self.df_conn["Neuron 1"] = self.df_conn["Neuron 1"].astype(str).str.strip()
        self.df_conn["Neuron 2"] = self.df_conn["Neuron 2"].astype(str).str.strip()
        self.df_conn["Type"] = self.df_conn["Type"].astype(str).str.strip()

    def construir_grafo(self) -> nx.MultiDiGraph:
        """Executa o pipeline completo de instanciação de nós e multi-arestas tipadas."""
        if self.df_conn is None or self.df_type is None:
            self.carregar_planilhas()

        nmj_por_neuronio = (
            self.df_conn[self.df_conn["Type"] == "NMJ"]
            .groupby("Neuron 1")["Nbr"]
            .sum()
            .to_dict()
        )

        self.graph = nx.MultiDiGraph()
        col_soma_reg = "Soma region" if "Soma region" in self.df_type.columns else "Soma Region"

        # 1. Adiciona os vértices com metadados morfológicos e funcionais
        for _, row in self.df_type.iterrows():
            neuronio = str(row["Neuron"]).strip()
            peso_nmj = int(nmj_por_neuronio.get(neuronio, 0))

            self.graph.add_node(
                neuronio,
                tipo_funcional=self.classificar_funcao(neuronio, tem_nmj=(peso_nmj > 0)),
                soma_pos=float(row.get("Soma Position", 0.0)),
                regiao_soma=str(row.get(col_soma_reg, "N/A")).strip(),
                span=str(row.get("Span", "N/A")).strip(),
                ganglio_ay=str(row.get("AY NeuronType", "N/A")).strip(),
                peso_nmj=peso_nmj
            )

        # 2. Filtra e adiciona apenas conexões químicas enviadas ('S', 'Sp') e junções elétricas ('EJ')
        df_arestas = self.df_conn[self.df_conn["Type"].isin(["S", "Sp", "EJ"])]

        for _, row in df_arestas.iterrows():
            u = row["Neuron 1"]
            v = row["Neuron 2"]
            peso = int(row["Nbr"])
            tipo_cod = row["Type"]

            if u == v or v == "NMJ":
                continue

            for no in (u, v):
                if no not in self.graph:
                    peso_m = int(nmj_por_neuronio.get(no, 0))
                    self.graph.add_node(
                        no,
                        tipo_funcional=self.classificar_funcao(no, tem_nmj=(peso_m > 0)),
                        soma_pos=None,
                        regiao_soma="N/A",
                        span="N/A",
                        ganglio_ay="N/A",
                        peso_nmj=peso_m
                    )

            camada = "Eletrica" if tipo_cod == "EJ" else "Quimica"

            if self.graph.has_edge(u, v, key=camada):
                self.graph[u][v][camada]["weight"] += peso
            else:
                self.graph.add_edge(u, v, key=camada, weight=peso, synapse_type=camada)

        return self.graph
