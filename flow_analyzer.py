"""
Módulo de análise estatística, resumo topológico, influência e Rich-Club Ponderado.
"""

from typing import Dict, List, Optional, Tuple
import textwrap
import numpy as np
import pandas as pd
import networkx as nx
from config import ORDEM_CLASSES_PADRAO


JUSTIFICATIVAS_BIOLOGICAS_TOP: Dict[str, str] = {
    "AVAL": (
        "Interneurônio de comando condutor primário de marcha retrógrada (locomoção de ré e manobra "
        "de fuga por evasão rápida). Integra estímulos sensoriais nociceptivos e mecânicos convergentes "
        "(ex.: ASH, FLP, ALM); projeta-se longitudinalmente por todo o corpo estabelecendo extensas "
        "junções comunicantes elétricas (gap junctions) com motoneurônios do tipo A (DA/VA) responsáveis "
        "pela ondulação para trás. Apresenta o maior Grau Ponderado (força sináptica total = 606), maior "
        "PageRank (0.0350) e maior Centralidade de Intermediação (0.2994) de todo o conectoma, operando "
        "como o principal hub integrador e gargalo de tráfego do sistema nervoso somático."
    ),
    "AVAR": (
        "Homólogo contralateral direito de AVAL. Atua em estreita cooperação e sinergia bilateral "
        "com AVAL na orquestração motora da reversão de marcha e fuga. Apresenta a mais alta Centralidade "
        "de Proximidade (1.6246, 100.0% global) de todo o conectoma, permitindo a comunicação e "
        "difusão de sinais no menor tempo geodésico médio para qualquer outro neurônio da rede. É o "
        "2º colocado em Grau Ponderado (563), Intermediação (0.2299) e PageRank (0.0330). Forma com AVAL "
        "a espinha dorsal pré-motora do organismo."
    ),
    "AVBR": (
        "Interneurônio de comando condutor da locomoção anterógrada (marcha para frente e exploração). "
        "Acopla-se fortemente via junções comunicantes elétricas e sinapses químicas aos motoneurônios "
        "da classe B (DB/VB), que geram a ondulação progressiva para frente. Estabelece com o par AVAL/AVAR "
        "o circuito oscilatório bi-estável central que comuta alternadamente o sentido do movimento "
        "do verme. Apresenta altíssimo Grau Ponderado (248, percentil 99.3%), Proximidade (percentil 96.8%) "
        "e PageRank (percentil 97.1%), consolidando-se como o 3º vértice mais influente globalmente."
    ),
    "PVCR": (
        "Interneurônio de comando posterior condutor de locomoção anterógrada. Localizado na cauda, "
        "recebe estímulos mecanossensoriais posteriores (ex.: PLM) e dispara a marcha para frente através "
        "de conexões sinápticas ascendentes com AVB e motoneurônios B."
    ),
    "AVBL": (
        "Par contralateral esquerdo de AVBR. Co-coordena a locomoção anterógrada contínua, conectando "
        "os interneurônios integradores cefálicos aos circuitos motores de avanço ao longo do cordão ventral."
    ),
    "PVCL": (
        "Par contralateral esquerdo de PVCR. Complementa a mediação do reflexo de avanço após toque "
        "mecânico na cauda, integrando o circuito de resposta anterógrada rápida."
    ),
    "VD01": (
        "Motoneurônio inervador da musculatura ventral; participa da transmissão do padrão ondulatório motor."
    ),
    "VA08": (
        "Motoneurônio do cordão ventral; acoplado eletricamente aos interneurônios de comando de reversão."
    ),
}


class ConnectomeAnalyzer:
    """Realiza análises de fluxo entre classes, influência e Rich-Club Ponderado."""

    def __init__(self, graph: nx.MultiDiGraph, agrupar_comando: bool = False):
        self.graph = graph
        self.agrupar_comando = agrupar_comando
        self.df_fluxo: pd.DataFrame = self._extrair_dataframe_fluxo()

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
            tipo_u = self.graph.nodes[u].get("tipo_funcional", "Desconhecido")
            tipo_v = self.graph.nodes[v].get("tipo_funcional", "Desconhecido")
            peso = dados.get("weight", 1)
            distancia = self._distancia_soma(u, v)

            registros.append({
                "Neuronio_Origem": u,
                "Neuronio_Destino": v,
                "Origem": tipo_u,
                "Destino": tipo_v,
                "Camada": camada,
                "Peso_Sinaptico": peso,
                "Distancia_Soma": distancia,
                "Custo_Cabeamento": float(peso) * distancia,
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

    def calcular_custo_cabeamento(self, camada: Optional[str] = None) -> pd.DataFrame:
        """
        Retorna o custo de cabeamento espacial por multi-aresta:
        peso sináptico × distância longitudinal entre os somas (soma_pos).

        Se ``camada`` for informada ('Quimica' ou 'Eletrica'), filtra por ela.
        """
        colunas = [
            "Neuronio_Origem", "Neuronio_Destino", "Origem", "Destino",
            "Camada", "Peso_Sinaptico", "Distancia_Soma", "Custo_Cabeamento"
        ]
        if self.df_fluxo.empty:
            return pd.DataFrame(columns=colunas)

        df = self.df_fluxo
        if camada is not None:
            df = df[df["Camada"] == camada]
        return df[colunas].reset_index(drop=True)

    def exibir_resumo_custo_cabeamento(self, top_n: int = 10) -> None:
        """Imprime no console o custo total de cabeamento por camada e as conexões mais caras."""
        if self.df_fluxo.empty:
            print("Nenhuma conexão disponível para cálculo de custo de cabeamento.")
            return

        print("\n=== CUSTO DE CABEAMENTO ESPACIAL (Peso Sináptico × Distância entre Somas) ===")
        for camada in ("Quimica", "Eletrica"):
            sub = self.df_fluxo[self.df_fluxo["Camada"] == camada]
            if sub.empty:
                continue
            peso_total = float(sub["Peso_Sinaptico"].sum())
            custo_total = float(sub["Custo_Cabeamento"].sum())
            dist_media = custo_total / peso_total if peso_total > 0 else 0.0
            print(
                f"Camada {camada}: custo total = {custo_total:.1f} | "
                f"distância média ponderada = {dist_media:.3f}"
            )

        top = self.df_fluxo.nlargest(top_n, "Custo_Cabeamento")
        print(f"\nTop {top_n} conexões por custo de cabeamento:")
        print(top[
            ["Neuronio_Origem", "Neuronio_Destino", "Camada",
             "Peso_Sinaptico", "Distancia_Soma", "Custo_Cabeamento"]
        ].to_string(index=False))

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
        calculadas sobre o grafo projetado (grau ponderado, PageRank, intermediação e proximidade).

        O "chefe" de cada tipo é o neurônio com o maior score composto (média dos três
        percentis canônicos), com desempate determinístico por força de saída e, por fim, nome.
        """
        projetado = self._projetar_camada_influencia(camada)

        if projetado.number_of_edges() == 0:
            intermediacao = {n: 0.0 for n in projetado}
            page_rank = {n: 0.0 for n in projetado}
            proximidade = {n: 0.0 for n in projetado}
        else:
            intermediacao = nx.betweenness_centrality(
                projetado, weight="distance", normalized=True
            )
            page_rank = nx.pagerank(projetado, weight="weight")
            proximidade = nx.closeness_centrality(
                projetado, distance="distance", wf_improved=True
            )

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
                "Proximidade": proximidade[neuronio],
            })

        colunas = [
            "Tipo", "Neuronio", "Posicao_No_Tipo", "E_Chefe",
            "Grau_Ponderado", "Forca_Entrada", "Forca_Saida",
            "PageRank", "Intermediacao", "Proximidade",
            "Pct_Grau_Ponderado", "Pct_PageRank", "Pct_Intermediacao", "Pct_Proximidade",
            "Score_Chefe"
        ]
        resultados = pd.DataFrame(registros)
        if resultados.empty:
            return pd.DataFrame(columns=colunas)

        # Percentis calculados dentro de cada tipo funcional (método average para empates).
        resultados["Pct_Grau_Ponderado"] = resultados.groupby("Tipo")["Grau_Ponderado"].rank(pct=True, method="average")
        resultados["Pct_PageRank"] = resultados.groupby("Tipo")["PageRank"].rank(pct=True, method="average")
        resultados["Pct_Intermediacao"] = resultados.groupby("Tipo")["Intermediacao"].rank(pct=True, method="average")
        resultados["Pct_Proximidade"] = resultados.groupby("Tipo")["Proximidade"].rank(pct=True, method="average")

        # Score composto: média dos percentis (garante um único chefe por tipo).
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
            "Lider_Grau_Ponderado", "Lider_PageRank", "Lider_Intermediacao", "Lider_Proximidade"
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
            ("Proximidade", "Lider_Proximidade"),
        ]:
            maximo = ranking.groupby("Tipo")[coluna].transform("max")
            lideres = ranking[ranking[coluna] == maximo].drop_duplicates("Tipo", keep="first")
            mapa = dict(zip(lideres["Tipo"].astype(str), lideres["Neuronio"]))
            resumo[rotulo] = resumo["Tipo"].map(mapa)

        return resumo[colunas].reset_index(drop=True)

    def ranking_global_unificado(self, camada: str = "Combinada") -> pd.DataFrame:
        """
        Calcula o ranking global unificado de todos os vértices do conectoma (sem estratificação por tipo),
        agregando os percentis globais dos quatro pilares de centralidade:
          1. Grau Ponderado (Força Sináptica Total s_u = s_u^in + s_u^out)
          2. Centralidade de Intermediação (Betweenness Centrality ponderada por d = 1/w)
          3. Centralidade de Proximidade (Closeness Centrality ponderada por d = 1/w)
          4. PageRank (fluxo estacionário ponderado por w)

        O Score Global Unificado é a média aritmética simples dos quatro percentis globais:
          Score_Global = (Pct_Grau_Ponderado + Pct_Intermediacao + Pct_Proximidade + Pct_PageRank) / 4.0

        O desempate segue a ordem: Score_Global desc, Força_Saida desc, Grau_Ponderado desc, Neuronio asc.
        """
        projetado = self._projetar_camada_influencia(camada)

        if projetado.number_of_edges() == 0:
            intermediacao = {n: 0.0 for n in projetado}
            page_rank = {n: 0.0 for n in projetado}
            proximidade = {n: 0.0 for n in projetado}
        else:
            intermediacao = nx.betweenness_centrality(
                projetado, weight="distance", normalized=True
            )
            page_rank = nx.pagerank(projetado, weight="weight")
            proximidade = nx.closeness_centrality(
                projetado, distance="distance", wf_improved=True
            )

        forca_entrada = dict(projetado.in_degree(weight="weight"))
        forca_saida = dict(projetado.out_degree(weight="weight"))

        registros = []
        for neuronio in projetado.nodes:
            dados_no = self.graph.nodes[neuronio]
            tipo = dados_no.get("tipo_funcional", "Desconhecido")
            fe = forca_entrada[neuronio]
            fs = forca_saida[neuronio]
            gp = fe + fs

            justificativa = JUSTIFICATIVAS_BIOLOGICAS_TOP.get(
                neuronio,
                f"Neurônio {neuronio} ({tipo}): integração no circuito funcional do nematódeo."
            )

            registros.append({
                "Neuronio": neuronio,
                "Tipo": tipo,
                "Grau_Ponderado": gp,
                "Forca_Entrada": fe,
                "Forca_Saida": fs,
                "Intermediacao": intermediacao[neuronio],
                "Proximidade": proximidade[neuronio],
                "PageRank": page_rank[neuronio],
                "Justificativa_Biologica": justificativa
            })

        colunas = [
            "Posicao_Global", "Neuronio", "Tipo", "Score_Global",
            "Grau_Ponderado", "Forca_Entrada", "Forca_Saida",
            "Intermediacao", "Proximidade", "PageRank",
            "Pct_Grau_Ponderado", "Pct_Intermediacao", "Pct_Proximidade", "Pct_PageRank",
            "Justificativa_Biologica"
        ]
        resultados = pd.DataFrame(registros)
        if resultados.empty:
            return pd.DataFrame(columns=colunas)

        resultados["Pct_Grau_Ponderado"] = resultados["Grau_Ponderado"].rank(pct=True, method="average")
        resultados["Pct_Intermediacao"] = resultados["Intermediacao"].rank(pct=True, method="average")
        resultados["Pct_Proximidade"] = resultados["Proximidade"].rank(pct=True, method="average")
        resultados["Pct_PageRank"] = resultados["PageRank"].rank(pct=True, method="average")

        resultados["Score_Global"] = resultados[
            ["Pct_Grau_Ponderado", "Pct_Intermediacao", "Pct_Proximidade", "Pct_PageRank"]
        ].mean(axis=1)

        ordenados = resultados.sort_values(
            ["Score_Global", "Forca_Saida", "Grau_Ponderado", "Neuronio"],
            ascending=[False, False, False, True]
        ).reset_index(drop=True)
        ordenados["Posicao_Global"] = ordenados.index + 1

        return ordenados[colunas]

    def obter_top_vertices_globais(self, top_n: int = 3, camada: str = "Combinada") -> pd.DataFrame:
        """Retorna as top_n primeiras linhas do ranking global unificado."""
        df_ranking = self.ranking_global_unificado(camada=camada)
        return df_ranking.head(top_n).copy()

    def exibir_relatorio_top_vertices(self, df_top: pd.DataFrame) -> None:
        """
        Imprime um relatório formatado e detalhado no console com os vértices mais importantes,
        suas métricas quantitativas, percentis e fundamentação biológica do papel de cada um.
        """
        if df_top is None or df_top.empty:
            print("Nenhum vértice encontrado no ranking global.")
            return

        print("\n" + "╔" + "═" * 92 + "╗")
        print("║      OS TRÊS VÉRTICES MAIS IMPORTANTES DO CONECTOMA DE C. ELEGANS (TOP 3 GLOBAL)          ║")
        print("║  Metodologia: Ranking Global Unificado (Grau Ponderado + Betweenness + Closeness + PageRank)║")
        print("╚" + "═" * 92 + "╝")

        medalhas = {1: "🥇 1º LUGAR", 2: "🥈 2º LUGAR", 3: "🥉 3º LUGAR"}

        for _, linha in df_top.iterrows():
            pos = int(linha["Posicao_Global"])
            nome = str(linha["Neuronio"])
            tipo = str(linha["Tipo"])
            score = float(linha["Score_Global"])
            gp = float(linha["Grau_Ponderado"])
            fe = float(linha["Forca_Entrada"])
            fs = float(linha["Forca_Saida"])
            interm = float(linha["Intermediacao"])
            prox = float(linha["Proximidade"])
            pr = float(linha["PageRank"])
            pct_gp = float(linha["Pct_Grau_Ponderado"]) * 100
            pct_int = float(linha["Pct_Intermediacao"]) * 100
            pct_prox = float(linha["Pct_Proximidade"]) * 100
            pct_pr = float(linha["Pct_PageRank"]) * 100
            rotulo_pos = medalhas.get(pos, f"#{pos}")

            print(f"\n{rotulo_pos}: NEURÔNIO '{nome}' | Classe: {tipo}")
            print(f"  • Score Global Composto: {score:.4f} (Média dos 4 percentis globais)")
            print(f"  • Métricas Absolutas:")
            print(f"      - Grau Ponderado (Força Sináptica): {gp:6.1f} (Entrada: {fe:5.1f} | Saída: {fs:5.1f})")
            print(f"      - Centralidade de Intermediação:     {interm:8.5f}")
            print(f"      - Centralidade de Proximidade:       {prox:8.5f}")
            print(f"      - PageRank Ponderado:               {pr:8.5f}")
            print(f"  • Posição Relativa no Conectoma (Percentis Globais):")
            print(f"      - Força Sináptica:   {pct_gp:5.1f}% superior de toda a rede")
            print(f"      - Intermediação:     {pct_int:5.1f}% superior (controle de caminhos mínimos)")
            print(f"      - Proximidade:       {pct_prox:5.1f}% superior (alcance geodésico)")
            print(f"      - PageRank:          {pct_pr:5.1f}% superior (fluxo estacionário)")
            print(f"  • Justificativa Funcional e Relevância Biológica:")
            texto_just = str(linha.get("Justificativa_Biologica", "")).strip()
            linhas_just = textwrap.wrap(texto_just, width=86)
            for lj in linhas_just:
                print(f"      {lj}")

        print("\n" + "─" * 94)
        print("  SÍNTESE COMPARATIVA DOS TOP 3 NEURÔNIOS:")
        print("─" * 94)
        tabela_resumo = df_top[
            ["Posicao_Global", "Neuronio", "Tipo", "Score_Global",
             "Grau_Ponderado", "Intermediacao", "Proximidade", "PageRank"]
        ].rename(columns={
            "Posicao_Global": "Rank",
            "Grau_Ponderado": "Força_Total",
            "Intermediacao": "Betweenness",
            "Proximidade": "Closeness"
        })
        print(tabela_resumo.to_string(index=False))
        print("═" * 94 + "\n")

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