"""
Ponto de entrada principal (Main) para construção, análise e visualização do conectoma C. elegans.
"""

from connectome_builder import ConnectomeBuilder
from flow_analyzer import ConnectomeAnalyzer
from visualizer import ConnectomeVisualizer


def main() -> None:
    # 1. Carregamento das planilhas e construção do MultiDiGraph
    builder = ConnectomeBuilder(connect_path="NeuronConnect.xls")
    grafo_multi = builder.construir_grafo()

    # 2. Análise de fluxo sináptico e impressão de relatórios no console
    analyzer = ConnectomeAnalyzer(grafo_multi, agrupar_comando=False)
    analyzer.exibir_resumo_console(neuronio_exemplo="AVAL")

    mat_quimica, mat_eletrica = analyzer.obter_matrizes_quimica_eletrica()
    ordem_classes = analyzer.obter_ordem_classes()

    # 3. Identificação dos neurônios mais influentes em cada classe funcional
    resultados_influencia = analyzer.analisar_influencia(camada="Quimica")
    destaques = resultados_influencia[resultados_influencia["Destaque_Classe"]]
    colunas_destaques = [
        "Tipo", "Neuronio", "Score_Influencia", "Grau_Saida",
        "Forca_Saida", "Intermediacao", "PageRank"
    ]
    print("\n=== NEURÔNIOS MAIS INFLUENTES POR CLASSE (CAMADA QUÍMICA) ===")
    print(destaques[colunas_destaques].to_string(index=False))
    influentes = dict(zip(destaques["Neuronio"], destaques["Score_Influencia"]))

    # 4. Visualização estática (Heatmaps Matplotlib) e interativa (Pyvis HTML)
    visualizer = ConnectomeVisualizer(grafo_multi)
    visualizer.plotar_matrizes_fluxo(mat_quimica, mat_eletrica, ordem_classes)
    visualizer.exportar_html_interativo(
        "celegans_rede_interativa.html",
        notebook=False,
        influentes=influentes
    )


if __name__ == "__main__":
    main()
