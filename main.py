"""
Ponto de entrada principal (Main) para construção, análise e visualização do conectoma C. elegans.
Todas as figuras e o HTML interativo multi-layout são salvos automaticamente na pasta 'outputs/'.
"""

import os
from config import OUTPUT_DIR
from connectome_builder import ConnectomeBuilder
from flow_analyzer import ConnectomeAnalyzer
from visualizer import ConnectomeVisualizer


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # 4. Cálculo do Coeficiente Rich-Club Ponderado (Opsahl et al., 2008)
    df_rich_club, membros_elite, k_otimo = analyzer.calcular_rich_club_ponderado(
        camada="Combinada", n_iter_nulo=50, seed=42
    )
    print(f"\n=== COEFICIENTE RICH-CLUB PONDERADO (LIMIAR DE ELITE k > {k_otimo}) ===")
    print(f"Quantidade de neurônios no núcleo Rich-Club: {len(membros_elite)}")
    print(f"Neurônios membros: {', '.join(membros_elite)}")
    print("\nÚltimos degraus da curva Rich-Club Ponderada:")
    print(df_rich_club.tail(10).to_string(index=False))

    caminho_csv_rc = os.path.join(OUTPUT_DIR, "rich_club_ponderado.csv")
    df_rich_club.to_csv(caminho_csv_rc, index=False)
    print(f"\nTabela Rich-Club salva em: '{caminho_csv_rc}'")

    # 5. Visualizações salvas diretamente na pasta 'outputs/' e integradas no HTML interativo
    visualizer = ConnectomeVisualizer(grafo_multi, output_dir=OUTPUT_DIR)
    caminho_fluxo = visualizer.plotar_matrizes_fluxo(
        mat_quimica, mat_eletrica, ordem_classes, nome_arquivo="matrizes_fluxo.png"
    )
    caminho_rc = visualizer.plotar_rich_club_ponderado(
        df_rich_club, k_otimo, membros_elite,
        camada="Combinada", nome_arquivo="rich_club_ponderado.png"
    )
    visualizer.exportar_html_interativo(
        arquivo_saida="celegans_rede_interativa.html",
        notebook=False,
        influentes=influentes,
        membros_rich_club=set(membros_elite),
        graficos_png={
            "Rich-Club Ponderado": caminho_rc,
            "Matrizes de Fluxo (Química vs Elétrica)": caminho_fluxo
        }
    )


if __name__ == "__main__":
    main()