"""
Ponto de entrada principal (Main) para construção, análise e visualização do conectoma C. elegans.
Todas as figuras e o HTML interativo multi-layout são salvos automaticamente na pasta 'outputs/'.
"""

import os
import json
from config import (
    OUTPUT_DIR,
    SYNTHETIC_BENCHMARK_CONFIG,
    TOPOLOGY_CLASSIFICATION_CONFIG,
    NAVIGATION_ASSORTATIVITY_CONFIG,
)
from connectome_builder import ConnectomeBuilder
from flow_analyzer import ConnectomeAnalyzer
from navigation_analyzer import NavigationAssortativityAnalyzer
from synthetic_benchmark import SyntheticNetworkBenchmark
from topology_classifier import TopologyClassifier
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

    # 4b. Ranking estratificado por tipo funcional: métricas globais na camada combinada,
    #     com um "chefe" único por tipo (score composto dos percentis)
    ranking_tipos = analyzer.ranking_estratificado_por_tipo(camada="Combinada")
    resumo_chefes = analyzer.resumo_chefes_por_tipo(camada="Combinada")

    print("\n=== RANKING ESTRATIFICADO POR TIPO FUNCIONAL (MÉTRICAS GLOBAIS, CAMADA COMBINADA) ===")
    colunas_top = [
        "Tipo", "Neuronio", "Posicao_No_Tipo", "Grau_Ponderado",
        "PageRank", "Intermediacao", "Score_Chefe"
    ]
    print(ranking_tipos[ranking_tipos["Posicao_No_Tipo"] <= 5][colunas_top].to_string(index=False))

    print("\n=== CHEFE DE CADA CATEGORIA (SCORE COMPOSTO + LÍDERES POR MÉTRICA) ===")
    print(resumo_chefes.to_string(index=False))

    caminho_csv_ranking = os.path.join(OUTPUT_DIR, "ranking_estratificado.csv")
    ranking_tipos.to_csv(caminho_csv_ranking, index=False)
    caminho_csv_chefes = os.path.join(OUTPUT_DIR, "chefes_por_tipo.csv")
    resumo_chefes.to_csv(caminho_csv_chefes, index=False)
    print(f"\nTabelas salvas em: '{caminho_csv_ranking}' e '{caminho_csv_chefes}'")

    # 4c. Custo de cabeamento espacial: peso sináptico × distância longitudinal entre somas
    df_custo = analyzer.calcular_custo_cabeamento()
    analyzer.exibir_resumo_custo_cabeamento(top_n=10)
    caminho_csv_custo = os.path.join(OUTPUT_DIR, "custo_cabeamento.csv")
    df_custo.to_csv(caminho_csv_custo, index=False)
    print(f"\nCusto de cabeamento salvo em: '{caminho_csv_custo}'")

    # Os chefes de cada tipo substituem os destaques da camada química no destaque do HTML.
    influentes = dict(zip(resumo_chefes["Chefe"], resumo_chefes["Score_Chefe"]))

    # 5. Benchmark comparativo com redes sintéticas equivalentes (ER, WS e BA) via Monte Carlo
    benchmark = SyntheticNetworkBenchmark(grafo_multi, **SYNTHETIC_BENCHMARK_CONFIG)
    df_sinteticas, graus_por_modelo = benchmark.executar_benchmark()
    diagnostico_topo = benchmark.calcular_diagnostico_topologico(
        df_sinteticas, graus_por_modelo
    )
    benchmark.exibir_relatorio_console(df_sinteticas, diagnostico_topo)

    caminho_csv_sint = os.path.join(OUTPUT_DIR, "comparacao_redes_sinteticas.csv")
    df_sinteticas.to_csv(caminho_csv_sint, index=False)
    caminho_json_diag = os.path.join(OUTPUT_DIR, "diagnostico_topologico.json")
    with open(caminho_json_diag, "w", encoding="utf-8") as f:
        json.dump(diagnostico_topo, f, indent=2, ensure_ascii=False)
    print(f"Benchmark salvo em: '{caminho_csv_sint}' e '{caminho_json_diag}'")

    # 5b. Diagnóstico topológico formal com evidências (Aleatório vs. Small-World vs. Scale-Free)
    classifier = TopologyClassifier(grafo_multi, **TOPOLOGY_CLASSIFICATION_CONFIG)
    resultado_topo = classifier.executar_diagnostico_completo()
    classifier.exibir_relatorio_console(resultado_topo)

    caminho_scorecard = os.path.join(OUTPUT_DIR, "scorecard_hipoteses.csv")
    resultado_topo["scorecard_df"].to_csv(caminho_scorecard, index=False)
    caminho_modelos_cauda = os.path.join(OUTPUT_DIR, "comparacao_modelos_cauda.csv")
    resultado_topo["modelos_cauda_df"].to_csv(caminho_modelos_cauda, index=False)
    caminho_ws_sweep = os.path.join(OUTPUT_DIR, "ws_transicao_sweep.csv")
    resultado_topo["ws_sweep_df"].to_csv(caminho_ws_sweep, index=False)
    caminho_evidencias_json = os.path.join(OUTPUT_DIR, "evidencias_classificacao_topologica.json")
    with open(caminho_evidencias_json, "w", encoding="utf-8") as f:
        json.dump(resultado_topo["resumo_json"], f, indent=2, ensure_ascii=False)
    print(
        f"Evidências de classificação topológica salvas em: '{caminho_scorecard}', "
        f"'{caminho_modelos_cauda}', '{caminho_ws_sweep}' e '{caminho_evidencias_json}'"
    )

    # 5c. Navegabilidade (Roteamento Guloso / Greedy Routing) e Assortatividade
    nav_analyzer = NavigationAssortativityAnalyzer(
        grafo_multi,
        membros_rich_club=set(membros_elite),
        **NAVIGATION_ASSORTATIVITY_CONFIG
    )
    resultado_nav = nav_analyzer.executar_analise_completa()
    nav_analyzer.exibir_relatorio_console(resultado_nav)

    caminho_csv_nav = os.path.join(OUTPUT_DIR, "navegacao_greedy_resumo.csv")
    resultado_nav["df_navegacao"].to_csv(caminho_csv_nav, index=False)
    caminho_csv_nav_fluxo = os.path.join(OUTPUT_DIR, "navegacao_por_fluxo_funcional.csv")
    resultado_nav["df_nav_fluxo"].to_csv(caminho_csv_nav_fluxo, index=False)
    caminho_csv_assort = os.path.join(OUTPUT_DIR, "assortatividade_resumo.csv")
    resultado_nav["df_assortatividade"].to_csv(caminho_csv_assort, index=False)
    caminho_json_nav = os.path.join(OUTPUT_DIR, "navegacao_e_assortatividade.json")
    with open(caminho_json_nav, "w", encoding="utf-8") as f:
        json.dump(resultado_nav["resumo_json"], f, indent=2, ensure_ascii=False)
    print(
        f"Navegação e assortatividade salvas em: '{caminho_csv_nav}', "
        f"'{caminho_csv_nav_fluxo}', '{caminho_csv_assort}' e '{caminho_json_nav}'"
    )

    # 6. Visualizações salvas diretamente na pasta 'outputs/' e integradas no HTML interativo
    visualizer = ConnectomeVisualizer(grafo_multi, output_dir=OUTPUT_DIR)
    caminho_fluxo = visualizer.plotar_matrizes_fluxo(
        mat_quimica, mat_eletrica, ordem_classes, nome_arquivo="matrizes_fluxo.png"
    )
    caminho_rc = visualizer.plotar_rich_club_ponderado(
        df_rich_club, k_otimo, membros_elite,
        camada="Combinada", nome_arquivo="rich_club_ponderado.png"
    )
    caminho_sinteticas = visualizer.plotar_comparacao_redes_sinteticas(
        df_sinteticas, diagnostico_topo, nome_arquivo="comparacao_redes_sinteticas.png"
    )
    caminho_dist_grau = visualizer.plotar_distribuicao_grau_comparativa(
        graus_por_modelo, diagnostico_topo, nome_arquivo="distribuicao_grau_comparativa.png"
    )
    caminho_sw = visualizer.plotar_evidencias_small_world(
        resultado_topo["ws_sweep_df"], resultado_topo["resumo_json"]["small_world"]
    )
    caminho_sf = visualizer.plotar_evidencias_scale_free(
        resultado_topo["modelos_cauda_df"], resultado_topo["resumo_json"]["scale_free"]
    )
    caminho_nav = visualizer.plotar_navegacao_greedy(
        resultado_nav["df_navegacao"],
        resultado_nav["df_nav_fluxo"],
        resultado_nav["resumo_json"]["navegacao"]
    )
    caminho_assort = visualizer.plotar_assortatividade(
        resultado_nav["df_assortatividade"],
        resultado_nav["df_knn"],
        resultado_nav["resumo_json"]["assortatividade"]
    )
    visualizer.exportar_html_interativo(
        arquivo_saida="celegans_rede_interativa.html",
        notebook=False,
        influentes=influentes,
        membros_rich_club=set(membros_elite),
        graficos_png={
            "Rich-Club Ponderado": caminho_rc,
            "Matrizes de Fluxo (Química vs Elétrica)": caminho_fluxo,
            "Comparação com Redes Sintéticas": caminho_sinteticas,
            "Distribuição de Grau (Log-Log)": caminho_dist_grau,
            "Evidências: Small-World": caminho_sw,
            "Evidências: Livre de Escala (Cauda)": caminho_sf,
            "Navegação (Greedy Routing)": caminho_nav,
            "Assortatividade e k_nn(k)": caminho_assort,
        },
        tabelas_ranking={
            "🏆 Chefes por Tipo": resumo_chefes,
            "Top 5 por Tipo": ranking_tipos[ranking_tipos["Posicao_No_Tipo"] <= 5][
                ["Tipo", "Posicao_No_Tipo", "Neuronio", "Grau_Ponderado",
                 "PageRank", "Intermediacao", "Score_Chefe"]
            ],
            "💰 Top 20 Custo de Cabeamento": df_custo.nlargest(20, "Custo_Cabeamento")[
                ["Neuronio_Origem", "Neuronio_Destino", "Camada",
                 "Peso_Sinaptico", "Distancia_Soma", "Custo_Cabeamento"]
            ]
        }
    )


if __name__ == "__main__":
    main()