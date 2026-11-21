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
    PRESERVATION_ANALYSIS_CONFIG,
    DIFFUSION_ANALYSIS_CONFIG,
)
from connectome_builder import ConnectomeBuilder
from diffusion_analyzer import InformationDiffusionAnalyzer
from flow_analyzer import ConnectomeAnalyzer
from navigation_analyzer import NavigationAssortativityAnalyzer
from preservation_analyzer import InformationPreservationAnalyzer
from synthetic_benchmark import SyntheticNetworkBenchmark
from topology_classifier import TopologyClassifier
from visualizer import ConnectomeVisualizer
from network_metrics import NetworkMetricsAnalyzer


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

    # 4c. Ranking Global Unificado: cálculo dos três vértices mais importantes do conectoma
    #     agregando Grau Ponderado, Intermediação, Proximidade e PageRank
    ranking_global = analyzer.ranking_global_unificado(camada="Combinada")
    df_top3 = analyzer.obter_top_vertices_globais(top_n=3, camada="Combinada")

    caminho_csv_global = os.path.join(OUTPUT_DIR, "ranking_global_unificado.csv")
    ranking_global.to_csv(caminho_csv_global, index=False)
    caminho_csv_top3 = os.path.join(OUTPUT_DIR, "top3_vertices_mais_importantes.csv")
    df_top3.to_csv(caminho_csv_top3, index=False)
    print(f"Rankings globais salvos em: '{caminho_csv_global}' e '{caminho_csv_top3}'")

    # Exibição do relatório formatado dos três vértices mais importantes no terminal
    analyzer.exibir_relatorio_top_vertices(df_top3)

    # 4d. Custo de cabeamento espacial: peso sináptico × distância longitudinal entre somas
    df_custo = analyzer.calcular_custo_cabeamento()
    analyzer.exibir_resumo_custo_cabeamento(top_n=10)
    caminho_csv_custo = os.path.join(OUTPUT_DIR, "custo_cabeamento.csv")
    df_custo.to_csv(caminho_csv_custo, index=False)
    print(f"\nCusto de cabeamento salvo em: '{caminho_csv_custo}'")

    df_robustez, limiares_fc = NetworkMetricsAnalyzer.simular_robustez_percolacao(
        grafo_multi, membros_rich_club=membros_elite, passo=0.02, n_sim_aleatorio=30, seed=42
    )
    caminho_csv_rob = os.path.join(OUTPUT_DIR, "robustez_percolacao.csv")
    df_robustez.to_csv(caminho_csv_rob, index=False)

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

    # 5d. Preservação de Informação (Reciprocidade, SCC/Bow-Tie, Modularidade e Ciclos Curtos)
    preserv_analyzer = InformationPreservationAnalyzer(
        grafo_multi,
        membros_rich_club=set(membros_elite),
        **PRESERVATION_ANALYSIS_CONFIG
    )
    resultado_preserv = preserv_analyzer.executar_analise_completa()
    preserv_analyzer.exibir_relatorio_console(resultado_preserv)

    caminho_csv_preserv_res = os.path.join(OUTPUT_DIR, "preservacao_metricas_resumo.csv")
    resultado_preserv["df_resumo"].to_csv(caminho_csv_preserv_res, index=False)
    caminho_csv_bowtie = os.path.join(OUTPUT_DIR, "preservacao_bowtie_classes.csv")
    resultado_preserv["df_bowtie_classes"].to_csv(caminho_csv_bowtie, index=False)
    caminho_csv_ciclos = os.path.join(OUTPUT_DIR, "preservacao_ciclos_motifs.csv")
    resultado_preserv["df_ciclos_motifs"].to_csv(caminho_csv_ciclos, index=False)
    caminho_json_preserv = os.path.join(OUTPUT_DIR, "preservacao_informacao.json")
    with open(caminho_json_preserv, "w", encoding="utf-8") as f:
        json.dump(resultado_preserv["resumo_json"], f, indent=2, ensure_ascii=False)
    print(
        f"Preservação de informação salva em: '{caminho_csv_preserv_res}', "
        f"'{caminho_csv_bowtie}', '{caminho_csv_ciclos}' e '{caminho_json_preserv}'"
    )

    # 5e. Facilidade de Difusão de Informação (Métricas Estruturais + Simulações SI e IC/SIR)
    diff_analyzer = InformationDiffusionAnalyzer(
        grafo_multi,
        membros_rich_club=set(membros_elite),
        **DIFFUSION_ANALYSIS_CONFIG
    )
    resultado_diff = diff_analyzer.executar_analise_completa()
    diff_analyzer.exibir_relatorio_console(resultado_diff)

    caminho_csv_diff_est = os.path.join(OUTPUT_DIR, "difusao_estrutural_resumo.csv")
    resultado_diff["df_estrutural"].to_csv(caminho_csv_diff_est, index=False)
    caminho_csv_diff_si = os.path.join(OUTPUT_DIR, "difusao_si_resumo.csv")
    resultado_diff["df_si_resumo"].to_csv(caminho_csv_diff_si, index=False)
    caminho_csv_diff_ic = os.path.join(OUTPUT_DIR, "difusao_ic_curvas.csv")
    resultado_diff["df_ic_curvas"].to_csv(caminho_csv_diff_ic, index=False)
    caminho_json_diff = os.path.join(OUTPUT_DIR, "difusao_informacao.json")
    with open(caminho_json_diff, "w", encoding="utf-8") as f:
        json.dump(resultado_diff["resumo_json"], f, indent=2, ensure_ascii=False, allow_nan=False)
    print(
        f"Difusão de informação salva em: '{caminho_csv_diff_est}', "
        f"'{caminho_csv_diff_si}', '{caminho_csv_diff_ic}' e '{caminho_json_diff}'"
    )

    # 6. Visualizações salvas diretamente na pasta 'outputs/' e integradas no HTML interativo
    visualizer = ConnectomeVisualizer(grafo_multi, output_dir=OUTPUT_DIR)
    caminho_top_vertices = visualizer.plotar_top_vertices_mais_importantes(
        ranking_global, top_n=3, nome_arquivo="top_vertices_mais_importantes.png"
    )
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

    top_globais_map = dict(zip(df_top3["Neuronio"], df_top3["Posicao_Global"]))
    top_detalhes_map = {row["Neuronio"]: row.to_dict() for _, row in df_top3.iterrows()}

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
    caminho_preserv_scc = visualizer.plotar_preservacao_reciprocidade_scc(
        resultado_preserv["df_resumo"],
        resultado_preserv["mat_reciprocidade"],
        resultado_preserv["df_bowtie_classes"],
        resultado_preserv["resumo_json"]
    )
    caminho_preserv_mod = visualizer.plotar_preservacao_modularidade_ciclos(
        resultado_preserv["df_modularidade"],
        resultado_preserv["df_ciclos_motifs"],
        resultado_preserv["df_ciclos_por_classe"],
        resultado_preserv["resumo_json"]
    )
    caminho_diff_est = visualizer.plotar_difusao_estrutural(
        resultado_diff["df_estrutural"],
        resultado_diff["df_distancias"],
        resultado_diff["resumo_json"]
    )
    caminho_diff_sim = visualizer.plotar_simulacao_difusao(
        resultado_diff["df_si_curvas"],
        resultado_diff["df_si_resumo"],
        resultado_diff["df_ic_curvas"],
        resultado_diff["resumo_json"]
    )
    visualizer.exportar_html_interativo(
        arquivo_saida="celegans_rede_interativa.html",
        notebook=False,
        influentes=influentes,
        top_globais=top_globais_map,
        top_detalhes=top_detalhes_map,
        membros_rich_club=set(membros_elite),
        graficos_png={
            "Top 3 Vértices Mais Importantes": caminho_top_vertices,
            "Rich-Club Ponderado": caminho_rc,
            "Matrizes de Fluxo (Química vs Elétrica)": caminho_fluxo,
            "Comparação com Redes Sintéticas": caminho_sinteticas,
            "Distribuição de Grau (Log-Log)": caminho_dist_grau,
            "Evidências: Small-World": caminho_sw,
            "Evidências: Livre de Escala (Cauda)": caminho_sf,
            "Navegação (Greedy Routing)": caminho_nav,
            "Assortatividade e k_nn(k)": caminho_assort,
            "Preservação: Reciprocidade e SCC": caminho_preserv_scc,
            "Preservação: Modularidade e Ciclos": caminho_preserv_mod,
            "Difusão: Métricas Estruturais": caminho_diff_est,
            "Difusão: Simulação SI e Cascata (IC)": caminho_diff_sim,
        },
        tabelas_ranking={
            "🌟 Top 3 Vértices Globais": df_top3[
                ["Posicao_Global", "Neuronio", "Tipo", "Score_Global",
                 "Grau_Ponderado", "Intermediacao", "Proximidade", "PageRank",
                 "Pct_Grau_Ponderado", "Pct_Intermediacao", "Pct_Proximidade", "Pct_PageRank"]
            ],
            "🏅 Ranking Global (Top 25)": ranking_global.head(25)[
                ["Posicao_Global", "Neuronio", "Tipo", "Score_Global",
                 "Grau_Ponderado", "Forca_Entrada", "Forca_Saida",
                 "Intermediacao", "Proximidade", "PageRank"]
            ],
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

    # Exibição final da visualização dos três vértices mais importantes pós-execução completa
    print("\n" + "=" * 94)
    print("  PIPELINE EXECUTADO COM SUCESSO! RESUMO DOS VÉRTICES MAIS IMPORTANTES DO CONECTOMA:")
    print("=" * 94)
    analyzer.exibir_relatorio_top_vertices(df_top3)


if __name__ == "__main__":
    main()