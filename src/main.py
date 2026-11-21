"""Ponto de entrada principal para construção, análise e visualização do conectoma C. elegans."""

import os
import sys

if __package__:
    _src_dir = os.path.dirname(os.path.abspath(__file__))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

from config import (
    CONNECT_DATA_PATH, DIFFUSION_ANALYSIS_CONFIG, NAVIGATION_ASSORTATIVITY_CONFIG,
    OUTPUT_DIR, PRESERVATION_ANALYSIS_CONFIG, ROBUSTNESS_ANALYSIS_CONFIG,
    SYNTHETIC_BENCHMARK_CONFIG, TOPOLOGY_CLASSIFICATION_CONFIG, WIRING_NULL_CONFIG,
)
from connectome_builder import ConnectomeBuilder
from diffusion_analyzer import InformationDiffusionAnalyzer
from flow_analyzer import ConnectomeAnalyzer
from graph_utils import salvar_artefatos
from navigation_analyzer import NavigationAssortativityAnalyzer
from preservation_analyzer import InformationPreservationAnalyzer
from robustness_analyzer import RobustnessAnalyzer
from synthetic_benchmark import SyntheticNetworkBenchmark
from topology_classifier import TopologyClassifier
from visualizer import ConnectomeVisualizer


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    grafo = ConnectomeBuilder(connect_path=CONNECT_DATA_PATH).construir_grafo()
    print(f"[1/9] Conectoma construído: N={grafo.number_of_nodes()}, M={grafo.number_of_edges()}")

    analyzer = ConnectomeAnalyzer(grafo, agrupar_comando=False)
    mat_q, mat_e = analyzer.obter_matrizes_quimica_eletrica()
    ordem_classes = analyzer.obter_ordem_classes()
    df_rc, elite, k_otimo = analyzer.calcular_rich_club_ponderado(camada="Combinada", n_iter_nulo=50, seed=42)
    rc_set = set(elite)
    ranking_tipos = analyzer.ranking_estratificado_por_tipo(camada="Combinada")
    resumo_chefes = analyzer.resumo_chefes_por_tipo(camada="Combinada")
    ranking_global = analyzer.ranking_global_unificado(camada="Combinada")
    df_top3 = analyzer.obter_top_vertices_globais(top_n=3, camada="Combinada")
    df_custo = analyzer.calcular_custo_cabeamento()
    res_wiring = analyzer.avaliar_custo_cabeamento_nulo(membros_rich_club=rc_set, **WIRING_NULL_CONFIG)
    print(f"[2/9] Fluxo, rankings, Rich-Club (k>{k_otimo}, n={len(elite)}) e cabeamento concluídos.")

    res_rob = RobustnessAnalyzer(grafo, membros_rich_club=rc_set, **ROBUSTNESS_ANALYSIS_CONFIG).executar_analise_completa()
    print("[3/9] Análise de robustez concluída.")

    bench = SyntheticNetworkBenchmark(grafo, **SYNTHETIC_BENCHMARK_CONFIG)
    df_sint, graus_mod = bench.executar_benchmark()
    diag_topo = bench.calcular_diagnostico_topologico(df_sint, graus_mod)
    print(f"[4/9] Benchmark sintético concluído: sigma={diag_topo['sigma_small_world']:.2f}, omega={diag_topo['omega_small_world']:+.3f}")

    res_topo = TopologyClassifier(grafo, baselines_externos=bench.baselines_er, **TOPOLOGY_CLASSIFICATION_CONFIG).executar_diagnostico_completo()
    print("[5/9] Classificação topológica concluída.")

    res_nav = NavigationAssortativityAnalyzer(grafo, membros_rich_club=rc_set, **NAVIGATION_ASSORTATIVITY_CONFIG).executar_analise_completa()
    print("[6/9] Navegabilidade e assortatividade concluídas.")

    res_pres = InformationPreservationAnalyzer(grafo, membros_rich_club=rc_set, **PRESERVATION_ANALYSIS_CONFIG).executar_analise_completa()
    print("[7/9] Preservação de informação concluída.")

    res_diff = InformationDiffusionAnalyzer(grafo, membros_rich_club=rc_set, **DIFFUSION_ANALYSIS_CONFIG).executar_analise_completa()
    print("[8/9] Difusão de informação concluída.")

    salvar_artefatos(
        OUTPUT_DIR,
        csvs={
            "rich_club_ponderado.csv": df_rc, "ranking_estratificado.csv": ranking_tipos,
            "chefes_por_tipo.csv": resumo_chefes, "ranking_global_unificado.csv": ranking_global,
            "top3_vertices_mais_importantes.csv": df_top3, "custo_cabeamento.csv": df_custo,
            "custo_cabeamento_resumo.csv": res_wiring["df_custo_nulo"],
            "robustez_percolacao.csv": res_rob["df_percolacao"], "robustez_schneider_resumo.csv": res_rob["df_schneider"],
            "robustez_sinteticas_comparacao.csv": res_rob["df_sinteticas_resumo"],
            "robustez_funcional_direcionada.csv": res_rob["df_funcional_dir"],
            "comparacao_redes_sinteticas.csv": df_sint, "scorecard_hipoteses.csv": res_topo["scorecard_df"],
            "comparacao_modelos_cauda.csv": res_topo["modelos_cauda_df"], "ws_transicao_sweep.csv": res_topo["ws_sweep_df"],
            "navegacao_greedy_resumo.csv": res_nav["df_navegacao"], "navegacao_por_fluxo_funcional.csv": res_nav["df_nav_fluxo"],
            "assortatividade_resumo.csv": res_nav["df_assortatividade"],
            "preservacao_metricas_resumo.csv": res_pres["df_resumo"],
            "preservacao_bowtie_classes.csv": res_pres["df_bowtie_classes"],
            "preservacao_ciclos_motifs.csv": res_pres["df_ciclos_motifs"],
            "difusao_estrutural_resumo.csv": res_diff["df_estrutural"],
            "difusao_si_resumo.csv": res_diff["df_si_resumo"], "difusao_ic_curvas.csv": res_diff["df_ic_curvas"],
        },
        jsons={
            "custo_cabeamento_modelo_nulo.json": res_wiring["resumo_json"],
            "robustez_resumo.json": res_rob["resumo_json"], "diagnostico_topologico.json": diag_topo,
            "evidencias_classificacao_topologica.json": res_topo["resumo_json"],
            "navegacao_e_assortatividade.json": res_nav["resumo_json"],
            "preservacao_informacao.json": res_pres["resumo_json"], "difusao_informacao.json": res_diff["resumo_json"],
        },
    )

    vis = ConnectomeVisualizer(grafo, output_dir=OUTPUT_DIR)
    graficos = {
        "Top 3 Vértices Mais Importantes": vis.plotar_top_vertices_mais_importantes(ranking_global, top_n=3),
        "Rich-Club Ponderado": vis.plotar_rich_club_ponderado(df_rc, k_otimo, elite, camada="Combinada"),
        "Matrizes de Fluxo (Química vs Elétrica)": vis.plotar_matrizes_fluxo(mat_q, mat_e, ordem_classes),
        "Comparação com Redes Sintéticas": vis.plotar_comparacao_redes_sinteticas(df_sint, diag_topo),
        "Distribuição de Grau (Log-Log)": vis.plotar_distribuicao_grau_comparativa(graus_mod, diag_topo),
        "Evidências: Small-World": vis.plotar_evidencias_small_world(res_topo["ws_sweep_df"], res_topo["resumo_json"]["small_world"]),
        "Evidências: Livre de Escala (Cauda)": vis.plotar_evidencias_scale_free(res_topo["modelos_cauda_df"], res_topo["resumo_json"]["scale_free"]),
        "Navegação (Greedy Routing)": vis.plotar_navegacao_greedy(res_nav["df_navegacao"], res_nav["df_nav_fluxo"], res_nav["resumo_json"]["navegacao"]),
        "Assortatividade e k_nn(k)": vis.plotar_assortatividade(res_nav["df_assortatividade"], res_nav["df_knn"], res_nav["resumo_json"]["assortatividade"]),
        "Preservação: Reciprocidade e SCC": vis.plotar_preservacao_reciprocidade_scc(res_pres["df_resumo"], res_pres["mat_reciprocidade"], res_pres["df_bowtie_classes"], res_pres["resumo_json"]),
        "Preservação: Modularidade e Ciclos": vis.plotar_preservacao_modularidade_ciclos(res_pres["df_modularidade"], res_pres["df_ciclos_motifs"], res_pres["df_ciclos_por_classe"], res_pres["resumo_json"]),
        "Difusão: Métricas Estruturais": vis.plotar_difusao_estrutural(res_diff["df_estrutural"], res_diff["df_distancias"], res_diff["resumo_json"]),
        "Difusão: Simulação SI e Cascata (IC)": vis.plotar_simulacao_difusao(res_diff["df_si_curvas"], res_diff["df_si_resumo"], res_diff["df_ic_curvas"], res_diff["resumo_json"]),
        "Robustez: Percolação e Schneider (R)": vis.plotar_robustez_percolacao(res_rob["df_percolacao"], res_rob["df_schneider"], res_rob["resumo_json"]),
        "Robustez: Sintéticas e Fluxo Direcionado": vis.plotar_robustez_sinteticas_e_funcional(res_rob["df_sinteticas_curvas"], res_rob["df_sinteticas_resumo"], res_rob["df_funcional_dir"], res_rob["resumo_json"]),
        "Custo de Cabeamento (Modelo Nulo)": vis.plotar_custo_cabeamento(res_wiring),
    }
    vis.exportar_html_interativo(
        arquivo_saida="celegans_rede_interativa.html", notebook=False,
        influentes=dict(zip(resumo_chefes["Chefe"], resumo_chefes["Score_Chefe"])),
        top_globais=dict(zip(df_top3["Neuronio"], df_top3["Posicao_Global"])),
        top_detalhes={row["Neuronio"]: row.to_dict() for _, row in df_top3.iterrows()},
        membros_rich_club=rc_set, graficos_png=graficos,
        tabelas_ranking={
            "Top 3 Vértices Globais": df_top3[["Posicao_Global", "Neuronio", "Tipo", "Score_Global", "Grau_Ponderado", "Intermediacao", "Proximidade", "PageRank", "Pct_Grau_Ponderado", "Pct_Intermediacao", "Pct_Proximidade", "Pct_PageRank"]],
            "Ranking Global (Top 25)": ranking_global.head(25)[["Posicao_Global", "Neuronio", "Tipo", "Score_Global", "Grau_Ponderado", "Forca_Entrada", "Forca_Saida", "Intermediacao", "Proximidade", "PageRank"]],
            "Chefes por Tipo": resumo_chefes,
            "Top 5 por Tipo": ranking_tipos[ranking_tipos["Posicao_No_Tipo"] <= 5][["Tipo", "Posicao_No_Tipo", "Neuronio", "Grau_Ponderado", "PageRank", "Intermediacao", "Score_Chefe"]],
            "Top 20 Custo de Cabeamento": df_custo.nlargest(20, "Custo_Cabeamento")[["Neuronio_Origem", "Neuronio_Destino", "Camada", "Peso_Sinaptico", "Distancia_Soma", "Custo_Cabeamento"]],
        },
    )
    print(f"[9/9] {len(graficos)} figuras PNG e dashboard HTML gerados em '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
