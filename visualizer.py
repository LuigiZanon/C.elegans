"""
Módulo de visualização estática (Matplotlib) e interativa (Pyvis) do conectoma.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from config import (
    PALETA_CORES_FUNCIONAIS,
    CENTROS_GRUPOS_FUNCIONAIS,
    PYVIS_PHYSICS_OPTIONS
)


class ConnectomeVisualizer:
    """Gera gráficos de matrizes de fluxo e exporta a topologia interativa em HTML."""

    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    @staticmethod
    def plotar_matrizes_fluxo(
        mat_quimica: pd.DataFrame,
        mat_eletrica: pd.DataFrame,
        ordem_classes: List[str],
        salvar_caminho: Optional[str] = None
    ) -> None:
        """Renderiza os heatmaps comparativos das camadas Química e Elétrica."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

        matrizes = [
            (mat_quimica, "Camada Química (Fluxo Direcionado: Origem → Destino)", "YlOrRd"),
            (mat_eletrica, "Camada Elétrica / Gap Junctions (Acoplamento Simétrico)", "PuBu")
        ]

        for ax, (mat, titulo, cmap) in zip(axes, matrizes):
            soma_linhas = mat.sum(axis=1).replace(0, 1)
            mat_pct = mat.div(soma_linhas, axis=0) * 100

            im = ax.imshow(mat.values, cmap=cmap, aspect="auto")

            ax.set_xticks(np.arange(len(ordem_classes)))
            ax.set_yticks(np.arange(len(ordem_classes)))
            ax.set_xticklabels(ordem_classes, rotation=30, ha="right", fontsize=10)
            ax.set_yticklabels(ordem_classes, fontsize=10)
            ax.set_xlabel("Neurônio de Destino (Pós-sináptico)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Neurônio de Origem (Pré-sináptico)", fontsize=11, fontweight="bold")
            ax.set_title(titulo, fontsize=12, fontweight="bold", pad=12)

            limiar_cor = mat.values.max() / 2.0 if mat.values.size > 0 else 0
            for i in range(len(ordem_classes)):
                for j in range(len(ordem_classes)):
                    valor = mat.values[i, j]
                    pct = mat_pct.values[i, j]
                    cor_texto = "white" if valor > limiar_cor else "black"
                    ax.text(
                        j, i, f"{valor}\n({pct:.1f}%)",
                        ha="center", va="center", color=cor_texto, fontsize=9, fontweight="bold"
                    )

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Volume Sináptico Acumulado")

        plt.tight_layout()
        if salvar_caminho:
            plt.savefig(salvar_caminho, dpi=300, bbox_inches="tight")
        plt.show()

    def exportar_html_interativo(
        self,
        arquivo_saida: str = "celegans_rede_interativa.html",
        notebook: bool = False
    ) -> None:
        """Gera HTML em tela cheia com os neurônios agrupados espacialmente por tipo funcional."""
        net = Network(
            notebook=notebook,
            height="100vh",
            width="100vw",
            directed=True,
            bgcolor="#1a1a1a",
            font_color="white"
        )
        net.set_options(PYVIS_PHYSICS_OPTIONS)

        g_vis = nx.DiGraph()

        # 1. Agrupa os neurônios por tipo funcional e ordena pelo grau (hubs no centro do grupo)
        nos_por_tipo = {}
        for n, d in self.graph.nodes(data=True):
            tipo = d.get("tipo_funcional", "Desconhecido")
            nos_por_tipo.setdefault(tipo, []).append((n, d))

        angulo_aureo = np.pi * (3.0 - np.sqrt(5.0))  # ~137.5 graus para distribuição uniforme

        for tipo, lista_nos in nos_por_tipo.items():
            # Ordena decrescentemente pelo grau para posicionar os maiores hubs no centro do cluster
            lista_nos.sort(key=lambda item: self.graph.degree(item[0]), reverse=True)
            cx, cy = CENTROS_GRUPOS_FUNCIONAIS.get(tipo, (0, 0))
            id_ancora = f"__ancora_{tipo}__"

            # Cria nó âncora invisível e fixo no centro do cluster
            g_vis.add_node(
                id_ancora,
                label=" ",
                x=float(cx),
                y=float(cy),
                fixed={"x": True, "y": True},
                physics=True,
                size=0.1,
                color="rgba(0,0,0,0)",
                borderWidth=0,
                chosen=False
            )

            for idx, (n, d) in enumerate(lista_nos):
                # Posição inicial em espiral de Fibonacci ao redor do centro (cx, cy)
                raio = 52.0 * np.sqrt(idx)
                theta = idx * angulo_aureo
                pos_x = float(cx + raio * np.cos(theta))
                pos_y = float(cy + raio * np.sin(theta))

                grau_no = self.graph.degree(n)
                tamanho = 10 + (grau_no / 5.0)
                tooltip = (
                    f"Neurônio: {n}\n"
                    f"Tipo: {tipo}\n"
                    f"Grau Total: {grau_no}\n"
                    f"Região (Soma): {d.get('regiao_soma', 'N/A')}"
                )

                g_vis.add_node(
                    n,
                    label=n,
                    title=tooltip,
                    color=PALETA_CORES_FUNCIONAIS.get(tipo, "grey"),
                    size=tamanho,
                    borderWidth=1.5,
                    x=pos_x,
                    y=pos_y,
                    physics=True
                )

                # Mola invisível que prende o neurônio ao centro do seu próprio cluster de cor
                g_vis.add_edge(
                    id_ancora,
                    n,
                    physics=True,
                    color="rgba(0,0,0,0)",
                    width=0.01,
                    hoverWidth=0,
                    selectionWidth=0,
                    arrows={"to": {"enabled": False}}
                )

        # 2. Adiciona as sinapses reais com physics=False para não misturar os grupos de cores
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            peso = d.get("weight", 1)
            if g_vis.has_edge(u, v):
                g_vis[u][v]["peso_acumulado"] += peso
                g_vis[u][v]["width"] = float(0.7 + np.log1p(g_vis[u][v]["peso_acumulado"]) * 0.85)
                g_vis[u][v]["title"] += f" | {k}: {peso}"
            else:
                g_vis.add_edge(
                    u,
                    v,
                    peso_acumulado=peso,
                    width=float(0.7 + np.log1p(peso) * 0.85),
                    title=f"{u} → {v} ({k}: {peso})",
                    color="rgba(160, 160, 160, 0.22)",
                    physics=False
                )

        net.from_nx(g_vis)

        if notebook:
            net.show(arquivo_saida, notebook=True)
        else:
            net.save_graph(arquivo_saida)

        # 3. Injeta CSS Fullscreen, Legenda de Cores e controle de física sem vibração
        patch_fullscreen_e_legenda = """
        <style>
            html, body {
                margin: 0 !important;
                padding: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                overflow: hidden !important;
                background-color: #1a1a1a !important;
            }
            .card, #mynetwork {
                margin: 0 !important;
                padding: 0 !important;
                border: none !important;
                width: 100vw !important;
                height: 100vh !important;
                background-color: #1a1a1a !important;
            }
            #painel-legenda {
                position: fixed;
                top: 16px;
                left: 16px;
                z-index: 9999;
                background: rgba(30, 30, 30, 0.9);
                color: #fff;
                border: 1px solid #444;
                padding: 12px 16px;
                border-radius: 8px;
                font-family: sans-serif;
                font-size: 13px;
                line-height: 1.8;
                pointer-events: none;
            }
            .item-legenda {
                display: flex;
                align-items: center;
                gap: 8px;
            }
            .bolinha-cor {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
            }
            #btn-toggle-physics {
                position: fixed;
                top: 16px;
                right: 16px;
                z-index: 9999;
                background: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                padding: 8px 14px;
                border-radius: 6px;
                cursor: pointer;
                font-family: sans-serif;
                font-size: 13px;
            }
            #btn-toggle-physics:hover {
                background: #3c3c3c;
            }
        </style>
        <div id="painel-legenda">
            <strong>Clusters Funcionais</strong>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#4285F4;"></span>Sensorial (Esquerda)</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#FBBC05;"></span>Interneurônio (Topo)</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#EA4335;"></span>Comando / Hubs (Centro)</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#34A853;"></span>Motor (Direita)</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#9C27B0;"></span>Faríngeo (Inferior Esq.)</div>
        </div>
        <button id="btn-toggle-physics" onclick="alternarFisica()">Física: Congelada (Clique p/ Ativar)</button>
        <script type="text/javascript">
            var fisicaAtiva = false;
            if (typeof network !== 'undefined') {
                network.once("stabilizationIterationsDone", function() {
                    network.setOptions({ physics: { enabled: false } });
                    network.fit({ animation: { duration: 400 } });
                    fisicaAtiva = false;
                });
            }
            function alternarFisica() {
                fisicaAtiva = !fisicaAtiva;
                network.setOptions({ physics: { enabled: fisicaAtiva } });
                var btn = document.getElementById("btn-toggle-physics");
                btn.innerText = fisicaAtiva ? "Física: Ativa (Clique p/ Congelar)" : "Física: Congelada (Clique p/ Ativar)";
            }
        </script>
        </body>
        """

        with open(arquivo_saida, "r", encoding="utf-8") as f:
            conteudo_html = f.read()

        conteudo_html = conteudo_html.replace("</body>", patch_fullscreen_e_legenda)

        with open(arquivo_saida, "w", encoding="utf-8") as f:
            f.write(conteudo_html)

        print(f"Arquivo '{arquivo_saida}' gerado com agrupamento por tipo funcional!")