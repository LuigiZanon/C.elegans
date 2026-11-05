"""
Módulo de visualização estática (Matplotlib) e interativa (Pyvis) do conectoma.
Todos os arquivos são salvos na pasta de saída configurada sem interromper a execução.
"""

import os
from typing import Dict, List, Optional, Set
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Backend não-interativo: garante que nenhuma janela trave a execução
import matplotlib.pyplot as plt
from pyvis.network import Network
from config import (
    OUTPUT_DIR,
    PALETA_CORES_FUNCIONAIS,
    CENTROS_GRUPOS_FUNCIONAIS,
    PYVIS_PHYSICS_OPTIONS
)


class ConnectomeVisualizer:
    """Gera gráficos estáticos e exporta a topologia interativa em HTML na pasta outputs."""

    def __init__(self, graph: nx.MultiDiGraph, output_dir: str = OUTPUT_DIR):
        self.graph = graph
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _resolver_caminho(self, nome_arquivo: str) -> str:
        """Garante que o arquivo seja salvo dentro de self.output_dir."""
        if os.path.dirname(nome_arquivo):
            os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
            return nome_arquivo
        return os.path.join(self.output_dir, nome_arquivo)

    def plotar_matrizes_fluxo(
        self,
        mat_quimica: pd.DataFrame,
        mat_eletrica: pd.DataFrame,
        ordem_classes: List[str],
        nome_arquivo: str = "matrizes_fluxo.png"
    ) -> str:
        """Renderiza e salva os heatmaps comparativos das camadas Química e Elétrica sem bloquear."""
        caminho_saida = self._resolver_caminho(nome_arquivo)
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
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de matrizes de fluxo salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_rich_club_ponderado(
        self,
        df_rc: pd.DataFrame,
        k_otimo: int,
        membros_elite: List[str],
        camada: str = "Combinada",
        nome_arquivo: str = "rich_club_ponderado.png"
    ) -> str:
        """
        Plota em dois painéis:
          1. Coeficiente Rich-Club Ponderado empírico phi_w(k) vs. Modelo Nulo phi_w_null(k).
          2. Coeficiente Normalizado rho_w(k) = phi_w(k) / phi_w_null(k), destacando o regime > 1.
        Salva diretamente na pasta outputs sem interromper a execução.
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))

        k_vals = df_rc["k"].values
        phi_w = df_rc["Phi_w"].values
        phi_null = df_rc["Phi_w_Null"].values
        rho_w = df_rc["Rho_w"].values

        # Painel 1: Curva empírica vs Modelo Nulo
        ax1.plot(
            k_vals, phi_w, color="#EA4335", lw=2.5, marker="o", markersize=4,
            label=r"Conectoma Real $\phi^{w}(k)$"
        )
        ax1.plot(
            k_vals, phi_null, color="#555555", lw=2.0, linestyle="--",
            label=r"Modelo Nulo $\phi^{w}_{\mathrm{null}}(k)$ (Grau Preservado)"
        )
        ax1.axvline(
            k_otimo, color="#4285F4", linestyle=":", lw=1.8,
            label=f"Limiar Rich-Club (k > {k_otimo})"
        )
        ax1.set_title(f"Coeficiente Rich-Club Ponderado ({camada})", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Grau Total (k)", fontsize=11, fontweight="bold")
        ax1.set_ylabel(r"$\phi^{w}(k)$", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="upper left", fontsize=9.5)

        # Painel 2: Coeficiente Normalizado rho_w(k)
        ax2.plot(
            k_vals, rho_w, color="#34A853", lw=2.5, marker="s", markersize=4,
            label=r"Razão Normalizada $\rho^{w}(k) = \phi^{w}(k) / \phi^{w}_{\mathrm{null}}(k)$"
        )
        ax2.axhline(
            1.0, color="#222222", linestyle="--", lw=1.5,
            label=r"Limiar de Referência ($\rho^{w} = 1.0$)"
        )
        ax2.axvline(
            k_otimo, color="#4285F4", linestyle=":", lw=1.8,
            label=f"Núcleo Elite (k > {k_otimo}, n={len(membros_elite)})"
        )
        ax2.fill_between(
            k_vals, 1.0, rho_w, where=(rho_w > 1.0),
            color="#34A853", alpha=0.18,
            label=r"Regime Rich-Club ($\rho^{w} > 1$)"
        )

        # Anota os neurônios do clube de elite no painel direito
        elite_str = ", ".join(membros_elite[:12])
        if len(membros_elite) > 12:
            elite_str += "..."
        ax2.text(
            0.03, 0.93, f"Núcleo Rich-Club (k > {k_otimo}):\n{elite_str}",
            transform=ax2.transAxes, fontsize=9, verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8f9fa", "edgecolor": "#bbbbbb"}
        )

        ax2.set_title(f"Rich-Club Ponderado Normalizado ({camada})", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Grau Total (k)", fontsize=11, fontweight="bold")
        ax2.set_ylabel(r"$\rho^{w}(k)$", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="lower left", fontsize=9.5)

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico do Coeficiente Rich-Club Ponderado salvo em: '{caminho_saida}'")
        return caminho_saida

    def exportar_html_interativo(
        self,
        arquivo_saida: str = "celegans_rede_interativa.html",
        notebook: bool = False,
        influentes: Optional[Dict[str, float]] = None,
        membros_rich_club: Optional[Set[str]] = None
    ) -> str:
        """Gera HTML em tela cheia na pasta outputs sem abrir janelas bloqueantes."""
        caminho_saida = self._resolver_caminho(arquivo_saida)
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

        nos_por_tipo = {}
        for n, d in self.graph.nodes(data=True):
            tipo = d.get("tipo_funcional", "Desconhecido")
            nos_por_tipo.setdefault(tipo, []).append((n, d))

        angulo_aureo = np.pi * (3.0 - np.sqrt(5.0))

        for tipo, lista_nos in nos_por_tipo.items():
            lista_nos.sort(key=lambda item: self.graph.degree(item[0]), reverse=True)
            cx, cy = CENTROS_GRUPOS_FUNCIONAIS.get(tipo, (0, 0))
            id_ancora = f"__ancora_{tipo}__"

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
                raio = 52.0 * np.sqrt(idx)
                theta = idx * angulo_aureo
                pos_x = float(cx + raio * np.cos(theta))
                pos_y = float(cy + raio * np.sin(theta))

                grau_no = self.graph.degree(n)
                tamanho = 10 + (grau_no / 5.0)
                eh_rc = membros_rich_club is not None and n in membros_rich_club
                tooltip = (
                    f"Neurônio: {n}\n"
                    f"Tipo: {tipo}\n"
                    f"Grau Total: {grau_no}\n"
                    f"Região (Soma): {d.get('regiao_soma', 'N/A')}\n"
                    f"Membro Rich-Club: {'Sim' if eh_rc else 'Não'}"
                )
                eh_influente = influentes is not None and n in influentes
                if eh_influente:
                    tooltip += (
                        f"\nMais influente da classe: {tipo}"
                        f"\nEscore de influência: {influentes[n]:.3f}"
                    )

                g_vis.add_node(
                    n,
                    label=f"★ {n}" if eh_influente else n,
                    title=tooltip,
                    color=PALETA_CORES_FUNCIONAIS.get(tipo, "grey"),
                    size=tamanho * (1.5 if eh_influente else 1.0),
                    shape="star" if eh_influente else "dot",
                    borderWidth=4 if (eh_influente or eh_rc) else 1.5,
                    x=pos_x,
                    y=pos_y,
                    physics=True
                )

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
        net.save_graph(caminho_saida)

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
            <div class="item-legenda">★ Mais influente da classe</div>
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

        with open(caminho_saida, "r", encoding="utf-8") as f:
            conteudo_html = f.read()

        conteudo_html = conteudo_html.replace("</body>", patch_fullscreen_e_legenda)

        with open(caminho_saida, "w", encoding="utf-8") as f:
            f.write(conteudo_html)

        print(f"Grafo interativo HTML salvo em: '{caminho_saida}'")
        return caminho_saida