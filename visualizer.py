"""
Módulo de visualização estática (Matplotlib) e interativa (Pyvis) do conectoma.
Inclui caixa de seleção com 5 modos de exibição (layouts topológicos/biológicos)
e filtro de camadas do multigrafo (Química, Elétrica e Multiplex).
"""

import os
import json
import base64
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Backend não-interativo: garante que nenhuma janela trave a execução
import matplotlib.pyplot as plt
from pyvis.network import Network
from config import (
    OUTPUT_DIR,
    ORDEM_CLASSES_PADRAO,
    PALETA_CORES_FUNCIONAIS,
    CENTROS_GRUPOS_FUNCIONAIS,
    PYVIS_PHYSICS_OPTIONS
)


class ConnectomeVisualizer:
    """Gera gráficos estáticos e exporta a topologia interativa multi-layout em HTML."""

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

    @staticmethod
    def _imagem_para_base64(caminho_imagem: str) -> str:
        """Converte um arquivo PNG para string Base64 para embutir no modal HTML."""
        if not caminho_imagem or not os.path.exists(caminho_imagem):
            return ""
        with open(caminho_imagem, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _calcular_forca_por_camada(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calcula a força sináptica total (entrada + saída) de cada nó nas camadas Química e Elétrica."""
        forca_q = {n: 0.0 for n in self.graph.nodes()}
        forca_e = {n: 0.0 for n in self.graph.nodes()}

        for u, v, k, d in self.graph.edges(keys=True, data=True):
            peso = float(d.get("weight", 1.0))
            camada = d.get("synapse_type", k)
            if camada == "Eletrica":
                forca_e[u] = forca_e.get(u, 0.0) + peso
                forca_e[v] = forca_e.get(v, 0.0) + peso
            else:
                forca_q[u] = forca_q.get(u, 0.0) + peso
                forca_q[v] = forca_q.get(v, 0.0) + peso

        return forca_q, forca_e

    def _calcular_todos_layouts(
        self,
        membros_rich_club: Set[str]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Pré-calcula as coordenadas (x, y) livres de sobreposição para os 5 modos de exibição:
          1. clusters: Ilhas por Papel Funcional (Espiral de Fibonacci)
          2. anatomico: Posição Longitudinal Real no Corpo (Cabeça -> Meio -> Cauda)
          3. hierarquico: Estágios Direcionais de Fluxo (Sensorial -> Interneurônio -> Comando -> Motor)
          4. rich_club: Anéis Concêntricos Núcleo-Periferia (Core-Periphery Rich-Club)
          5. multiplex: Polaridade Sináptica (Domínio Químico vs. Multiplex vs. Elétrico)
        """
        layouts: Dict[str, Dict[str, Dict[str, float]]] = {
            "clusters": {},
            "anatomico": {},
            "hierarquico": {},
            "rich_club": {},
            "multiplex": {}
        }

        angulo_aureo = np.pi * (3.0 - np.sqrt(5.0))
        nos_por_tipo: Dict[str, List[Tuple[str, dict]]] = {}
        for n, d in self.graph.nodes(data=True):
            tipo = d.get("tipo_funcional", "Desconhecido")
            nos_por_tipo.setdefault(tipo, []).append((n, d))

        for tipo in nos_por_tipo:
            nos_por_tipo[tipo].sort(key=lambda item: self.graph.degree(item[0]), reverse=True)

        # -------------------------------------------------------------------------
        # MODO 1: Clusters Funcionais (Ilhas por Papel Biológico)
        # -------------------------------------------------------------------------
        for tipo, lista_nos in nos_por_tipo.items():
            cx, cy = CENTROS_GRUPOS_FUNCIONAIS.get(tipo, (0, 0))
            for idx, (n, _) in enumerate(lista_nos):
                raio = 52.0 * np.sqrt(idx)
                theta = idx * angulo_aureo
                layouts["clusters"][n] = {
                    "x": round(float(cx + raio * np.cos(theta)), 2),
                    "y": round(float(cy + raio * np.sin(theta)), 2)
                }

        # -------------------------------------------------------------------------
        # MODO 2: Anatomia Longitudinal do Verme (Cabeça -> Corpo Médio -> Cauda)
        # -------------------------------------------------------------------------
        faixas_y_anatomicas = {
            "Sensorial": -430.0,
            "Faríngeo": -200.0,
            "Interneurônio": 20.0,
            "Interneurônio (Comando)": 230.0,
            "Motor": 450.0,
            "Desconhecido": 600.0
        }
        todos_nos_ordenados_pos = sorted(
            self.graph.nodes(data=True),
            key=lambda item: (float(item[1].get("soma_pos") or 0.5), item[0])
        )
        total_n = max(len(todos_nos_ordenados_pos) - 1, 1)
        contadores_faixa = {k: 0 for k in faixas_y_anatomicas}

        for rank_idx, (n, d) in enumerate(todos_nos_ordenados_pos):
            soma_pos = float(d.get("soma_pos") or 0.5)
            soma_pos = min(max(soma_pos, 0.0), 1.0)
            frac_rank = rank_idx / total_n
            # Combina a coordenada física soma_pos com o rank longitudinal para evitar colisão no gânglio cefálico
            pos_hibrida = 0.45 * soma_pos + 0.55 * frac_rank
            pos_x = -1350.0 + pos_hibrida * 2700.0

            tipo = d.get("tipo_funcional", "Desconhecido")
            y_base = faixas_y_anatomicas.get(tipo, 0.0)
            idx_local = contadores_faixa.get(tipo, 0)
            contadores_faixa[tipo] = idx_local + 1
            deslocamento_y = ((idx_local % 5) - 2) * 34.0

            layouts["anatomico"][n] = {
                "x": round(float(pos_x), 2),
                "y": round(float(y_base + deslocamento_y), 2)
            }

        # -------------------------------------------------------------------------
        # MODO 3: Hierarquia Direcional de Fluxo (Feedforward: Sensorial -> Motor)
        # -------------------------------------------------------------------------
        config_colunas = {
            "Faríngeo": (-1280.0, 2),
            "Sensorial": (-680.0, 3),
            "Interneurônio": (0.0, 3),
            "Interneurônio (Comando)": (620.0, 1),
            "Motor": (1200.0, 4),
            "Desconhecido": (1500.0, 1)
        }
        for tipo, lista_nos in nos_por_tipo.items():
            x_col, num_subcolunas = config_colunas.get(tipo, (0.0, 2))
            linhas_totais = int(np.ceil(len(lista_nos) / num_subcolunas))
            espacamento_y = 54.0 if linhas_totais <= 30 else 44.0
            espacamento_x = 92.0

            for idx, (n, _) in enumerate(lista_nos):
                linha = idx // num_subcolunas
                subcol = idx % num_subcolunas
                offset_x = (subcol - (num_subcolunas - 1) / 2.0) * espacamento_x
                offset_y = (linha - (linhas_totais - 1) / 2.0) * espacamento_y
                layouts["hierarquico"][n] = {
                    "x": round(float(x_col + offset_x), 2),
                    "y": round(float(offset_y), 2)
                }

        # -------------------------------------------------------------------------
        # MODO 4: Arquitetura Concêntrica Rich-Club (Núcleo-Periferia)
        # -------------------------------------------------------------------------
        todos_por_grau = sorted(
            self.graph.nodes(data=True),
            key=lambda item: self.graph.degree(item[0]),
            reverse=True
        )
        elite_rc = [item for item in todos_por_grau if item[0] in membros_rich_club]
        if not elite_rc:
            elite_rc = todos_por_grau[:10]
        set_elite = {item[0] for item in elite_rc}

        restantes = [item for item in todos_por_grau if item[0] not in set_elite]
        n_rest = len(restantes)
        corte_anel1 = max(int(n_rest * 0.22), 1)
        corte_anel2 = max(int(n_rest * 0.58), corte_anel1 + 1)

        anel_1 = restantes[:corte_anel1]
        anel_2 = restantes[corte_anel1:corte_anel2]
        anel_3 = restantes[corte_anel2:]

        # Ordena anéis externos por classe funcional para formar setores cromáticos limpos
        ordem_setor = {c: i for i, c in enumerate(ORDEM_CLASSES_PADRAO)}
        anel_2.sort(key=lambda item: (ordem_setor.get(item[1].get("tipo_funcional", ""), 99), -self.graph.degree(item[0])))
        anel_3.sort(key=lambda item: (ordem_setor.get(item[1].get("tipo_funcional", ""), 99), -self.graph.degree(item[0])))

        aneis = [
            (elite_rc, 115.0, 0.0),
            (anel_1, 330.0, 0.0),
            (anel_2, 580.0, 28.0),
            (anel_3, 860.0, 38.0)
        ]
        for lista_anel, raio_base, alternancia_r in aneis:
            qtd = max(len(lista_anel), 1)
            for idx, (n, _) in enumerate(lista_anel):
                ang = (2.0 * np.pi * idx) / qtd - (np.pi / 2.0)
                r_efetivo = raio_base + ((idx % 2) * alternancia_r)
                layouts["rich_club"][n] = {
                    "x": round(float(r_efetivo * np.cos(ang)), 2),
                    "y": round(float(r_efetivo * np.sin(ang)), 2)
                }

        # -------------------------------------------------------------------------
        # MODO 5: Polaridade Sináptica (Camada Química vs. Elétrica / Multiplex)
        # -------------------------------------------------------------------------
        forca_q, forca_e = self._calcular_forca_por_camada()
        faixas_x_multiplex = {
            "Sensorial": -980.0,
            "Interneurônio": -320.0,
            "Interneurônio (Comando)": 220.0,
            "Motor": 820.0,
            "Faríngeo": -1380.0,
            "Desconhecido": 1250.0
        }
        cont_multiplex = {k: 0 for k in faixas_x_multiplex}

        for tipo, lista_nos in nos_por_tipo.items():
            x_centro = faixas_x_multiplex.get(tipo, 0.0)
            for idx, (n, _) in enumerate(lista_nos):
                fq = forca_q.get(n, 0.0)
                fe = forca_e.get(n, 0.0)
                soma_qe = fq + fe
                # Índice de polaridade: -1 (100% Químico / Topo) a +1 (100% Elétrico / Base); 0 = Multiplex
                indice_pol = (fe - fq) / soma_qe if soma_qe > 0 else 0.0
                pos_y = indice_pol * 620.0 + ((idx % 3) - 1) * 24.0

                col_local = (idx % 5) - 2
                pos_x = x_centro + col_local * 72.0 + ((idx // 5) % 2) * 22.0
                cont_multiplex[tipo] += 1

                layouts["multiplex"][n] = {
                    "x": round(float(pos_x), 2),
                    "y": round(float(pos_y), 2)
                }

        return layouts

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
        Plota em dois painéis o Coeficiente Rich-Club Ponderado empírico e normalizado.
        Salva diretamente na pasta outputs sem interromper a execução.
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))

        k_vals = df_rc["k"].values
        phi_w = df_rc["Phi_w"].values
        phi_null = df_rc["Phi_w_Null"].values
        rho_w = df_rc["Rho_w"].values

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
        membros_rich_club: Optional[Set[str]] = None,
        graficos_png: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Gera HTML em tela cheia na pasta outputs com caixas de seleção (<select>) para:
          - Alternar entre 5 modos de exibição (layouts topológicos/anatômicos)
          - Filtrar/colorir as camadas do multigrafo (Química, Elétrica e Multiplex)
          - Visualizar os gráficos PNG de métricas em modal integrado
        """
        caminho_saida = self._resolver_caminho(arquivo_saida)
        rc_set = set(membros_rich_club) if membros_rich_club else set()
        layouts_precalculados = self._calcular_todos_layouts(rc_set)
        forca_q, forca_e = self._calcular_forca_por_camada()

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

            for n, d in lista_nos:
                coord_ini = layouts_precalculados["clusters"][n]
                grau_no = self.graph.degree(n)
                tamanho = 10 + (grau_no / 5.0)
                eh_rc = n in rc_set
                fq = int(forca_q.get(n, 0))
                fe = int(forca_e.get(n, 0))
                soma_p = d.get("soma_pos", 0.0)
                soma_str = f"{float(soma_p):.2f}" if soma_p is not None else "N/A"

                tooltip = (
                    f"Neurônio: {n}\n"
                    f"Tipo: {tipo}\n"
                    f"Grau Total: {grau_no} (Força Quím: {fq} | Elét: {fe})\n"
                    f"Posição Soma: {soma_str} | Região: {d.get('regiao_soma', 'N/A')}\n"
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
                    x=float(coord_ini["x"]),
                    y=float(coord_ini["y"]),
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

        # Consolida arestas registrando separadamente os pesos Químico e Elétrico para o filtro dinâmico JS
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            peso = int(d.get("weight", 1))
            camada = d.get("synapse_type", k)
            pq = peso if camada == "Quimica" else 0
            pe = peso if camada == "Eletrica" else 0

            if g_vis.has_edge(u, v):
                g_vis[u][v]["peso_acumulado"] += peso
                g_vis[u][v]["peso_quimico"] += pq
                g_vis[u][v]["peso_eletrico"] += pe
                g_vis[u][v]["width"] = float(0.7 + np.log1p(g_vis[u][v]["peso_acumulado"]) * 0.85)
                g_vis[u][v]["title"] += f" | {camada}: {peso}"
            else:
                g_vis.add_edge(
                    u,
                    v,
                    peso_acumulado=peso,
                    peso_quimico=pq,
                    peso_eletrico=pe,
                    width=float(0.7 + np.log1p(peso) * 0.85),
                    title=f"{u} → {v} ({camada}: {peso})",
                    color="rgba(160, 160, 160, 0.22)",
                    physics=False
                )

        net.from_nx(g_vis)
        net.save_graph(caminho_saida)

        # Prepara abas do modal caso graficos_png tenha sido fornecido
        botoes_abas = ""
        conteudos_abas = ""
        if graficos_png:
            for idx, (titulo_aba, caminho_img) in enumerate(graficos_png.items()):
                b64 = self._imagem_para_base64(caminho_img)
                if not b64:
                    continue
                ativo_cls = "aba-ativa" if idx == 0 else ""
                display_stl = "block" if idx == 0 else "none"
                id_aba = f"aba_grafico_{idx}"
                botoes_abas += (
                    f'<button class="btn-aba {ativo_cls}" '
                    f'onclick="trocarAba(event, \'{id_aba}\')">{titulo_aba}</button>'
                )
                conteudos_abas += (
                    f'<div id="{id_aba}" class="conteudo-aba" style="display:{display_stl};">'
                    f'<img src="data:image/png;base64,{b64}" alt="{titulo_aba}" />'
                    f'</div>'
                )

        layouts_json = json.dumps(layouts_precalculados)
        rc_json = json.dumps(list(rc_set))
        btn_modal_html = (
            '<button class="btn-topo btn-destaque" onclick="alternarModalMetricas()">📊 Gráficos de Métricas</button>'
            if botoes_abas else ""
        )

        patch_interface = f"""
        <style>
            html, body {{
                margin: 0 !important;
                padding: 0 !important;
                width: 100vw !important;
                height: 100vh !important;
                overflow: hidden !important;
                background-color: #1a1a1a !important;
            }}
            .card, #mynetwork {{
                margin: 0 !important;
                padding: 0 !important;
                border: none !important;
                width: 100vw !important;
                height: 100vh !important;
                background-color: #1a1a1a !important;
            }}
            #painel-legenda {{
                position: fixed;
                top: 16px;
                left: 16px;
                z-index: 9000;
                background: rgba(26, 26, 26, 0.94);
                color: #fff;
                border: 1px solid #444;
                padding: 14px 18px;
                border-radius: 8px;
                font-family: sans-serif;
                font-size: 13px;
                line-height: 1.7;
                max-width: 320px;
                pointer-events: none;
                box-shadow: 0 6px 20px rgba(0,0,0,0.45);
            }}
            .item-legenda {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .bolinha-cor {{
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
                flex-shrink: 0;
            }}
            .linha-cor {{
                width: 18px;
                height: 3px;
                display: inline-block;
                flex-shrink: 0;
            }}
            .descricao-modo {{
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #3a3a3a;
                font-size: 12px;
                color: #cccccc;
                line-height: 1.45;
            }}
            #barra-controles-topo {{
                position: fixed;
                top: 16px;
                right: 16px;
                z-index: 9000;
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 10px;
                background: rgba(26, 26, 26, 0.92);
                padding: 10px 14px;
                border-radius: 8px;
                border: 1px solid #444;
                box-shadow: 0 6px 20px rgba(0,0,0,0.45);
                font-family: sans-serif;
            }}
            .grupo-select {{
                display: flex;
                flex-direction: column;
                gap: 3px;
            }}
            .label-select {{
                color: #aaa;
                font-size: 10.5px;
                font-weight: bold;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .select-custom {{
                background: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                padding: 7px 10px;
                border-radius: 6px;
                font-size: 13px;
                font-family: sans-serif;
                cursor: pointer;
                outline: none;
            }}
            .select-custom:hover, .select-custom:focus {{
                border-color: #4285F4;
                background: #333333;
            }}
            .btn-topo {{
                background: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555;
                padding: 8px 13px;
                border-radius: 6px;
                cursor: pointer;
                font-family: sans-serif;
                font-size: 13px;
                font-weight: 600;
                align-self: flex-end;
                transition: background 0.2s;
            }}
            .btn-topo:hover {{
                background: #3c3c3c;
            }}
            .btn-destaque {{
                background: #1e4d2b;
                border-color: #34A853;
            }}
            .btn-destaque:hover {{
                background: #286639;
            }}
            #modal-metricas {{
                display: none;
                position: fixed;
                inset: 0;
                z-index: 9999;
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(4px);
                align-items: center;
                justify-content: center;
            }}
            .modal-caixa {{
                background: #222;
                border: 1px solid #444;
                border-radius: 10px;
                width: 88vw;
                max-width: 1350px;
                max-height: 90vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                box-shadow: 0 12px 32px rgba(0,0,0,0.65);
                font-family: sans-serif;
            }}
            .modal-cabecalho {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 20px;
                background: #181818;
                border-bottom: 1px solid #333;
            }}
            .abas-container {{ display: flex; gap: 8px; }}
            .btn-aba {{
                background: #2c2c2c;
                color: #ccc;
                border: 1px solid #444;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 13px;
                font-weight: bold;
            }}
            .btn-aba.aba-ativa {{
                background: #4285F4;
                color: #fff;
                border-color: #4285F4;
            }}
            .btn-fechar-modal {{
                background: #EA4335;
                color: #fff;
                border: none;
                padding: 7px 14px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
            }}
            .modal-corpo {{
                padding: 18px;
                overflow-y: auto;
                text-align: center;
                background: #fff;
            }}
            .conteudo-aba img {{
                max-width: 100%;
                max-height: 76vh;
                object-fit: contain;
                border-radius: 4px;
            }}
        </style>

        <div id="painel-legenda">
            <strong id="titulo-legenda">1. Clusters Funcionais</strong>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#4285F4;"></span>Sensorial</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#FBBC05;"></span>Interneurônio</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#EA4335;"></span>Comando / Hubs</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#34A853;"></span>Motor</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#9C27B0;"></span>Faríngeo</div>
            <div class="item-legenda">★ Mais influente da classe</div>
            <div id="legenda-arestas" style="display:none; margin-top:6px; padding-top:6px; border-top:1px solid #3a3a3a;">
                <div class="item-legenda"><span class="linha-cor" style="background:#FF9800;"></span>Sinapse Química (S/Sp)</div>
                <div class="item-legenda"><span class="linha-cor" style="background:#00BCD4;"></span>Junção Elétrica (EJ)</div>
                <div class="item-legenda"><span class="linha-cor" style="background:#FF4081;"></span>Multiplex (Quím + Elét)</div>
            </div>
            <div id="texto-explicativo-modo" class="descricao-modo">
                Agrupa espacialmente os neurônios em 5 ilhas segundo seu papel biológico, posicionando os maiores hubs no centro de cada grupo.
            </div>
        </div>

        <div id="barra-controles-topo">
            <div class="grupo-select">
                <span class="label-select">Modo de Exibição (Layout)</span>
                <select id="select-modo-exibicao" class="select-custom" onchange="mudarModoExibicao(this.value)">
                    <option value="clusters">1. Clusters Funcionais (Ilhas por Papel)</option>
                    <option value="anatomico">2. Anatomia Longitudinal (Cabeça → Corpo → Cauda)</option>
                    <option value="hierarquico">3. Hierarquia de Fluxo (Sensorial → Comando → Motor)</option>
                    <option value="rich_club">4. Anéis Concêntricos Rich-Club (Núcleo-Periferia)</option>
                    <option value="multiplex">5. Polaridade Sináptica (Química vs. Elétrica)</option>
                </select>
            </div>

            <div class="grupo-select">
                <span class="label-select">Camada do Multigrafo</span>
                <select id="select-camada-arestas" class="select-custom" onchange="filtrarCamadaSinaptica(this.value)">
                    <option value="padrao">Todas (Padrão Consolidado)</option>
                    <option value="colorir_camadas">Destacar Cores (Química / Elétrica / Multiplex)</option>
                    <option value="quimica">Somente Sinapses Químicas (Direcionadas)</option>
                    <option value="eletrica">Somente Junções Elétricas (Gap Junctions)</option>
                    <option value="multiplex_only">Somente Pares Multiplex (Química + Elétrica)</option>
                </select>
            </div>

            {btn_modal_html}
            <button id="btn-rc-focus" class="btn-topo" onclick="destacarRichClub()">🎯 Focar Rich-Club</button>
            <button id="btn-toggle-physics" class="btn-topo" onclick="alternarFisica()">Física: Congelada</button>
        </div>

        <div id="modal-metricas" onclick="fecharModalFora(event)">
            <div class="modal-caixa">
                <div class="modal-cabecalho">
                    <div class="abas-container">{botoes_abas}</div>
                    <button class="btn-fechar-modal" onclick="alternarModalMetricas()">✕ Fechar (ESC)</button>
                </div>
                <div class="modal-corpo">{conteudos_abas}</div>
            </div>
        </div>

        <script type="text/javascript">
            var fisicaAtiva = false;
            var rcDestacado = false;
            var coordenadasLayouts = {layouts_json};
            var membrosRichClub = new Set({rc_json});

            var descricoesModos = {{
                "clusters": {{
                    titulo: "1. Clusters Funcionais",
                    texto: "Agrupa os neurônios em 5 ilhas por função biológica (Sensorial, Interneurônio, Comando, Motor e Faríngeo), com hubs no centro."
                }},
                "anatomico": {{
                    titulo: "2. Anatomia Longitudinal Real",
                    texto: "Eixo X: Posição física do soma no corpo do verme (Esquerda = Cabeça [H] → Centro = Corpo [M] → Direita = Cauda [T]). Eixo Y: Camada funcional."
                }},
                "hierarquico": {{
                    titulo: "3. Hierarquia Direcional de Fluxo",
                    texto: "Organiza a rede em colunas da esquerda para a direita seguindo o processamento neural: Faríngeo / Sensorial → Interneurônio → Comando → Motor."
                }},
                "rich_club": {{
                    titulo: "4. Anéis Concêntricos Rich-Club",
                    texto: "Arquitetura Núcleo-Periferia: os membros da elite Rich-Club ocupam o anel central (destacados em dourado), cercados por anéis periféricos."
                }},
                "multiplex": {{
                    titulo: "5. Polaridade Sináptica (Multigrafo)",
                    texto: "Eixo Y separa neurônios predominantemente Químicos (Topo), Multiplex Balanceados (Centro) e Elétricos / Gap Junctions (Base)."
                }}
            }};

            if (typeof network !== 'undefined') {{
                network.once("stabilizationIterationsDone", function() {{
                    network.setOptions({{ physics: {{ enabled: false }} }});
                    // Salva as posições estabilizadas reais do modo 'clusters'
                    var posicoesAtuais = network.getPositions();
                    for (var idNo in posicoesAtuais) {{
                        if (!idNo.startsWith("__ancora_") && coordenadasLayouts["clusters"][idNo]) {{
                            coordenadasLayouts["clusters"][idNo] = {{
                                x: posicoesAtuais[idNo].x,
                                y: posicoesAtuais[idNo].y
                            }};
                        }}
                    }}
                    network.fit({{ animation: {{ duration: 400 }} }});
                    fisicaAtiva = false;
                }});
            }}

            function mudarModoExibicao(modo) {{
                if (typeof network === 'undefined' || !coordenadasLayouts[modo]) return;

                // Congela a física para preservar perfeitamente a geometria do modo selecionado
                fisicaAtiva = false;
                network.setOptions({{ physics: {{ enabled: false }} }});
                document.getElementById("btn-toggle-physics").innerText = "Física: Congelada";

                var mapaCoords = coordenadasLayouts[modo];
                var atualizacoesNos = [];
                var todosNos = nodes.get();

                for (var i = 0; i < todosNos.length; i++) {{
                    var no = todosNos[i];
                    if (String(no.id).startsWith("__ancora_")) continue;
                    var destino = mapaCoords[no.id];
                    if (!destino) continue;

                    var ehRC = membrosRichClub.has(no.id);
                    var larguraBorda = (no.shape === "star" || ehRC) ? 4 : 1.5;
                    if (modo === "rich_club" && ehRC) {{
                        larguraBorda = 6;
                    }}

                    atualizacoesNos.push({{
                        id: no.id,
                        x: destino.x,
                        y: destino.y,
                        borderWidth: larguraBorda
                    }});
                }}

                nodes.update(atualizacoesNos);

                // Atualiza o texto explicativo da legenda
                if (descricoesModos[modo]) {{
                    document.getElementById("titulo-legenda").innerText = descricoesModos[modo].titulo;
                    document.getElementById("texto-explicativo-modo").innerText = descricoesModos[modo].texto;
                }}

                // Se mudou para o modo Multiplex, ativa automaticamente a coloração das camadas se estiver no padrão
                var selectCamada = document.getElementById("select-camada-arestas");
                if (modo === "multiplex" && selectCamada.value === "padrao") {{
                    selectCamada.value = "colorir_camadas";
                    filtrarCamadaSinaptica("colorir_camadas");
                }}

                network.fit({{
                    animation: {{ duration: 650, easingFunction: "easeInOutQuad" }}
                }});
            }}

            function filtrarCamadaSinaptica(filtro) {{
                if (typeof edges === 'undefined') return;
                var todasArestas = edges.get();
                var atualizacoesArestas = [];
                var legendaArestas = document.getElementById("legenda-arestas");

                legendaArestas.style.display = (filtro === "padrao") ? "none" : "block";

                for (var i = 0; i < todasArestas.length; i++) {{
                    var ar = todasArestas[i];
                    if (String(ar.from).startsWith("__ancora_")) continue;

                    var pq = ar.peso_quimico || 0;
                    var pe = ar.peso_eletrico || 0;
                    var pt = ar.peso_acumulado || (pq + pe) || 1;

                    var ocultar = false;
                    var cor = "rgba(160, 160, 160, 0.22)";
                    var setaAtiva = true;
                    var pesoRef = pt;

                    if (filtro === "quimica") {{
                        ocultar = (pq === 0);
                        cor = "rgba(255, 152, 0, 0.42)";
                        setaAtiva = true;
                        pesoRef = pq;
                    }} else if (filtro === "eletrica") {{
                        ocultar = (pe === 0);
                        cor = "rgba(0, 188, 212, 0.48)";
                        setaAtiva = false; // Gap junctions são simétricas
                        pesoRef = pe;
                    }} else if (filtro === "multiplex_only") {{
                        ocultar = !(pq > 0 && pe > 0);
                        cor = "rgba(255, 64, 129, 0.72)";
                        setaAtiva = true;
                        pesoRef = pt;
                    }} else if (filtro === "colorir_camadas") {{
                        ocultar = false;
                        if (pq > 0 && pe > 0) {{
                            cor = "rgba(255, 64, 129, 0.65)"; // Multiplex
                            setaAtiva = true;
                        }} else if (pe > 0) {{
                            cor = "rgba(0, 188, 212, 0.38)"; // Elétrica
                            setaAtiva = false;
                        }} else {{
                            cor = "rgba(255, 152, 0, 0.30)"; // Química
                            setaAtiva = true;
                        }}
                    }}

                    var largura = 0.7 + Math.log1p(Math.max(pesoRef, 1)) * 0.85;
                    atualizacoesArestas.push({{
                        id: ar.id,
                        hidden: ocultar,
                        color: cor,
                        width: largura,
                        arrows: {{ to: {{ enabled: setaAtiva, scaleFactor: 0.45 }} }}
                    }});
                }}

                edges.update(atualizacoesArestas);
            }}

            function alternarFisica() {{
                fisicaAtiva = !fisicaAtiva;
                network.setOptions({{ physics: {{ enabled: fisicaAtiva }} }});
                var btn = document.getElementById("btn-toggle-physics");
                btn.innerText = fisicaAtiva ? "Física: Ativa" : "Física: Congelada";
            }}

            function destacarRichClub() {{
                var listaRC = Array.from(membrosRichClub);
                if (!listaRC.length || typeof network === 'undefined') return;
                rcDestacado = !rcDestacado;
                var btn = document.getElementById("btn-rc-focus");
                if (rcDestacado) {{
                    network.selectNodes(listaRC, true);
                    network.fit({{ nodes: listaRC, animation: {{ duration: 600 }} }});
                    btn.innerText = "🔄 Visão Completa";
                }} else {{
                    network.unselectAll();
                    network.fit({{ animation: {{ duration: 500 }} }});
                    btn.innerText = "🎯 Focar Rich-Club";
                }}
            }}

            function alternarModalMetricas() {{
                var modal = document.getElementById("modal-metricas");
                if (!modal) return;
                modal.style.display = (modal.style.display === "flex") ? "none" : "flex";
            }}

            function fecharModalFora(e) {{
                if (e.target.id === "modal-metricas") alternarModalMetricas();
            }}

            function trocarAba(evt, idAba) {{
                var conteudos = document.getElementsByClassName("conteudo-aba");
                for (var i = 0; i < conteudos.length; i++) conteudos[i].style.display = "none";
                var botoes = document.getElementsByClassName("btn-aba");
                for (var i = 0; i < botoes.length; i++) botoes[i].classList.remove("aba-ativa");
                document.getElementById(idAba).style.display = "block";
                evt.currentTarget.classList.add("aba-ativa");
            }}

            document.addEventListener("keydown", function(e) {{
                if (e.key === "Escape") {{
                    var modal = document.getElementById("modal-metricas");
                    if (modal) modal.style.display = "none";
                }}
            }});
        </script>
        </body>
        """

        with open(caminho_saida, "r", encoding="utf-8") as f:
            conteudo_html = f.read()

        conteudo_html = conteudo_html.replace("</body>", patch_interface)

        with open(caminho_saida, "w", encoding="utf-8") as f:
            f.write(conteudo_html)

        print(f"Grafo interativo multi-layout salvo em: '{caminho_saida}'")
        return caminho_saida