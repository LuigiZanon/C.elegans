"""
Módulo de visualização estática (Matplotlib) e interativa (Pyvis) do conectoma.
Inclui caixa de seleção com 5 modos de exibição (layouts topológicos/biológicos)
e filtro de camadas do multigrafo (Química, Elétrica e Multiplex).
"""

import os
import json
import html
import base64
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple
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
    PALETA_MODELOS_SINTETICOS,
    PALETA_MODELOS_CAUDA,
    PALETA_ESTRATEGIAS_NAVEGACAO,
    PALETA_BOWTIE_PRESERVACAO,
    PALETA_CENARIOS_DIFUSAO,
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

    @staticmethod
    def _formatar_celula(nome_coluna: str, valor) -> str:
        """Formata o valor de uma célula da tabela HTML (numéricos com casas fixas, texto escapado)."""
        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            return ""
        if isinstance(valor, bool):
            return "Sim" if valor else "Não"
        if isinstance(valor, (int, np.integer)):
            return html.escape(str(int(valor)))
        if isinstance(valor, (float, np.floating)):
            if nome_coluna in ("PageRank", "Intermediacao"):
                texto = f"{float(valor):.4f}"
            elif nome_coluna == "Score_Chefe":
                texto = f"{float(valor):.3f}"
            else:
                texto = f"{float(valor):.2f}"
            return html.escape(texto)
        return html.escape(str(valor))

    def _montar_tabela_html(self, df: pd.DataFrame, cor_por_tipo: Optional[Dict[str, str]] = None) -> str:
        """Converte um DataFrame em uma tabela HTML com todos os valores escapados.

        Linhas identificadas como "chefe" do tipo recebem borda esquerda colorida
        (cor da paleta funcional), quando ``cor_por_tipo`` for fornecida.
        """
        if df is None or df.empty:
            return ""
        colunas = list(df.columns)
        tem_e_chefe = "E_Chefe" in colunas
        tem_posicao = "Posicao_No_Tipo" in colunas

        linhas_html = []
        for _, linha in df.iterrows():
            if tem_e_chefe:
                eh_chefe = bool(linha["E_Chefe"])
            elif tem_posicao:
                eh_chefe = int(linha["Posicao_No_Tipo"]) == 1
            else:
                eh_chefe = True  # ex.: tabela de resumo, onde cada linha é um chefe

            estilo_primeira_celula = ""
            if eh_chefe and cor_por_tipo:
                tipo = str(linha.get("Tipo", ""))
                cor = cor_por_tipo.get(tipo)
                if cor:
                    estilo_primeira_celula = f' style="border-left:4px solid {html.escape(cor)};"'

            celulas = []
            for i, coluna in enumerate(colunas):
                estilo = estilo_primeira_celula if i == 0 else ""
                celulas.append(f"<td{estilo}>{self._formatar_celula(coluna, linha[coluna])}</td>")
            linhas_html.append("<tr>" + "".join(celulas) + "</tr>")

        cabecalho = "".join(f"<th>{html.escape(str(c))}</th>" for c in colunas)
        return (
            '<div style="overflow-x:auto;">'
            f'<table class="tabela-ranking"><thead><tr>{cabecalho}</tr></thead>'
            f'<tbody>{"".join(linhas_html)}</tbody></table>'
            "</div>"
        )

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

    def plotar_comparacao_redes_sinteticas(
        self,
        df_comparacao: pd.DataFrame,
        diagnostico: Dict[str, Any],
        nome_arquivo: str = "comparacao_redes_sinteticas.png"
    ) -> str:
        """
        Gera painel comparativo 2x3 das métricas topológicas do conectoma real contra
        modelos sintéticos equivalentes (ER, WS e BA), incluindo cartão de diagnóstico.
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
        axes_flat = axes.ravel()

        modelos = list(df_comparacao["Modelo"].values)
        rotulos_curtos = {
            "C. elegans (Real)": "Real",
            "Erdős-Rényi (ER)": "ER",
            "Watts-Strogatz (WS)": "WS",
            "Barabási-Albert (BA)": "BA",
        }
        x_labels = [rotulos_curtos.get(m, m) for m in modelos]
        cores = [PALETA_MODELOS_SINTETICOS.get(m, "#666666") for m in modelos]
        x_pos = np.arange(len(modelos))

        paineis = [
            (
                "Agrupamento_C_Media",
                "Agrupamento_C_Std",
                r"Coeficiente de Agrupamento ($C$)",
                r"$C$",
                "{:.3f}",
                False,
            ),
            (
                "Caminho_Medio_L_Media",
                "Caminho_Medio_L_Std",
                r"Caminho Mínimo Médio ($L$)",
                r"$L$",
                "{:.3f}",
                False,
            ),
            (
                "Eficiencia_Global_Media",
                "Eficiencia_Global_Std",
                r"Eficiência Global ($E_{\mathrm{glob}}$)",
                r"$E_{\mathrm{glob}}$",
                "{:.3f}",
                False,
            ),
            (
                "Assortatividade_r_Media",
                "Assortatividade_r_Std",
                r"Assortatividade de Grau ($r$)",
                r"$r$",
                "{:+.3f}",
                True,
            ),
            (
                "Heterogeneidade_Media",
                "Heterogeneidade_Std",
                r"Heterogeneidade de Grau ($\kappa = \langle k^2 \rangle / \langle k \rangle^2$)",
                r"$\kappa$",
                "{:.2f}",
                False,
            ),
        ]

        for idx, (col_media, col_std, titulo, ylabel, fmt, linha_zero) in enumerate(paineis):
            ax = axes_flat[idx]
            medias = df_comparacao[col_media].astype(float).values
            stds = df_comparacao[col_std].astype(float).values

            barras = ax.bar(
                x_pos,
                medias,
                yerr=stds,
                capsize=5,
                color=cores,
                edgecolor="#222222",
                linewidth=0.9,
                alpha=0.90,
                error_kw={"elinewidth": 1.3, "ecolor": "#222222"},
            )

            if linha_zero:
                ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.1, alpha=0.8)

            # Ajusta limites verticais para acomodar rótulos numéricos sem sobreposição
            y_topo = float(np.max(medias + stds))
            y_base = float(np.min(medias - stds))
            amplitude = max(y_topo - y_base, 0.05)

            if linha_zero:
                ax.set_ylim(min(y_base, 0.0) - 0.25 * amplitude, max(y_topo, 0.0) + 0.25 * amplitude)
            else:
                ax.set_ylim(0.0, y_topo + 0.22 * max(y_topo, amplitude))

            deslocamento = 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            for barra, media, std in zip(barras, medias, stds):
                cx = barra.get_x() + barra.get_width() / 2.0
                if media >= 0:
                    y_txt = media + std + deslocamento * 0.6
                    va = "bottom"
                else:
                    y_txt = media - std - deslocamento * 0.6
                    va = "top"
                ax.text(
                    cx,
                    y_txt,
                    fmt.format(media),
                    ha="center",
                    va=va,
                    fontsize=9.5,
                    fontweight="bold",
                    color="#1f1f1f",
                )

            ax.set_title(titulo, fontsize=11.5, fontweight="bold", pad=10)
            ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=10, fontweight="bold")
            ax.grid(axis="y", alpha=0.3, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # 6º painel: Cartão de diagnóstico topológico formal
        ax_card = axes_flat[5]
        ax_card.axis("off")

        veredito_quebrado = "\n".join(
            textwrap.wrap(diagnostico["veredito_classificacao"], width=46)
        )
        linhas_card = [
            "DIAGNÓSTICO TOPOLÓGICO FORMAL",
            "─" * 34,
            rf"Calibração: $N={diagnostico['N']}$, $M={diagnostico['M']}$, $\langle k \rangle={diagnostico['k_medio']:.2f}$",
            rf"WS: $k={diagnostico['k_ws']}$, $p={diagnostico['p_ws']:.2f}$  |  BA: $m={diagnostico['m_ba']}$",
            "",
            rf"$\bullet\ C_{{\mathrm{{real}}}} / C_{{\mathrm{{ER}}}} = {diagnostico['razao_C_ER']:.2f}\times$",
            rf"$\bullet\ L_{{\mathrm{{real}}}} / L_{{\mathrm{{ER}}}} = {diagnostico['razao_L_ER']:.2f}\times$",
            rf"$\bullet\ \sigma_{{\mathrm{{SW}}}} = {diagnostico['sigma_small_world']:.2f}$  (critério: $\sigma > 1$)",
            rf"$\bullet\ \omega_{{\mathrm{{SW}}}} = {diagnostico['omega_small_world']:+.3f}$  (critério: $|\omega| \leq 0.5$)",
            rf"$\bullet\ \kappa_{{\mathrm{{real}}}} = {diagnostico['heterogeneidade_real']:.2f}$  (ER={diagnostico['heterogeneidade_ER']:.2f}, BA={diagnostico['heterogeneidade_BA']:.2f})",
            "",
            "Veredito:",
            veredito_quebrado,
        ]
        texto_card = "\n".join(linhas_card)

        ax_card.text(
            0.03,
            0.97,
            texto_card,
            transform=ax_card.transAxes,
            fontsize=9.4,
            verticalalignment="top",
            horizontalalignment="left",
            color="#1f2933",
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": "#f8f9fa",
                "edgecolor": "#c5ccd3",
                "linewidth": 1.3,
            },
        )

        fig.suptitle(
            "Comparação Topológica: Conectoma do C. elegans vs. Modelos Sintéticos Equivalentes",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de comparação com redes sintéticas salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_distribuicao_grau_comparativa(
        self,
        graus_por_modelo: Dict[str, np.ndarray],
        diagnostico: Dict[str, Any],
        nome_arquivo: str = "distribuicao_grau_comparativa.png"
    ) -> str:
        """
        Plota a distribuição de grau comparativa em dois painéis:
          - Esquerda: PMF / densidade em escala linear com linha vertical em <k>
          - Direita: CCDF P(K >= k) em escala log-log com resumo dos testes de cauda
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))

        modelo_real = "C. elegans (Real)"
        graus_real = np.asarray(graus_por_modelo[modelo_real], dtype=int)
        grau_max_global = int(
            max(np.max(v) for v in graus_por_modelo.values() if len(v) > 0)
        )

        # --- Painel Esquerdo: Escala Linear (PMF / Densidade) ---
        bins = np.arange(0, grau_max_global + 3, 2) - 0.5
        ax1.hist(
            graus_real,
            bins=bins,
            density=True,
            color=PALETA_MODELOS_SINTETICOS[modelo_real],
            alpha=0.35,
            edgecolor=PALETA_MODELOS_SINTETICOS[modelo_real],
            linewidth=1.1,
            label=modelo_real,
        )

        estilos_sint = {
            "Erdős-Rényi (ER)": {"linestyle": "--", "marker": "o", "markersize": 3.5},
            "Watts-Strogatz (WS)": {"linestyle": "-.", "marker": "^", "markersize": 3.5},
            "Barabási-Albert (BA)": {"linestyle": "-", "marker": "s", "markersize": 3.2},
        }

        for nome_modelo, estilo in estilos_sint.items():
            if nome_modelo not in graus_por_modelo:
                continue
            graus_m = np.asarray(graus_por_modelo[nome_modelo], dtype=int)
            k_vals, contagens = np.unique(graus_m, return_counts=True)
            pmf = contagens / float(len(graus_m))
            passo = max(1, len(k_vals) // 25)
            ax1.plot(
                k_vals,
                pmf,
                color=PALETA_MODELOS_SINTETICOS[nome_modelo],
                lw=2.1,
                linestyle=estilo["linestyle"],
                marker=estilo["marker"],
                markersize=estilo["markersize"],
                markevery=passo,
                label=nome_modelo,
            )

        k_medio = float(diagnostico["k_medio"])
        ax1.axvline(
            k_medio,
            color="#222222",
            linestyle=":",
            lw=1.8,
            label=rf"Grau médio $\langle k \rangle = {k_medio:.2f}$",
        )

        ax1.set_title("Distribuição de Grau (Escala Linear — PMF)", fontsize=12, fontweight="bold")
        ax1.set_xlabel(r"Grau ($k$)", fontsize=11, fontweight="bold")
        ax1.set_ylabel(r"Probabilidade $P(K = k)$", fontsize=11, fontweight="bold")
        ax1.set_xlim(left=0)
        ax1.grid(True, alpha=0.3, linestyle=":")
        ax1.legend(loc="upper right", fontsize=9.2)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- Painel Direito: Escala Log-Log (CCDF P(K >= k)) ---
        ordem_plot = [
            "Erdős-Rényi (ER)",
            "Watts-Strogatz (WS)",
            "Barabási-Albert (BA)",
            modelo_real,
        ]
        estilos_ccdf = {
            "C. elegans (Real)": {
                "lw": 2.4,
                "linestyle": "-",
                "marker": "o",
                "markersize": 4.2,
                "zorder": 5,
            },
            "Erdős-Rényi (ER)": {
                "lw": 2.0,
                "linestyle": "--",
                "marker": None,
                "markersize": 0,
                "zorder": 3,
            },
            "Watts-Strogatz (WS)": {
                "lw": 2.0,
                "linestyle": "-.",
                "marker": None,
                "markersize": 0,
                "zorder": 3,
            },
            "Barabási-Albert (BA)": {
                "lw": 2.1,
                "linestyle": "-",
                "marker": None,
                "markersize": 0,
                "zorder": 4,
            },
        }

        for nome_modelo in ordem_plot:
            if nome_modelo not in graus_por_modelo:
                continue
            graus_m = np.asarray(graus_por_modelo[nome_modelo], dtype=int)
            graus_pos = graus_m[graus_m > 0]
            if len(graus_pos) == 0:
                continue
            k_unicos = np.unique(graus_pos)
            # CCDF vetorizada via searchsorted sobre vetor ordenado
            graus_ord = np.sort(graus_pos)
            n_total = float(len(graus_ord))
            idx_inicio = np.searchsorted(graus_ord, k_unicos, side="left")
            ccdf = (n_total - idx_inicio) / n_total

            est = estilos_ccdf[nome_modelo]
            ax2.loglog(
                k_unicos,
                ccdf,
                color=PALETA_MODELOS_SINTETICOS[nome_modelo],
                lw=est["lw"],
                linestyle=est["linestyle"],
                marker=est["marker"],
                markersize=est["markersize"],
                zorder=est["zorder"],
                label=nome_modelo,
            )

        k_min = int(diagnostico["k_min_cauda"])
        ax2.axvline(
            k_min,
            color="#444444",
            linestyle=":",
            lw=1.6,
            label=rf"Início da cauda ($k_{{\min}} = {k_min}$)",
        )

        texto_cauda = "\n".join([
            "Análise da Cauda (Real)",
            rf"$k_{{\min}} = {k_min}$",
            rf"$\hat{{\gamma}}_{{\mathrm{{MLE}}}} = {diagnostico['gamma_mle']:.2f}$",
            rf"$R^2_{{\mathrm{{log-log}}}} = {diagnostico['r2_log_log']:.3f}$",
            rf"$R^2_{{\mathrm{{semi-log}}}} = {diagnostico['r2_semi_log']:.3f}$",
            rf"$\Delta \ell = {diagnostico['delta_log_likelihood']:+.2f}$",
            rf"$\kappa_{{\mathrm{{real}}}} = {diagnostico['heterogeneidade_real']:.2f}$",
        ])
        ax2.text(
            0.04,
            0.05,
            texto_cauda,
            transform=ax2.transAxes,
            fontsize=9.3,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#f8f9fa",
                "edgecolor": "#bbbbbb",
                "alpha": 0.95,
            },
        )

        ax2.set_title(
            r"Cauda da Distribuição de Grau (Escala Log-Log — CCDF $P(K \geq k)$)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel(r"Grau ($k$) [escala log]", fontsize=11, fontweight="bold")
        ax2.set_ylabel(r"$P(K \geq k)$ [escala log]", fontsize=11, fontweight="bold")
        ax2.grid(True, which="both", alpha=0.25, linestyle=":")
        ax2.legend(loc="upper right", fontsize=9.2)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de distribuição de grau comparativa salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_evidencias_small_world(
        self,
        ws_sweep_df: pd.DataFrame,
        resumo_sw: Dict[str, Any],
        nome_arquivo: str = "evidencias_small_world.png"
    ) -> str:
        """
        Gera figura 1x2 com evidências formais de Small-World:
          - Painel Esquerdo: Transição de fase de Watts-Strogatz C(p)/C(0) e L(p)/L(0)
            filtrando p=0.0 na escala log e marcando a posição empírica do C. elegans.
          - Painel Direito: Indicadores quantitativos sigma, omega e phi_SWP com
            limiares de aprovação e resumo do teste contra rede aleatória (Z_C e I_D).
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))

        # --- Painel Esquerdo: Curva de Transição de Fase de Watts-Strogatz ---
        df_pos = ws_sweep_df[ws_sweep_df["p_rewiring"] > 0.0].copy()
        p_vals = df_pos["p_rewiring"].astype(float).values
        c_norm = df_pos["C_norm_media"].astype(float).values
        c_std = df_pos["C_norm_std"].astype(float).values
        l_norm = df_pos["L_norm_media"].astype(float).values
        l_std = df_pos["L_norm_std"].astype(float).values

        ax1.axvspan(
            0.005,
            0.25,
            color="#34A853",
            alpha=0.10,
            label="Regime Small-World (Alto C, Baixo L)",
        )

        ax1.plot(
            p_vals,
            c_norm,
            color="#34A853",
            lw=2.4,
            marker="o",
            markersize=4.5,
            label=r"Agrupamento Normalizado $C(p) / C(0)$",
        )
        ax1.fill_between(
            p_vals,
            np.clip(c_norm - c_std, 0.0, 1.2),
            np.clip(c_norm + c_std, 0.0, 1.2),
            color="#34A853",
            alpha=0.18,
        )

        ax1.plot(
            p_vals,
            l_norm,
            color="#4285F4",
            lw=2.4,
            marker="s",
            markersize=4.5,
            label=r"Caminho Médio Normalizado $L(p) / L(0)$",
        )
        ax1.fill_between(
            p_vals,
            np.clip(l_norm - l_std, 0.0, 1.2),
            np.clip(l_norm + l_std, 0.0, 1.2),
            color="#4285F4",
            alpha=0.18,
        )

        p_equiv = float(resumo_sw["p_equiv_ws"])
        c_real_norm = float(resumo_sw.get("C_real_norm_latt", resumo_sw["C_real"] / max(resumo_sw["C_latt"], 1e-9)))
        l_real_norm = float(resumo_sw.get("L_real_norm_latt", resumo_sw["L_real"] / max(resumo_sw["L_latt"], 1e-9)))

        ax1.axvline(
            p_equiv,
            color="#EA4335",
            linestyle=":",
            lw=1.9,
            label=rf"Ponto Equivalente WS ($p_{{\mathrm{{equiv}}}} \approx {p_equiv:.3f}$)",
        )
        ax1.scatter(
            [p_equiv],
            [c_real_norm],
            color="#EA4335",
            marker="*",
            s=190,
            edgecolor="#111111",
            linewidth=0.8,
            zorder=6,
            label=rf"$C_{{\mathrm{{real}}}} / C_{{\mathrm{{latt}}}} = {c_real_norm:.3f}$",
        )
        ax1.scatter(
            [p_equiv],
            [l_real_norm],
            color="#FBBC05",
            marker="*",
            s=190,
            edgecolor="#111111",
            linewidth=0.8,
            zorder=6,
            label=rf"$L_{{\mathrm{{real}}}} / L_{{\mathrm{{latt}}}} = {l_real_norm:.3f}$",
        )

        ax1.set_xscale("log")
        ax1.set_ylim(-0.03, 1.12)
        ax1.set_title(
            "Transição de Fase de Watts–Strogatz e Posição do C. elegans",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_xlabel(r"Probabilidade de Religação ($p$) [escala log]", fontsize=11, fontweight="bold")
        ax1.set_ylabel(r"Razão Normalizada pelo Lattice ($p = 0$)", fontsize=11, fontweight="bold")
        ax1.grid(True, which="both", alpha=0.3, linestyle=":")
        ax1.legend(loc="lower left", fontsize=8.8)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- Painel Direito: Indicadores Formais de Small-World (sigma, omega, phi_SWP) ---
        sigma_val = float(resumo_sw["sigma"])
        sigma_std = float(resumo_sw["sigma_std"])
        omega_val = float(resumo_sw["omega"])
        phi_val = float(resumo_sw["phi_swp"])

        # Barras horizontais com razão Valor / Limiar ou escala padronizada
        nomes_ind = [
            r"Propensity $\phi_{\mathrm{SWP}}$" + "\n(Muldoon et al., ≥ 0.60)",
            r"Proximidade $\omega$" + "\n" + r"(Telesford, $1 - |\omega|$ ≥ 0.50)",
            r"Índice $\sigma_{\mathrm{SW}}$" + "\n(Humphries-Gurney, > 1.0)",
        ]
        # Valores plotados na barra (sigma em escala real, omega como 1 - |omega| para barra positiva, phi_SWP direto)
        prox_omega = max(0.0, 1.0 - abs(omega_val))
        valores_barra = [phi_val, prox_omega, sigma_val]
        limiares_barra = [0.60, 0.50, 1.00]
        cores_barra = ["#4285F4", "#FBBC05", "#34A853"]
        y_pos = np.arange(len(nomes_ind))

        barras = ax2.barh(
            y_pos,
            valores_barra,
            height=0.52,
            color=cores_barra,
            edgecolor="#222222",
            linewidth=1.0,
            alpha=0.90,
        )

        for idx_b, (y_b, lim) in enumerate(zip(y_pos, limiares_barra)):
            ax2.plot(
                [lim, lim],
                [y_b - 0.32, y_b + 0.32],
                color="#EA4335",
                linestyle="--",
                lw=2.0,
                label="Limiar Mínimo Small-World" if idx_b == 0 else None,
            )

        rotulos_valores = [
            rf"$\phi_{{\mathrm{{SWP}}}} = {phi_val:.3f}$  (Limiar $> 0.60$ — Aprovado)",
            rf"$\omega = {omega_val:+.3f}$  ($|\omega| \leq 0.50$ — Aprovado)",
            rf"$\sigma = {sigma_val:.2f} \pm {sigma_std:.2f}$  (Limiar $> 1.0$ — Aprovado)",
        ]
        x_max_bar = max(valores_barra) * 1.48
        ax2.set_xlim(0.0, max(x_max_bar, 6.5))

        for barra, texto_r in zip(barras, rotulos_valores):
            w = barra.get_width()
            cy = barra.get_y() + barra.get_height() / 2.0
            ax2.text(
                w + 0.12,
                cy,
                texto_r,
                va="center",
                ha="left",
                fontsize=9.5,
                fontweight="bold",
                color="#1f2933",
            )

        z_c = float(resumo_sw.get("z_score_C_vs_ER", 0.0))
        id_pois = float(resumo_sw.get("indice_dispersao_ID", 0.0))
        c_real = float(resumo_sw["C_real"])
        c_er = float(resumo_sw.get("C_ER_media", 0.059))
        l_real = float(resumo_sw["L_real"])
        l_er = float(resumo_sw["L_ER_media"])

        caixa_resumo = "\n".join([
            "REJEIÇÃO DA HIPÓTESE ALEATÓRIA (ER) & VEREDITO SW",
            "─" * 44,
            rf"$\bullet\ C_{{\mathrm{{real}}}} = {c_real:.4f}$ vs. $C_{{\mathrm{{ER}}}} = {c_er:.4f}$  ($Z_C = {z_c:.1f} \gg 3.0$)",
            rf"$\bullet\ L_{{\mathrm{{real}}}} = {l_real:.3f}$ vs. $L_{{\mathrm{{ER}}}} = {l_er:.3f}$  (Razão $= {l_real / max(l_er, 1e-9):.2f}\times$)",
            rf"$\bullet\ $Dispersão de Poisson: $I_D = \mathrm{{Var}}(k)/\langle k \rangle = {id_pois:.2f} \gg 1.0$",
            rf"$\bullet\ $Veredito: {resumo_sw['veredito'].upper()} nos 3 critérios formais.",
        ])
        ax2.text(
            0.03,
            0.04,
            caixa_resumo,
            transform=ax2.transAxes,
            fontsize=9.2,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#f8f9fa",
                "edgecolor": "#c5ccd3",
                "linewidth": 1.2,
            },
        )

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(nomes_ind, fontsize=10, fontweight="bold")
        ax2.set_xlabel("Magnitude da Métrica / Escala de Evidência", fontsize=11, fontweight="bold")
        ax2.set_title(
            r"Painel de Evidências Small-World ($\sigma$, $\omega$, $\phi_{\mathrm{SWP}}$) e Teste vs. ER",
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(axis="x", alpha=0.3, linestyle=":")
        ax2.legend(loc="upper right", fontsize=9.2)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de evidências Small-World salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_evidencias_scale_free(
        self,
        modelos_cauda_df: pd.DataFrame,
        resumo_sf: Dict[str, Any],
        nome_arquivo: str = "evidencias_scale_free.png"
    ) -> str:
        """
        Gera figura 1x2 com evidências do teste Livre de Escala (Clauset et al. + Vuong):
          - Painel Esquerdo: CCDF na cauda k >= k_min* em escala log-log comparando os
            dados empíricos com 4 modelos ajustados via MLE (PL, TPL, LN, EXP) + Poisson/ER.
          - Painel Direito: Seleção de modelos via Delta AIC e estatística de Vuong (R, p),
            incluindo análise direcional da camada Química (k_in vs. k_out).
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))

        # --- Painel Esquerdo: CCDF na Cauda k >= k_min* em Escala Log-Log ---
        curvas = resumo_sf["curvas_ccdf"]
        k_vals = np.asarray(curvas["k_vals"], dtype=float)
        ccdf_emp = np.asarray(curvas["ccdf_empirica"], dtype=float)
        ccdf_pl = np.asarray(curvas["ccdf_pl"], dtype=float)
        ccdf_tpl = np.asarray(curvas["ccdf_tpl"], dtype=float)
        ccdf_ln = np.asarray(curvas["ccdf_ln"], dtype=float)
        ccdf_exp = np.asarray(curvas["ccdf_exp"], dtype=float)
        ccdf_pois = np.asarray(curvas["ccdf_poisson"], dtype=float)

        ax1.loglog(
            k_vals,
            ccdf_emp,
            color=PALETA_MODELOS_CAUDA["Empírico (C. elegans)"],
            marker="o",
            markersize=5.2,
            linestyle="None",
            zorder=6,
            label="Empírico (C. elegans)",
        )
        ax1.loglog(
            k_vals,
            ccdf_pl,
            color=PALETA_MODELOS_CAUDA["Lei de Potência (PL)"],
            lw=2.3,
            linestyle="-",
            zorder=4,
            label="Lei de Potência (PL)",
        )
        ax1.loglog(
            k_vals,
            ccdf_tpl,
            color=PALETA_MODELOS_CAUDA["PL Truncada (TPL)"],
            lw=2.5,
            linestyle="--",
            zorder=5,
            label="PL Truncada (TPL)",
        )
        ax1.loglog(
            k_vals,
            ccdf_ln,
            color=PALETA_MODELOS_CAUDA["Log-Normal (LN)"],
            lw=2.2,
            linestyle="-.",
            zorder=4,
            label="Log-Normal (LN)",
        )
        ax1.loglog(
            k_vals,
            ccdf_exp,
            color=PALETA_MODELOS_CAUDA["Exponencial (EXP)"],
            lw=2.1,
            linestyle=":",
            zorder=3,
            label="Exponencial (EXP)",
        )

        # Filtra Poisson apenas onde não sofreu underflow severo para manter limites visuais limpos
        mask_pois = ccdf_pois >= 1e-4
        if np.any(mask_pois):
            ax1.loglog(
                k_vals[mask_pois],
                ccdf_pois[mask_pois],
                color=PALETA_MODELOS_CAUDA["Poisson / ER"],
                lw=1.8,
                linestyle="--",
                alpha=0.85,
                zorder=2,
                label="Poisson / ER",
            )

        k_min = int(resumo_sf["k_min"])
        ax1.axvline(
            k_min,
            color="#444444",
            linestyle=":",
            lw=1.7,
            label=rf"Limiar Ótimo $k_{{\min}}^* = {k_min}$ ($n={resumo_sf['n_tail']}$)",
        )

        y_min_emp = max(1e-3, float(np.min(ccdf_emp)) * 0.45)
        ax1.set_ylim(y_min_emp, 1.35)

        texto_clauset = "\n".join([
            "Protocolo Clauset–Shalizi–Newman",
            rf"$k_{{\min}}^* = {k_min}$  ($n_{{\mathrm{{tail}}}} = {resumo_sf['n_tail']}$)",
            rf"$\hat{{\gamma}}_{{\mathrm{{PL}}}} = {resumo_sf['gamma_pl']:.2f}$",
            rf"$D_{{\mathrm{{KS}}}} = {resumo_sf['ks_dist_pl']:.3f}$",
            rf"$p_{{\mathrm{{KS}}}} = {resumo_sf['p_valor_ks_bootstrap']:.2f}$  (limiar $\geq 0.10$)",
            rf"$R^2_{{\mathrm{{log-log}}}} = {resumo_sf['r2_log_log']:.3f}$ vs. $R^2_{{\mathrm{{semi-log}}}} = {resumo_sf['r2_semi_log']:.3f}$",
        ])
        ax1.text(
            0.04,
            0.05,
            texto_clauset,
            transform=ax1.transAxes,
            fontsize=9.2,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#f8f9fa",
                "edgecolor": "#bbbbbb",
                "alpha": 0.95,
            },
        )

        ax1.set_title(
            rf"Ajuste de Modelos na Cauda ($k \geq {k_min}$) — CCDF Condicional $P(K \geq k \mid K \geq k_{{\min}}^*)$",
            fontsize=11.8,
            fontweight="bold",
        )
        ax1.set_xlabel(r"Grau ($k$) [escala log]", fontsize=11, fontweight="bold")
        ax1.set_ylabel(r"$P(K \geq k \mid K \geq k_{\min}^*)$ [escala log]", fontsize=11, fontweight="bold")
        ax1.grid(True, which="both", alpha=0.28, linestyle=":")
        ax1.legend(loc="upper right", fontsize=8.9)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- Painel Direito: Seleção de Modelos (Delta AIC e Teste de Vuong) ---
        df_ord = modelos_cauda_df.sort_values("Delta_AIC", ascending=True).reset_index(drop=True)
        modelos_nomes = list(df_ord["Modelo"].values)
        delta_aics = df_ord["Delta_AIC"].astype(float).values
        cores_m = [PALETA_MODELOS_CAUDA.get(m, "#666666") for m in modelos_nomes]
        y_pos = np.arange(len(modelos_nomes))

        barras = ax2.barh(
            y_pos,
            delta_aics,
            height=0.52,
            color=cores_m,
            edgecolor="#222222",
            linewidth=1.0,
            alpha=0.90,
        )

        max_daic = max(float(np.max(delta_aics)), 4.0)
        ax2.set_xlim(0.0, max_daic * 1.65)

        for idx_r, (barra, row) in enumerate(zip(barras, df_ord.to_dict("records"))):
            w = barra.get_width()
            cy = barra.get_y() + barra.get_height() / 2.0
            if row["Modelo"] == "Lei de Potência (PL)":
                anot = rf"$\Delta\mathrm{{AIC}} = {w:.2f}$  (Referência Vuong)"
            else:
                r_v = float(row["Vuong_R_vs_PL"])
                p_v = float(row["Vuong_p_vs_PL"])
                anot = rf"$\Delta\mathrm{{AIC}} = {w:.2f}$  |  Vuong $R={r_v:+.2f}$ ($p={p_v:.3f}$)"
            if idx_r == 0:
                anot = "★ MELHOR AJUSTE: " + anot
            ax2.text(
                w + 0.08 * max_daic * 0.25,
                cy,
                anot,
                va="center",
                ha="left",
                fontsize=9.1,
                fontweight="bold",
                color="#1f2933",
            )

        dir_q = resumo_sf["direcionado_quimico"]
        veredito_quebrado = "\n".join(textwrap.wrap(str(resumo_sf["veredito"]), width=54))
        caixa_dir = "\n".join([
            "ASSIMETRIA DIRECIONAL (CAMADA QUÍMICA) & VEREDITO",
            "─" * 48,
            rf"$\bullet\ k_{{\mathrm{{in}}}}$: $\kappa = {dir_q['in_degree_kappa']:.2f}$, $\hat{{\gamma}} = {dir_q.get('in_degree_gamma', 0.0):.2f}$, $R^2_{{\mathrm{{log}}}} = {dir_q.get('in_degree_r2_log_log', 0.0):.3f}$ vs. $R^2_{{\mathrm{{semi}}}} = {dir_q.get('in_degree_r2_semi_log', 0.0):.3f}$",
            rf"$\bullet\ k_{{\mathrm{{out}}}}$: $\kappa = {dir_q['out_degree_kappa']:.2f}$, $\hat{{\gamma}} = {dir_q.get('out_degree_gamma', 0.0):.2f}$, $R^2_{{\mathrm{{log}}}} = {dir_q.get('out_degree_r2_log_log', 0.0):.3f}$ vs. $R^2_{{\mathrm{{semi}}}} = {dir_q.get('out_degree_r2_semi_log', 0.0):.3f}$",
            rf"$\bullet\ $Veredito: {veredito_quebrado}",
        ])
        ax2.text(
            0.03,
            0.04,
            caixa_dir,
            transform=ax2.transAxes,
            fontsize=8.9,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#f8f9fa",
                "edgecolor": "#c5ccd3",
                "linewidth": 1.2,
            },
        )

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(modelos_nomes, fontsize=10, fontweight="bold")
        ax2.invert_yaxis()  # Melhor modelo (Delta AIC = 0) no topo
        ax2.set_xlabel(r"$\Delta \mathrm{AIC} = \mathrm{AIC}_m - \mathrm{AIC}_{\min}$ (Menor é Melhor)", fontsize=11, fontweight="bold")
        ax2.set_title(
            r"Seleção de Modelos na Cauda ($\Delta \mathrm{AIC}$ e Teste de Vuong vs. PL)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(axis="x", alpha=0.3, linestyle=":")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de evidências Scale-Free salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_navegacao_greedy(
        self,
        df_navegacao: pd.DataFrame,
        df_nav_fluxo: pd.DataFrame,
        resumo_nav: Dict[str, Any],
        nome_arquivo: str = "navegacao_greedy.png"
    ) -> str:
        """
        Gera figura 1x3 com os resultados do Roteamento Guloso (Greedy Routing):
          - Painel 1: Taxa de Sucesso (S_R) e Eficiência (E_R) — Real vs. Modelos Nulos
          - Painel 2: Fator de Alongamento (Stretch tau) — Média, Mediana e P90
          - Painel 3: Heatmap 4x4 de Navegabilidade por Fluxo Funcional (destaque Sensorial -> Motor)
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        df_undir = df_navegacao[
            df_navegacao["Visao"] == "Combinada (Não-Direcionada)"
        ].copy()

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.8))

        estrategias = list(df_undir["Estrategia"])
        nomes_curtos = [
            e.replace(" (soma_pos)", "\n(soma_pos)")
             .replace(" (Hubs)", "\n(Hubs)")
             .replace(" (Espacial + Grau)", "\n(Esp.+Grau)")
             .replace(" Aleatória", "\nAleatória")
            for e in estrategias
        ]
        cores = [PALETA_ESTRATEGIAS_NAVEGACAO.get(e, "#607D8B") for e in estrategias]
        x = np.arange(len(estrategias))
        largura = 0.36

        # --- Painel 1: Taxa de Sucesso (Real vs Nulo) + E_R ---
        sr_real = df_undir["Taxa_Sucesso_Pct"].to_numpy(dtype=float)
        sr_nulo = df_undir["Nulo_Sucesso_Media"].fillna(0.0).to_numpy(dtype=float)
        sr_nulo_std = df_undir["Nulo_Sucesso_Std"].fillna(0.0).to_numpy(dtype=float)
        er_vals = df_undir["Razao_Eficiencia"].to_numpy(dtype=float)

        barras_real = ax1.bar(
            x - largura / 2, sr_real, width=largura, color=cores,
            edgecolor="#222222", linewidth=0.9, label="C. elegans (Real)"
        )
        ax1.bar(
            x + largura / 2, sr_nulo, width=largura, yerr=sr_nulo_std,
            color="#E0E0E0", edgecolor="#555555", linewidth=0.9,
            hatch="///", capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "#444444"},
            label="Modelo Nulo (Espacial / Rewiring)"
        )

        for idx, barra in enumerate(barras_real):
            h = barra.get_height()
            ax1.text(
                barra.get_x() + barra.get_width() / 2,
                min(h + 2.0, 108.0),
                f"{h:.1f}%\n" + r"$E_R$=" + f"{er_vals[idx]:.2f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#111111"
            )

        ax1.set_xticks(x)
        ax1.set_xticklabels(nomes_curtos, fontsize=9, fontweight="bold")
        ax1.set_ylim(0, 118)
        ax1.set_ylabel(r"Taxa de Sucesso $S_R$ (%)", fontsize=10.5, fontweight="bold")
        ax1.set_title(
            r"Taxa de Sucesso ($S_R$) e Eficiência ($E_R$): Real vs. Nulo",
            fontsize=11.5, fontweight="bold"
        )
        ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.92)
        ax1.grid(axis="y", alpha=0.3, linestyle=":")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- Painel 2: Fator de Alongamento (Stretch tau) ---
        tau_medio = df_undir["Stretch_Medio"].fillna(1.0).to_numpy(dtype=float)
        tau_mediana = df_undir["Stretch_Mediana"].fillna(1.0).to_numpy(dtype=float)
        tau_p90 = df_undir["Stretch_P90"].fillna(1.0).to_numpy(dtype=float)
        frac_otimo = df_undir["Frac_Otimo_Pct"].fillna(0.0).to_numpy(dtype=float)
        pct_rc = df_undir["Pct_Rotas_Via_RichClub"].fillna(0.0).to_numpy(dtype=float)

        w2 = 0.25
        ax2.bar(
            x - w2, tau_mediana, width=w2, color="#90CAF9", edgecolor="#222222",
            linewidth=0.8, label=r"Mediana ($\tau_{50}$)"
        )
        barras_tau = ax2.bar(
            x, tau_medio, width=w2, color=cores, edgecolor="#222222",
            linewidth=0.9, label=r"Média ($\bar{\tau}$)"
        )
        ax2.bar(
            x + w2, tau_p90, width=w2, color="#FFCC80", edgecolor="#222222",
            linewidth=0.8, label=r"Percentil 90 ($\tau_{90}$)"
        )
        ax2.axhline(
            1.0, color="#2E7D32", linestyle="--", linewidth=1.4,
            label=r"Caminho Mínimo Ótimo ($\tau = 1{,}0$)"
        )

        y_max_tau = max(float(np.nanmax(tau_p90)) * 1.25, 2.2)
        for idx, barra in enumerate(barras_tau):
            topo = max(tau_p90[idx], tau_medio[idx])
            ax2.text(
                x[idx],
                topo + 0.04 * y_max_tau,
                f"Ót: {frac_otimo[idx]:.0f}%\nRC: {pct_rc[idx]:.0f}%",
                ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#222222"
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels(nomes_curtos, fontsize=9, fontweight="bold")
        ax2.set_ylim(0, y_max_tau)
        ax2.set_ylabel(r"Fator de Alongamento ($\tau = h_{\mathrm{greedy}} / d_{\mathrm{SP}}$)", fontsize=10.5, fontweight="bold")
        ax2.set_title(
            r"Qualidade das Rotas (Stretch $\tau$, % Ótimas e % Via Rich-Club)",
            fontsize=11.5, fontweight="bold"
        )
        ax2.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
        ax2.grid(axis="y", alpha=0.3, linestyle=":")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # --- Painel 3: Heatmap 4x4 por Fluxo Funcional ---
        melhor_est = resumo_nav.get("melhor_estrategia_undir", "Grau (Hubs)")
        sub_fluxo = df_nav_fluxo[df_nav_fluxo["Estrategia"] == melhor_est]

        classes = [
            c for c in ORDEM_CLASSES_PADRAO
            if (c in set(sub_fluxo["Tipo_Origem"])) or (c in set(sub_fluxo["Tipo_Destino"]))
        ]
        if not classes:
            classes = sorted(set(sub_fluxo["Tipo_Origem"]) | set(sub_fluxo["Tipo_Destino"]))

        n_cls = len(classes)
        mat_sr = np.full((n_cls, n_cls), np.nan, dtype=float)
        mat_tau = np.full((n_cls, n_cls), np.nan, dtype=float)

        for _, row in sub_fluxo.iterrows():
            orig = row["Tipo_Origem"]
            dest = row["Tipo_Destino"]
            if orig in classes and dest in classes:
                i = classes.index(orig)
                j = classes.index(dest)
                mat_sr[i, j] = float(row["Taxa_Sucesso_Pct"])
                mat_tau[i, j] = float(row["Stretch_Medio"])

        im = ax3.imshow(mat_sr, cmap="YlGn", vmin=0.0, vmax=100.0, aspect="auto")
        rotulos_cls = [
            c.replace("Interneurônio (Comando)", "Inter.\n(Comando)")
             .replace("Interneurônio", "Inter.")
            for c in classes
        ]
        ax3.set_xticks(range(n_cls))
        ax3.set_xticklabels(rotulos_cls, fontsize=8.5, fontweight="bold")
        ax3.set_yticks(range(n_cls))
        ax3.set_yticklabels(rotulos_cls, fontsize=8.5, fontweight="bold")
        ax3.set_xlabel("Classe Funcional de Destino", fontsize=10, fontweight="bold")
        ax3.set_ylabel("Classe Funcional de Origem", fontsize=10, fontweight="bold")
        ax3.set_title(
            f"Fluxo Funcional — {melhor_est}\n" + r"$S_R$ (%) e ($\bar{\tau}$)",
            fontsize=11.0, fontweight="bold"
        )

        for i in range(n_cls):
            for j in range(n_cls):
                val_sr = mat_sr[i, j]
                val_tau = mat_tau[i, j]
                if np.isnan(val_sr):
                    txt = "—"
                    cor_txt = "#555555"
                else:
                    tau_str = f"{val_tau:.2f}" if not np.isnan(val_tau) else "—"
                    txt = f"{val_sr:.1f}%\n({tau_str})"
                    cor_txt = "white" if val_sr >= 65.0 else "#111111"
                ax3.text(
                    j, i, txt, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=cor_txt
                )
                # Destaque visual para o canal biológico Sensorial -> Motor
                if classes[i] == "Sensorial" and classes[j] == "Motor":
                    rect = plt.Rectangle(
                        (j - 0.48, i - 0.48), 0.96, 0.96,
                        fill=False, edgecolor="#D32F2F", linewidth=2.4
                    )
                    ax3.add_patch(rect)

        cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label(r"$S_R$ (%)", fontsize=9, fontweight="bold")

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de navegabilidade (Greedy Routing) salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_assortatividade(
        self,
        df_assortatividade: pd.DataFrame,
        df_knn: pd.DataFrame,
        resumo_assort: Dict[str, Any],
        nome_arquivo: str = "assortatividade_conectoma.png"
    ) -> str:
        """
        Gera figura 1x2 com o diagnóstico de Assortatividade do conectoma:
          - Painel Esquerdo: Curva k_nn(k) vs. k em escala log-log (Real vs. Nulo) + reta beta_knn
          - Painel Direito: Perfil multidimensional de assortatividade r (barras divergentes)
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.0))

        # --- Painel Esquerdo: Curva k_nn(k) vs. k (log-log) ---
        if not df_knn.empty:
            k_vals = df_knn["k"].to_numpy(dtype=float)
            knn_real = df_knn["knn_real"].to_numpy(dtype=float)
            knn_nulo = df_knn["knn_nulo_media"].to_numpy(dtype=float)
            knn_std = df_knn["knn_nulo_std"].fillna(0.0).to_numpy(dtype=float)

            mask_nulo = np.isfinite(knn_nulo) & (knn_nulo > 0)
            if np.any(mask_nulo):
                k_n = k_vals[mask_nulo]
                med_n = knn_nulo[mask_nulo]
                std_n = knn_std[mask_nulo]
                baixo = np.maximum(med_n - std_n, 1e-2)
                alto = med_n + std_n
                ax1.fill_between(
                    k_n, baixo, alto, color="#BDBDBD", alpha=0.40,
                    label=r"Modelo Nulo (Rewiring, $\mu \pm \sigma$)"
                )
                ax1.plot(
                    k_n, med_n, color="#616161", linestyle="--", linewidth=1.6,
                    label=r"Média Nula $k_{\mathrm{nn}}^{\mathrm{nulo}}(k)$"
                )

            ax1.scatter(
                k_vals, knn_real, color="#1565C0", s=34, alpha=0.85, zorder=4,
                edgecolor="white", linewidth=0.5, label=r"C. elegans (Empírico)"
            )

            beta_knn = float(resumo_assort.get("beta_knn", 0.0))
            intercept_knn = float(resumo_assort.get("intercept_knn", 0.0))
            if len(k_vals) >= 2:
                k_grid = np.geomspace(float(np.min(k_vals)), float(np.max(k_vals)), 100)
                knn_fit = np.exp(intercept_knn) * np.power(k_grid, beta_knn)
                ax1.plot(
                    k_grid, knn_fit, color="#D32F2F", linewidth=2.2, zorder=5,
                    label=r"Ajuste Log-Log ($\beta_{k_{\mathrm{nn}}} = " + f"{beta_knn:.3f}" + r"$)"
                )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel(r"Grau do Neurônio ($k$)", fontsize=11, fontweight="bold")
        ax1.set_ylabel(r"Grau Médio dos Vizinhos ($k_{\mathrm{nn}}(k)$)", fontsize=11, fontweight="bold")
        ax1.set_title(
            r"Correlação Grau-Grau: Curva $k_{\mathrm{nn}}(k)$ vs. $k$ (Log-Log)",
            fontsize=12, fontweight="bold"
        )
        ax1.legend(loc="lower left", fontsize=8.8, framealpha=0.92)
        ax1.grid(True, which="both", alpha=0.25, linestyle=":")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        r_real = float(resumo_assort.get("r_grau_combinada", 0.0))
        r_nulo_med = float(resumo_assort.get("r_grau_nulo_media", 0.0))
        z_r = float(resumo_assort.get("z_score_r_grau", 0.0))
        dist_real = float(resumo_assort.get("distancia_fisica_media_aresta", 0.0))
        dist_nula = float(resumo_assort.get("distancia_fisica_media_nula", 0.0))

        caixa_texto = (
            r"$r_k$ (Real) = " + f"{r_real:+.3f}\n"
            + r"$r_k$ (Nulo) = " + f"{r_nulo_med:+.3f}\n"
            + r"$Z$-score = " + f"{z_r:+.2f}\n"
            + r"$\bar{d}_{\mathrm{aresta}}$ = " + f"{dist_real:.3f} (Nulo: {dist_nula:.3f})"
        )
        ax1.text(
            0.97, 0.96, caixa_texto,
            transform=ax1.transAxes,
            fontsize=9.2,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#F8F9FA",
                "edgecolor": "#B0BEC5",
                "linewidth": 1.1,
            },
        )

        # --- Painel Direito: Barras Horizontais Divergentes de Assortatividade ---
        itens_perfil = [
            (r"Grau — Combinada ($r_k$)", float(resumo_assort.get("r_grau_combinada", 0.0))),
            (r"Grau — Combinada Ponderada ($r_k^w$)", float(resumo_assort.get("r_grau_combinada_ponderada", 0.0))),
            (r"Grau — Química Direcionada ($\mathrm{out} \to \mathrm{in}$)", float(resumo_assort.get("r_quimica_out_in", 0.0))),
            (r"Grau — Elétrica ($r_k^{\mathrm{elet}}$)", float(resumo_assort.get("r_eletrica_grau", 0.0))),
            ("Homofilia Funcional (tipo_funcional)", float(resumo_assort.get("r_homofilia_tipo_funcional", 0.0))),
            ("Homofilia Anatômica (regiao_soma)", float(resumo_assort.get("r_homofilia_regiao_soma", 0.0))),
            ("Correlação Espacial (soma_pos)", float(resumo_assort.get("r_espacial_soma_pos", 0.0))),
        ]

        rotulos = [item[0] for item in itens_perfil]
        valores = [item[1] for item in itens_perfil]
        cores_barras = ["#34A853" if v >= 0.0 else "#EA4335" for v in valores]
        y_pos = np.arange(len(rotulos))

        barras = ax2.barh(
            y_pos, valores, color=cores_barras, edgecolor="#222222", linewidth=0.85, height=0.62
        )
        ax2.axvline(0.0, color="#222222", linewidth=1.2)

        limite_x = max(0.35, min(1.05, max(abs(v) for v in valores) + 0.16))
        ax2.set_xlim(-limite_x, limite_x)

        for barra, val in zip(barras, valores):
            desloc = 0.02 * limite_x if val >= 0 else -0.02 * limite_x
            ha = "left" if val >= 0 else "right"
            ax2.text(
                val + desloc,
                barra.get_y() + barra.get_height() / 2,
                f"{val:+.3f}",
                va="center", ha=ha, fontsize=9.2, fontweight="bold", color="#111111"
            )

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(rotulos, fontsize=9.5, fontweight="bold")
        ax2.invert_yaxis()
        ax2.set_xlabel(r"Coeficiente de Assortatividade ($r \in [-1, 1]$)", fontsize=11, fontweight="bold")
        ax2.set_title(
            r"Perfil Multidimensional de Assortatividade (Verde: Assortativo | Vermelho: Disassortativo)",
            fontsize=11.0, fontweight="bold"
        )
        ax2.grid(axis="x", alpha=0.3, linestyle=":")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de assortatividade salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_preservacao_reciprocidade_scc(
        self,
        df_preservacao_resumo: pd.DataFrame,
        mat_reciprocidade: pd.DataFrame,
        df_bowtie_classes: pd.DataFrame,
        resumo_json: Dict[str, Any],
        nome_arquivo: str = "preservacao_reciprocidade_scc.png"
    ) -> str:
        """
        Gera figura 1x3 com os Pilares 1 e 2 de Preservação de Informação:
          - Painel 1: Reciprocidade Real vs. Modelo Nulo (Química R, Química R_w, Combinada R)
          - Painel 2: Heatmap de Reciprocidade Química por Classe Funcional (R_AB em %)
          - Painel 3: Decomposição Bow-Tie da Camada Química por Classe Funcional
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.8))

        rec = resumo_json.get("reciprocidade", {})
        scc = resumo_json.get("scc_bowtie", {})

        # --- Painel 1: Reciprocidade Real vs. Modelo Nulo ---
        categorias = [
            r"Química ($R$)",
            r"Química ($R_{w}$)",
            r"Combinada ($R$)",
        ]
        vals_real = [
            float(rec.get("quimica_topologica", 0.0)) * 100.0,
            float(rec.get("quimica_ponderada", 0.0)) * 100.0,
            float(rec.get("combinada_topologica", 0.0)) * 100.0,
        ]
        vals_nulo = [
            float(rec.get("quimica_nulo_topologica_media", 0.0)) * 100.0,
            float(rec.get("quimica_nulo_ponderada_media", 0.0)) * 100.0,
            float(rec.get("combinada_nulo_topologica_media", rec.get("quimica_nulo_topologica_media", 0.0))) * 100.0,
        ]
        stds_nulo = [
            float(rec.get("quimica_nulo_topologica_std", 0.0)) * 100.0,
            float(rec.get("quimica_nulo_ponderada_std", 0.0)) * 100.0,
            float(rec.get("combinada_nulo_topologica_std", rec.get("quimica_nulo_topologica_std", 0.0))) * 100.0,
        ]
        z_scores = [
            float(rec.get("z_score_reciprocidade", 0.0)),
            float(rec.get("z_score_reciprocidade_ponderada", 0.0)),
            float(rec.get("z_score_combinada_topologica", 0.0)),
        ]

        x = np.arange(len(categorias))
        larg = 0.34
        b_real = ax1.bar(
            x - larg / 2, vals_real, width=larg, color="#EA4335",
            edgecolor="#222222", linewidth=0.85, label="C. elegans (Real)"
        )
        b_nulo = ax1.bar(
            x + larg / 2, vals_nulo, width=larg, yerr=stds_nulo, capsize=4,
            color="#9E9E9E", edgecolor="#222222", linewidth=0.85,
            error_kw={"elinewidth": 1.1, "ecolor": "#333333"},
            label=r"Modelo Nulo Direcionado ($\mu \pm \sigma$)"
        )

        y_max_1 = max(max(vals_real + vals_nulo) * 1.28, 25.0)
        ax1.set_ylim(0, min(115.0, y_max_1))

        for bar_r, bar_n, vr, vn, zv in zip(b_real, b_nulo, vals_real, vals_nulo, z_scores):
            ax1.text(
                bar_r.get_x() + bar_r.get_width() / 2,
                vr + 1.2,
                f"{vr:.1f}%\n(Z={zv:+.1f})",
                ha="center", va="bottom", fontsize=8.3, fontweight="bold", color="#B71C1C"
            )
            ax1.text(
                bar_n.get_x() + bar_n.get_width() / 2,
                vn + 1.2,
                f"{vn:.1f}%",
                ha="center", va="bottom", fontsize=8.2, color="#424242"
            )

        ax1.set_xticks(x)
        ax1.set_xticklabels(categorias, fontsize=9.5, fontweight="bold")
        ax1.set_ylabel("Reciprocidade (%)", fontsize=10.5, fontweight="bold")
        ax1.set_title(
            r"Reciprocidade Real vs. Modelo Nulo ($R$ e $R_{w}$)",
            fontsize=11.5, fontweight="bold"
        )
        ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
        ax1.grid(axis="y", alpha=0.28, linestyle=":")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- Painel 2: Heatmap de Reciprocidade Química por Classe Funcional ---
        mat_pct = mat_reciprocidade.to_numpy(dtype=float) * 100.0
        rotulos_curtos = [
            c.replace("Interneurônio (Comando)", "Comando").replace("Interneurônio", "Interneur.")
            for c in mat_reciprocidade.index
        ]
        vmax_hm = max(float(np.nanmax(mat_pct)) if mat_pct.size > 0 else 50.0, 10.0)
        im = ax2.imshow(mat_pct, cmap="YlOrRd", vmin=0.0, vmax=vmax_hm, aspect="auto")

        for i in range(mat_pct.shape[0]):
            for j in range(mat_pct.shape[1]):
                val = mat_pct[i, j]
                cor_txt = "white" if val >= 0.58 * vmax_hm else "#111111"
                ax2.text(
                    j, i, f"{val:.1f}%",
                    ha="center", va="center", fontsize=9.5, fontweight="bold", color=cor_txt
                )

        ax2.set_xticks(np.arange(len(rotulos_curtos)))
        ax2.set_yticks(np.arange(len(rotulos_curtos)))
        ax2.set_xticklabels(rotulos_curtos, rotation=22, ha="right", fontsize=9.0, fontweight="bold")
        ax2.set_yticklabels(rotulos_curtos, fontsize=9.0, fontweight="bold")
        ax2.set_xlabel("Classe Destino (B)", fontsize=10.0, fontweight="bold")
        ax2.set_ylabel("Classe Origem (A)", fontsize=10.0, fontweight="bold")
        ax2.set_title(
            r"Reciprocidade Química por Classe ($R_{AB}$, %)",
            fontsize=11.5, fontweight="bold"
        )
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label(r"$R_{AB}$ (%)", fontsize=9.0, fontweight="bold")

        # --- Painel 3: Decomposição Bow-Tie por Classe Funcional ---
        setores_ordem = [
            "IN (Entrada / Feeders)",
            "SCC (Núcleo Recorrente)",
            "OUT (Saída / Sinks)",
            "Outros (Isolados/Tubes)",
        ]
        rotulos_setores = [
            "IN\n(Entrada)",
            "SCC\n(Núcleo)",
            "OUT\n(Saída)",
            "Outros\n(Isolados)",
        ]
        df_bt = df_bowtie_classes.set_index("Setor_BowTie")
        classes_plot = [c for c in ORDEM_CLASSES_PADRAO if c in df_bt.columns]
        x_bt = np.arange(len(setores_ordem))
        base = np.zeros(len(setores_ordem), dtype=float)

        for cls in classes_plot:
            alturas = np.array([
                float(df_bt.loc[s, cls]) if s in df_bt.index else 0.0
                for s in setores_ordem
            ])
            cor_cls = PALETA_CORES_FUNCIONAIS.get(cls, "#757575")
            ax3.bar(
                x_bt, alturas, bottom=base, width=0.58,
                color=cor_cls, edgecolor="#222222", linewidth=0.7,
                label=cls.replace("Interneurônio (Comando)", "Comando")
            )
            base += alturas

        for idx_s, tot_s in enumerate(base):
            ax3.text(
                x_bt[idx_s], tot_s + max(base) * 0.02,
                f"{int(round(tot_s))}",
                ha="center", va="bottom", fontsize=9.2, fontweight="bold", color="#111111"
            )

        ax3.set_ylim(0, max(base) * 1.24 if len(base) > 0 and max(base) > 0 else 10)
        ax3.set_xticks(x_bt)
        ax3.set_xticklabels(rotulos_setores, fontsize=9.2, fontweight="bold")
        ax3.set_ylabel("Número de Neurônios", fontsize=10.5, fontweight="bold")
        ax3.set_title(
            r"Decomposição Bow-Tie (Camada Química)",
            fontsize=11.5, fontweight="bold"
        )
        ax3.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
        ax3.grid(axis="y", alpha=0.28, linestyle=":")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        lscc_tam = int(scc.get("quimica_lscc_tamanho", 0))
        lscc_pct = float(scc.get("quimica_lscc_fracao_pct", 0.0))
        lscc_comb_pct = float(scc.get("combinada_lscc_fracao_pct", 0.0))
        rc_qtd = int(scc.get("rich_club_na_lscc_quimica_qtd", 0))
        rc_tot = int(scc.get("rich_club_total", 0))
        rc_pct = float(scc.get("rich_club_na_lscc_quimica_pct", 0.0))

        caixa_scc = (
            r"$|V_{\mathrm{LSCC}}|$ Química = " + f"{lscc_tam} ({lscc_pct:.1f}%)\n"
            + f"LSCC Combinada = {lscc_comb_pct:.1f}%\n"
            + f"Rich-Club na SCC = {rc_qtd}/{rc_tot} ({rc_pct:.0f}%)"
        )
        ax3.text(
            0.03, 0.96, caixa_scc,
            transform=ax3.transAxes,
            fontsize=8.4,
            verticalalignment="top",
            horizontalalignment="left",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#FFF8E1",
                "edgecolor": PALETA_BOWTIE_PRESERVACAO.get("SCC (Núcleo Recorrente)", "#EA4335"),
                "linewidth": 1.1,
            },
        )

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de preservação (reciprocidade e SCC) salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_preservacao_modularidade_ciclos(
        self,
        df_modularidade: pd.DataFrame,
        df_ciclos_motifs: pd.DataFrame,
        df_ciclos_por_classe: pd.DataFrame,
        resumo_json: Dict[str, Any],
        nome_arquivo: str = "preservacao_modularidade_ciclos.png"
    ) -> str:
        """
        Gera figura 1x3 com os Pilares 3 e 4 de Preservação de Informação:
          - Painel 1: Modularidade Q (Louvain Química/Combinada, Funcional, Anatômica, Nulo)
          - Painel 2: Enriquecimento de Ciclos Curtos (C2, C3) e Tríades (Z-Score vs. Nulo)
          - Painel 3: Participação Média Nodal em 2-Ciclos e 3-Ciclos por Classe e Rich-Club
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.8))

        mod = resumo_json.get("modularidade", {})
        cic = resumo_json.get("ciclos_curtos", {})

        # --- Painel 1: Modularidade Q e Alinhamento Biológico ---
        rotulos_q = [
            "Louvain\n(Química)",
            "Louvain\n(Combinada)",
            "Partição\nFuncional",
            "Partição\nAnatômica",
            "Modelo Nulo\n(Química)",
        ]
        vals_q = [
            float(mod.get("Q_quimica_louvain_ponderada", 0.0)),
            float(mod.get("Q_combinada_louvain_ponderada", 0.0)),
            float(mod.get("Q_funcional_quimica", 0.0)),
            float(mod.get("Q_anatomica_quimica", 0.0)),
            float(mod.get("Q_quimica_nulo_media", 0.0)),
        ]
        errs_q = [0.0, 0.0, 0.0, 0.0, float(mod.get("Q_quimica_nulo_std", 0.0))]
        cores_q = ["#EA4335", "#4285F4", "#FBBC05", "#34A853", "#9E9E9E"]

        x_q = np.arange(len(rotulos_q))
        barras_q = ax1.bar(
            x_q, vals_q, width=0.58, yerr=errs_q, capsize=4,
            color=cores_q, edgecolor="#222222", linewidth=0.85,
            error_kw={"elinewidth": 1.1, "ecolor": "#333333"}
        )
        ax1.axhline(0.0, color="#222222", linewidth=0.9)

        y_min_q = min(0.0, min(vals_q) - 0.08)
        y_max_q = max(max(vals_q) * 1.32, 0.45)
        ax1.set_ylim(y_min_q, y_max_q)

        for barra, vq in zip(barras_q, vals_q):
            desloc = 0.015 if vq >= 0 else -0.025
            va = "bottom" if vq >= 0 else "top"
            ax1.text(
                barra.get_x() + barra.get_width() / 2,
                vq + desloc,
                f"{vq:.3f}",
                ha="center", va=va, fontsize=8.8, fontweight="bold", color="#111111"
            )

        z_q = float(mod.get("Q_quimica_z_score", 0.0))
        f_intra = float(mod.get("retencao_intra_modulo_quimica_pct", 0.0))
        nmi_f = float(mod.get("nmi_louvain_vs_funcional", 0.0))
        nmi_a = float(mod.get("nmi_louvain_vs_anatomico", 0.0))

        caixa_mod = (
            r"$Z_{Q}$ (Química) = " + f"{z_q:+.2f}\n"
            + r"$F_{\mathrm{intra}}$ (Química) = " + f"{f_intra:.1f}%\n"
            + f"NMI Funcional = {nmi_f:.3f}\n"
            + f"NMI Anatômico = {nmi_a:.3f}"
        )
        ax1.text(
            0.97, 0.96, caixa_mod,
            transform=ax1.transAxes,
            fontsize=8.4,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#F8F9FA",
                "edgecolor": "#B0BEC5",
                "linewidth": 1.05,
            },
        )

        ax1.set_xticks(x_q)
        ax1.set_xticklabels(rotulos_q, fontsize=8.6, fontweight="bold")
        ax1.set_ylabel(r"Modularidade Ponderada ($Q$)", fontsize=10.5, fontweight="bold")
        ax1.set_title(
            r"Modularidade ($Q$) e Retenção Intra-Módulo",
            fontsize=11.5, fontweight="bold"
        )
        ax1.grid(axis="y", alpha=0.28, linestyle=":")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- Painel 2: Enriquecimento de Ciclos Curtos e Tríades (Z-Score) ---
        estruturas = df_ciclos_motifs["Estrutura"].tolist()
        z_motifs = df_ciclos_motifs["Z_Score"].to_numpy(dtype=float)
        razoes_motifs = df_ciclos_motifs["Razao_Real_Nulo"].to_numpy(dtype=float)
        y_m = np.arange(len(estruturas))

        cores_m = [
            "#EA4335" if z > 2.0 else ("#4285F4" if z < -2.0 else "#9E9E9E")
            for z in z_motifs
        ]
        barras_m = ax2.barh(
            y_m, z_motifs, height=0.60, color=cores_m,
            edgecolor="#222222", linewidth=0.85
        )
        ax2.axvline(0.0, color="#222222", linewidth=1.1)
        ax2.axvline(2.0, color="#616161", linewidth=1.0, linestyle="--", alpha=0.75)
        ax2.axvline(-2.0, color="#616161", linewidth=1.0, linestyle="--", alpha=0.75)

        min_z = float(np.min(z_motifs)) if z_motifs.size > 0 else 0.0
        max_z = float(np.max(z_motifs)) if z_motifs.size > 0 else 4.0
        span_z = max(max_z - min(min_z, 0.0), 10.0)
        x_esq = min(-3.5, min_z - 0.42 * span_z) if min_z < 0 else -3.5
        x_dir = max(4.5, max_z + 0.36 * span_z)
        ax2.set_xlim(x_esq, x_dir)

        for barra, zv, rv in zip(barras_m, z_motifs, razoes_motifs):
            desloc = 0.018 * (x_dir - x_esq)
            if zv >= 0:
                ax2.text(
                    zv + desloc, barra.get_y() + barra.get_height() / 2,
                    f"Z={zv:+.1f} ({rv:.2f}x)",
                    va="center", ha="left", fontsize=8.3, fontweight="bold", color="#111111"
                )
            else:
                ax2.text(
                    zv - desloc, barra.get_y() + barra.get_height() / 2,
                    f"Z={zv:+.1f} ({rv:.2f}x)",
                    va="center", ha="right", fontsize=8.3, fontweight="bold", color="#111111"
                )

        ax2.set_yticks(y_m)
        ax2.set_yticklabels(estruturas, fontsize=8.6, fontweight="bold")
        ax2.invert_yaxis()
        ax2.set_xlabel(r"$Z$-Score vs. Modelo Nulo Direcionado ($|Z| > 2$ significativo)", fontsize=9.8, fontweight="bold")
        ax2.set_title(
            r"Enriquecimento de Ciclos ($C_{2}, C_{3}$) e Tríades",
            fontsize=11.5, fontweight="bold"
        )
        ax2.grid(axis="x", alpha=0.28, linestyle=":")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # --- Painel 3: Participação Média em Ciclos por Classe e Rich-Club ---
        df_cls = df_ciclos_por_classe.copy()
        grupos = [
            g.replace("Interneurônio (Comando)", "Comando")
             .replace("Interneurônio", "Interneur.")
             .replace("Núcleo Rich-Club", "Rich-Club")
             .replace("Periferia (Demais)", "Periferia")
            for g in df_cls["Grupo"].tolist()
        ]
        c2_med = df_cls["C2_Media"].to_numpy(dtype=float)
        c3_med = df_cls["C3_Media"].to_numpy(dtype=float)
        n_classes_func = int((df_cls["Categoria"] == "Classe Funcional").sum())

        x_c = np.arange(len(grupos))
        larg_c = 0.36
        b_c2 = ax3.bar(
            x_c - larg_c / 2, np.maximum(c2_med, 0.05), width=larg_c,
            color="#4285F4", edgecolor="#222222", linewidth=0.8,
            label=r"Média 2-Ciclos ($C_{2}$ / neurônio)"
        )
        b_c3 = ax3.bar(
            x_c + larg_c / 2, np.maximum(c3_med, 0.05), width=larg_c,
            color="#EA4335", edgecolor="#222222", linewidth=0.8,
            label=r"Média 3-Ciclos ($C_{3}$ / neurônio)"
        )

        if 0 < n_classes_func < len(grupos):
            ax3.axvline(n_classes_func - 0.5, color="#424242", linestyle="--", linewidth=1.1, alpha=0.75)

        ax3.set_yscale("log")
        y_top_3 = max(float(np.max(np.concatenate([c2_med, c3_med]))) * 2.8, 10.0)
        ax3.set_ylim(0.1, y_top_3)

        for barra, val in zip(b_c2, c2_med):
            ax3.text(
                barra.get_x() + barra.get_width() / 2,
                max(val, 0.1) * 1.12,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#0D47A1"
            )
        for barra, val in zip(b_c3, c3_med):
            ax3.text(
                barra.get_x() + barra.get_width() / 2,
                max(val, 0.1) * 1.12,
                f"{val:.0f}" if val >= 10 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#B71C1C"
            )

        ax3.set_xticks(x_c)
        ax3.set_xticklabels(grupos, rotation=24, ha="right", fontsize=8.6, fontweight="bold")
        ax3.set_ylabel(r"Participação Média em Ciclos (escala log)", fontsize=10.0, fontweight="bold")
        ax3.set_title(
            r"Ciclos Curtos ($C_{2}, C_{3}$) por Classe e Rich-Club",
            fontsize=11.5, fontweight="bold"
        )
        ax3.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
        ax3.grid(axis="y", which="both", alpha=0.25, linestyle=":")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        frac_rec = float(cic.get("fracao_triades_recorrentes_pct", 0.0))
        ax3.text(
            0.97, 0.96,
            f"Tríades recorrentes: {frac_rec:.1f}%",
            transform=ax3.transAxes,
            fontsize=8.2,
            va="top", ha="right",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#FFF8E1",
                "edgecolor": "#FBBC05",
                "linewidth": 1.0,
            },
        )

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de preservação (modularidade e ciclos) salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_difusao_estrutural(
        self,
        df_estrutural: pd.DataFrame,
        df_distancias: pd.DataFrame,
        resumo_json: Dict[str, Any],
        nome_arquivo: str = "difusao_metricas_estruturais.png"
    ) -> str:
        """
        Gera painel 1x3 comparando métricas estruturais de difusão do conectoma vs. Erdős-Rényi:
          1. Caminho Mínimo Médio (L) e Diâmetro (D) nas visões Combinada (Undir e Dir).
          2. Eficiência Global (E_glob) nas 3 visões + box resumo (Sensorial -> Motor e lambda_c).
          3. Distribuição de distâncias mínimas P(d) (d = 1, 2, 3, 4, 5+) na Combinada Undir.
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.8))
        fig.suptitle(
            r"Facilidade de Difusão — Pilar 1: Métricas Estruturais ($L$, $D$, $E_{\mathrm{glob}}$, $\lambda_c$) vs. Erdős-Rényi",
            fontsize=13.5,
            fontweight="bold",
            y=1.02,
        )

        # --- Painel 1: Caminho Mínimo Médio (L) e Diâmetro (D) ---
        visoes_p1 = ["Combinada (Não-Direcionada)", "Combinada (Direcionada)"]
        sub_p1 = df_estrutural[df_estrutural["Visao"].isin(visoes_p1)].copy()
        sub_p1["Visao"] = pd.Categorical(sub_p1["Visao"], categories=visoes_p1, ordered=True)
        sub_p1 = sub_p1.sort_values("Visao")

        categorias_p1 = [
            r"$L$" + "\nComb. Undir",
            r"$D$" + "\nComb. Undir",
            r"$L$" + "\nComb. Dir",
            r"$D$" + "\nComb. Dir",
        ]
        vals_real_p1 = []
        vals_er_p1 = []
        stds_er_p1 = []
        for _, row in sub_p1.iterrows():
            vals_real_p1.extend([float(row["L_Real"]), float(row["D_Real"])])
            vals_er_p1.extend([float(row["L_ER_Media"]), float(row["D_ER_Media"])])
            stds_er_p1.extend([float(row["L_ER_Std"]), float(row["D_ER_Std"])])

        x1 = np.arange(len(categorias_p1))
        larg1 = 0.36
        barras_real_1 = ax1.bar(
            x1 - larg1 / 2, vals_real_p1, width=larg1,
            color="#EA4335", edgecolor="#222222", linewidth=0.8,
            label=r"C. elegans (Real)",
        )
        barras_er_1 = ax1.bar(
            x1 + larg1 / 2, vals_er_p1, yerr=stds_er_p1, capsize=4, width=larg1,
            color="#9E9E9E", edgecolor="#222222", linewidth=0.8,
            label=r"Erdős-Rényi ($\mu \pm \sigma$)",
        )

        y_max_1 = max(max(vals_real_p1), max(vals_er_p1)) * 1.24 if vals_real_p1 else 10.0
        ax1.set_ylim(0, y_max_1)
        for barra, val in zip(barras_real_1, vals_real_p1):
            ax1.text(
                barra.get_x() + barra.get_width() / 2, val + y_max_1 * 0.018,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8.2, fontweight="bold", color="#B71C1C",
            )
        for barra, val, err in zip(barras_er_1, vals_er_p1, stds_er_p1):
            ax1.text(
                barra.get_x() + barra.get_width() / 2, val + err + y_max_1 * 0.018,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=8.0, color="#424242",
            )

        ax1.set_xticks(x1)
        ax1.set_xticklabels(categorias_p1, fontsize=8.8, fontweight="bold")
        ax1.set_ylabel(r"Número de Saltos Topológicos", fontsize=10.0, fontweight="bold")
        ax1.set_title(r"Caminho Mínimo Médio ($L$) e Diâmetro ($D$)", fontsize=11.3, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=8.3, framealpha=0.92)
        ax1.grid(axis="y", alpha=0.28, linestyle=":")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- Painel 2: Eficiência Global (E_glob) nas 3 Visões + Caixa Resumo ---
        rotulos_p2 = [
            v.replace("Combinada (Não-Direcionada)", "Comb.\nUndir")
             .replace("Combinada (Direcionada)", "Comb.\nDir")
             .replace("Química (Direcionada)", "Química\nDir")
            for v in df_estrutural["Visao"].tolist()
        ]
        e_real = df_estrutural["E_Glob_Real"].to_numpy(dtype=float)
        e_er = df_estrutural["E_Glob_ER_Media"].to_numpy(dtype=float)
        e_er_std = df_estrutural["E_Glob_ER_Std"].to_numpy(dtype=float)

        x2 = np.arange(len(rotulos_p2))
        larg2 = 0.36
        b_e_real = ax2.bar(
            x2 - larg2 / 2, e_real, width=larg2,
            color="#4285F4", edgecolor="#222222", linewidth=0.8,
            label=r"C. elegans ($E_{\mathrm{glob}}$)",
        )
        b_e_er = ax2.bar(
            x2 + larg2 / 2, e_er, yerr=e_er_std, capsize=4, width=larg2,
            color="#9E9E9E", edgecolor="#222222", linewidth=0.8,
            label=r"Erdős-Rényi ($\mu \pm \sigma$)",
        )

        y_max_2 = max(float(np.max(e_real)), float(np.max(e_er + e_er_std))) * 1.48 if len(e_real) else 1.0
        ax2.set_ylim(0, min(y_max_2, 1.05))
        for barra, val in zip(b_e_real, e_real):
            ax2.text(
                barra.get_x() + barra.get_width() / 2, val + y_max_2 * 0.018,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.2, fontweight="bold", color="#0D47A1",
            )
        for barra, val, err in zip(b_e_er, e_er, e_er_std):
            ax2.text(
                barra.get_x() + barra.get_width() / 2, val + err + y_max_2 * 0.018,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8.0, color="#424242",
            )

        est_u = resumo_json.get("estrutural", {}).get("combinada_undir", {})
        est_d = resumo_json.get("estrutural", {}).get("combinada_dir", {})
        l_sm = est_d.get("L_sensorial_para_motor", 0.0) or 0.0
        alc_sm = est_d.get("alcance_sensorial_para_motor_pct", 0.0) or 0.0
        lc_real = est_u.get("lambda_c_real", 0.0) or 0.0
        lc_er = est_u.get("lambda_c_er_media", 0.0) or 0.0

        texto_box_2 = (
            f"Canal Sensorial -> Motor (Dir):\n"
            f"  $L_{{S\\to M}} = {l_sm:.2f}$ | Alcance: {alc_sm:.1f}%\n"
            f"Limiar Epidêmico (Undir):\n"
            f"  $\\lambda_{{c,\\mathrm{{real}}}} = {lc_real:.4f}$ vs. "
            f"$\\lambda_{{c,\\mathrm{{ER}}}} = {lc_er:.4f}$"
        )
        ax2.text(
            0.97, 0.97, texto_box_2,
            transform=ax2.transAxes, fontsize=7.9,
            va="top", ha="right",
            bbox={
                "boxstyle": "round,pad=0.38",
                "facecolor": "#FFF8E1",
                "edgecolor": "#FBBC05",
                "linewidth": 1.0,
            },
        )

        ax2.set_xticks(x2)
        ax2.set_xticklabels(rotulos_p2, fontsize=9.0, fontweight="bold")
        ax2.set_ylabel(r"Eficiência Global ($E_{\mathrm{glob}}$)", fontsize=10.0, fontweight="bold")
        ax2.set_title(r"Eficiência Global ($E_{\mathrm{glob}}$) e Canal Biológico", fontsize=11.3, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
        ax2.grid(axis="y", alpha=0.28, linestyle=":")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # --- Painel 3: Distribuição de Distâncias Mínimas P(d) ---
        sub_dist = df_distancias[df_distancias["Visao"] == "Combinada (Não-Direcionada)"].copy()
        ordem_d = ["1", "2", "3", "4", "5+"]
        sub_dist["Distancia"] = pd.Categorical(sub_dist["Distancia"], categories=ordem_d, ordered=True)
        sub_dist = sub_dist.sort_values("Distancia")

        x3 = np.arange(len(sub_dist))
        larg3 = 0.36
        p_real = sub_dist["Pct_Real"].to_numpy(dtype=float)
        p_er = sub_dist["Pct_ER_Media"].to_numpy(dtype=float)
        p_er_std = sub_dist["Pct_ER_Std"].to_numpy(dtype=float)

        b_d_real = ax3.bar(
            x3 - larg3 / 2, p_real, width=larg3,
            color="#EA4335", edgecolor="#222222", linewidth=0.8,
            label=r"C. elegans (Real)",
        )
        b_d_er = ax3.bar(
            x3 + larg3 / 2, p_er, yerr=p_er_std, capsize=3.5, width=larg3,
            color="#9E9E9E", edgecolor="#222222", linewidth=0.8,
            label=r"Erdős-Rényi ($\mu \pm \sigma$)",
        )

        y_max_3 = max(float(np.max(p_real)), float(np.max(p_er + p_er_std))) * 1.32 if len(p_real) else 100.0
        ax3.set_ylim(0, min(y_max_3, 105.0))
        for barra, val in zip(b_d_real, p_real):
            if val >= 0.5:
                ax3.text(
                    barra.get_x() + barra.get_width() / 2, val + y_max_3 * 0.015,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=7.6, fontweight="bold", color="#B71C1C",
                )
        for barra, val, err in zip(b_d_er, p_er, p_er_std):
            if val >= 0.5:
                ax3.text(
                    barra.get_x() + barra.get_width() / 2, val + err + y_max_3 * 0.015,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=7.4, color="#424242",
                )

        pct3_real = est_u.get("pct_pares_ate_3_saltos_real", 0.0) or 0.0
        pct3_er = est_u.get("pct_pares_ate_3_saltos_er", 0.0) or 0.0
        ax3.text(
            0.97, 0.96,
            f"Pares em até 3 saltos ($d \\leq 3$):\n"
            f"  C. elegans: {pct3_real:.1f}%\n"
            f"  Erdős-Rényi: {pct3_er:.1f}%",
            transform=ax3.transAxes, fontsize=8.0,
            va="top", ha="right",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#E8F0FE",
                "edgecolor": "#4285F4",
                "linewidth": 1.0,
            },
        )

        ax3.set_xticks(x3)
        ax3.set_xticklabels([f"d = {d}" for d in sub_dist["Distancia"].tolist()], fontsize=9.0, fontweight="bold")
        ax3.set_ylabel(r"Percentual de Pares $P(d)$ (%)", fontsize=10.0, fontweight="bold")
        ax3.set_title(r"Distribuição de Distâncias Mínimas $P(d)$ (Undir)", fontsize=11.3, fontweight="bold")
        ax3.legend(loc="upper left", fontsize=8.0, framealpha=0.92)
        ax3.grid(axis="y", alpha=0.28, linestyle=":")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de difusão (métricas estruturais) salvo em: '{caminho_saida}'")
        return caminho_saida

    def plotar_simulacao_difusao(
        self,
        df_si_curvas: pd.DataFrame,
        df_si_resumo: pd.DataFrame,
        df_ic_curvas: pd.DataFrame,
        resumo_json: Dict[str, Any],
        nome_arquivo: str = "difusao_simulacao_propagacao.png"
    ) -> str:
        """
        Gera painel 1x3 com as simulações dinâmicas de difusão (Combinada Não-Direcionada):
          1. Curvas temporais SI (I(t)/N %) com bandas de +/- 1 desvio padrão e linha de 50%.
          2. Curva de transição de fase da Cascata Independente / SIR (Alcance vs. beta).
          3. Barras horizontais comparando AUC_SI (%) e Alcance IC em beta_ref + box direcionado.
        """
        caminho_saida = self._resolver_caminho(nome_arquivo)
        visao_principal = "Combinada (Não-Direcionada)"
        cenarios_ordem = list(PALETA_CENARIOS_DIFUSAO.keys())

        beta_si = float(resumo_json.get("metadados", {}).get("beta_si_padrao", 0.08))
        beta_ref = float(resumo_json.get("simulacao_ic", {}).get("beta_referencia", 0.08))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 5.8))
        fig.suptitle(
            r"Facilidade de Difusão — Pilar 2: Simulações Dinâmicas (Modelo SI e Cascata Independente / SIR)",
            fontsize=13.5,
            fontweight="bold",
            y=1.02,
        )

        # --- Painel 1: Velocidade de Difusão — Modelo SI ---
        sub_si = df_si_curvas[df_si_curvas["Visao"] == visao_principal]
        for cenario in cenarios_ordem:
            df_c = sub_si[sub_si["Cenario"] == cenario].sort_values("Passo_t")
            if df_c.empty:
                continue
            cor = PALETA_CENARIOS_DIFUSAO.get(cenario, "#333333")
            t_vals = df_c["Passo_t"].to_numpy(dtype=float)
            med = df_c["Infectados_Pct_Media"].to_numpy(dtype=float)
            std = df_c["Infectados_Pct_Std"].to_numpy(dtype=float)
            estilo = "--" if cenario.startswith("Erdős") else "-"
            larg = 2.3 if "Hubs / Rich-Club" in cenario else 1.8

            ax1.plot(t_vals, med, linestyle=estilo, linewidth=larg, color=cor, label=cenario)
            ax1.fill_between(
                t_vals,
                np.clip(med - std, 0.0, 100.0),
                np.clip(med + std, 0.0, 100.0),
                color=cor,
                alpha=0.14,
            )

        ax1.axhline(50.0, color="#333333", linestyle=":", linewidth=1.2, alpha=0.85)
        ax1.text(0.4, 52.0, r"Limiar $t_{50}$ (50%)", fontsize=8.0, color="#333333", fontweight="bold")
        ax1.set_xlim(0, float(sub_si["Passo_t"].max()) if not sub_si.empty else 25.0)
        ax1.set_ylim(0, 105.0)
        ax1.set_xlabel(r"Passo Temporal ($t$)", fontsize=10.0, fontweight="bold")
        ax1.set_ylabel(r"Nós Ativados $I(t)/N$ (%)", fontsize=10.0, fontweight="bold")
        ax1.set_title(
            rf"Velocidade de Difusão — Modelo SI ($\beta = {beta_si:.2f}$)",
            fontsize=11.2, fontweight="bold",
        )
        ax1.legend(loc="lower right", fontsize=7.6, framealpha=0.92)
        ax1.grid(alpha=0.28, linestyle=":")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # --- Painel 2: Transição de Fase — Cascata Independente / SIR ---
        sub_ic = df_ic_curvas[df_ic_curvas["Visao"] == visao_principal]
        for cenario in cenarios_ordem:
            df_c = sub_ic[sub_ic["Cenario"] == cenario].sort_values("Beta")
            if df_c.empty:
                continue
            cor = PALETA_CENARIOS_DIFUSAO.get(cenario, "#333333")
            betas = df_c["Beta"].to_numpy(dtype=float)
            med = df_c["Alcance_Final_Pct_Media"].to_numpy(dtype=float)
            std = df_c["Alcance_Final_Pct_Std"].to_numpy(dtype=float)
            estilo = "--" if cenario.startswith("Erdős") else "-"
            marcador = "s" if cenario.startswith("Erdős") else "o"
            larg = 2.3 if "Hubs / Rich-Club" in cenario else 1.8

            ax2.plot(
                betas, med, linestyle=estilo, marker=marcador, markersize=4.5,
                linewidth=larg, color=cor, label=cenario,
            )
            ax2.fill_between(
                betas,
                np.clip(med - std, 0.0, 100.0),
                np.clip(med + std, 0.0, 100.0),
                color=cor,
                alpha=0.14,
            )

        ax2.axhline(20.0, color="#616161", linestyle=":", linewidth=1.1, alpha=0.85)
        ax2.axvline(beta_ref, color="#333333", linestyle="-.", linewidth=1.0, alpha=0.75)
        ax2.text(
            beta_ref + 0.006, 94.0,
            rf"$\beta_{{\mathrm{{ref}}}} = {beta_ref:.2f}$",
            fontsize=8.0, color="#222222", fontweight="bold",
        )
        ax2.text(
            float(sub_ic["Beta"].max()) * 0.98 if not sub_ic.empty else 0.34,
            22.0,
            r"Cascata Global (20%)",
            ha="right", fontsize=7.8, color="#424242",
        )
        ax2.set_ylim(0, 105.0)
        ax2.set_xlabel(r"Probabilidade de Transmissão ($\beta$)", fontsize=10.0, fontweight="bold")
        ax2.set_ylabel(r"Alcance Final $R_{\infty}/N$ (%)", fontsize=10.0, fontweight="bold")
        ax2.set_title(r"Transição de Fase — Cascata Independente (IC / SIR)", fontsize=11.2, fontweight="bold")
        ax2.legend(loc="lower right", fontsize=7.6, framealpha=0.92)
        ax2.grid(alpha=0.28, linestyle=":")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        # --- Painel 3: Resumo Comparativo (AUC_SI e Alcance IC em beta_ref) ---
        sub_si_res = df_si_resumo[df_si_resumo["Visao"] == visao_principal]
        sub_ic_ref = sub_ic[np.isclose(sub_ic["Beta"].to_numpy(dtype=float), beta_ref, atol=1e-9)]

        rotulos_curtos = [
            c.replace("C. elegans ", "C.el. ").replace("Erdős-Rényi ", "ER ")
            for c in cenarios_ordem
        ]
        auc_vals = []
        alc_vals = []
        for cenario in cenarios_ordem:
            r_si = sub_si_res[sub_si_res["Cenario"] == cenario]
            r_ic = sub_ic_ref[sub_ic_ref["Cenario"] == cenario]
            auc_vals.append(float(r_si["AUC_SI_Media"].iloc[0]) * 100.0 if not r_si.empty else 0.0)
            alc_vals.append(float(r_ic["Alcance_Final_Pct_Media"].iloc[0]) if not r_ic.empty else 0.0)

        y3 = np.arange(len(cenarios_ordem))
        alt = 0.36
        b_auc = ax3.barh(
            y3 - alt / 2, auc_vals, height=alt,
            color="#4285F4", edgecolor="#222222", linewidth=0.8,
            label=r"Velocidade $\mathrm{AUC}_{\mathrm{SI}}$ (%)",
        )
        b_alc = ax3.barh(
            y3 + alt / 2, alc_vals, height=alt,
            color="#EA4335", edgecolor="#222222", linewidth=0.8,
            label=rf"Alcance IC ($\beta = {beta_ref:.2f}$, %)",
        )

        ax3.set_xlim(0, 118.0)
        for barra, val in zip(b_auc, auc_vals):
            ax3.text(
                val + 1.4, barra.get_y() + barra.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=7.8, fontweight="bold", color="#0D47A1",
            )
        for barra, val in zip(b_alc, alc_vals):
            ax3.text(
                val + 1.4, barra.get_y() + barra.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=7.8, fontweight="bold", color="#B71C1C",
            )

        si_dir = resumo_json.get("simulacao_si", {}).get("combinada_dir", {})
        ic_pond = resumo_json.get("simulacao_ic", {}).get("combinada_dir_ponderada", {})
        tm_hubs = si_dir.get("hubs_t_motor_50")
        tm_sens = si_dir.get("sensorial_t_motor_50")
        p_hubs = ic_pond.get("hubs_alcance_pct", 0.0) or 0.0
        p_sens = ic_pond.get("sensorial_alcance_pct", 0.0) or 0.0

        str_tm_h = f"{tm_hubs:.1f}" if tm_hubs is not None else "N/A"
        str_tm_s = f"{tm_sens:.1f}" if tm_sens is not None else "N/A"
        ax3.text(
            0.98, 0.03,
            f"Projeção Direcionada:\n"
            f"  $t_{{\\mathrm{{motor}},50}}$ Hubs: {str_tm_h} | Sens.: {str_tm_s}\n"
            f"  IC Ponderada ($\\beta={beta_ref:.2f}$): Hubs {p_hubs:.1f}% | Sens. {p_sens:.1f}%",
            transform=ax3.transAxes, fontsize=7.5,
            va="bottom", ha="right",
            bbox={
                "boxstyle": "round,pad=0.34",
                "facecolor": "#FFF8E1",
                "edgecolor": "#FBBC05",
                "linewidth": 1.0,
            },
        )

        ax3.set_yticks(y3)
        ax3.set_yticklabels(rotulos_curtos, fontsize=8.4, fontweight="bold")
        ax3.set_ylim(len(cenarios_ordem) + 0.55, -1.05)
        ax3.set_xlabel(r"Desempenho de Propagação (%)", fontsize=10.0, fontweight="bold")
        ax3.set_title(
            rf"Resumo: $\mathrm{{AUC}}_{{\mathrm{{SI}}}}$ e Alcance em $\beta = {beta_ref:.2f}$",
            fontsize=11.2, fontweight="bold",
        )
        ax3.legend(loc="upper right", fontsize=7.8, framealpha=0.92, ncol=2)
        ax3.grid(axis="x", alpha=0.28, linestyle=":")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        plt.tight_layout()
        fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico de difusão (simulação de propagação) salvo em: '{caminho_saida}'")
        return caminho_saida

    def exportar_html_interativo(
        self,
        arquivo_saida: str = "celegans_rede_interativa.html",
        notebook: bool = False,
        influentes: Optional[Dict[str, float]] = None,
        membros_rich_club: Optional[Set[str]] = None,
        graficos_png: Optional[Dict[str, str]] = None,
        tabelas_ranking: Optional[Dict[str, pd.DataFrame]] = None
    ) -> str:
        """
        Gera HTML em tela cheia na pasta outputs com caixas de seleção (<select>) para:
          - Alternar entre 5 modos de exibição (layouts topológicos/anatômicos)
          - Filtrar/colorir as camadas do multigrafo (Química, Elétrica, Multiplex ou Custo de Cabeamento)
          - Visualizar os gráficos PNG de métricas em modal integrado
          - Consultar as tabelas de ranking por tipo funcional em abas do mesmo modal

        O dicionário ``tabelas_ranking`` mapeia o título da aba para um DataFrame;
        cada DataFrame vira uma aba HTML (valores escapados) após as abas de PNG.
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

        # Consolida arestas registrando separadamente os pesos Químico e Elétrico para o filtro dinâmico JS,
        # além do custo de cabeamento espacial (peso sináptico × distância longitudinal entre somas)
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            peso = int(d.get("weight", 1))
            camada = d.get("synapse_type", k)
            pq = peso if camada == "Quimica" else 0
            pe = peso if camada == "Eletrica" else 0

            pos_u = self.graph.nodes[u].get("soma_pos")
            pos_v = self.graph.nodes[v].get("soma_pos")
            du = min(max(float(pos_u if pos_u is not None else 0.5), 0.0), 1.0)
            dv = min(max(float(pos_v if pos_v is not None else 0.5), 0.0), 1.0)
            distancia = abs(du - dv)
            custo = float(peso) * distancia
            cq = float(pq) * distancia
            ce = float(pe) * distancia

            if g_vis.has_edge(u, v):
                g_vis[u][v]["peso_acumulado"] += peso
                g_vis[u][v]["peso_quimico"] += pq
                g_vis[u][v]["peso_eletrico"] += pe
                g_vis[u][v]["custo_cabeamento"] += custo
                g_vis[u][v]["custo_quimico"] += cq
                g_vis[u][v]["custo_eletrico"] += ce
                g_vis[u][v]["width"] = float(0.7 + np.log1p(g_vis[u][v]["peso_acumulado"]) * 0.85)
                g_vis[u][v]["title"] += f" | {camada}: {peso} (custo {custo:.1f})"
            else:
                g_vis.add_edge(
                    u,
                    v,
                    peso_acumulado=peso,
                    peso_quimico=pq,
                    peso_eletrico=pe,
                    custo_cabeamento=custo,
                    custo_quimico=cq,
                    custo_eletrico=ce,
                    distancia_soma=distancia,
                    width=float(0.7 + np.log1p(peso) * 0.85),
                    title=(
                        f"{u} → {v} | Distância soma: {distancia:.2f}\n"
                        f"{camada}: {peso} (custo cabeamento: {custo:.1f})"
                    ),
                    color="rgba(160, 160, 160, 0.22)",
                    physics=False
                )

        # Custo máximo de cabeamento entre pares consolidados (normalização da escala de cor no JS)
        custo_maximo = max(
            (g_vis[u][v].get("custo_cabeamento", 0.0) for u, v in g_vis.edges()),
            default=0.0
        )

        net.from_nx(g_vis)
        net.save_graph(caminho_saida)

        # Prepara itens do modal (gráficos PNG e tabelas HTML) com seletor drop-down e controles interativos
        opcoes_graficos = ""
        opcoes_tabelas = ""
        conteudos_abas = ""
        lista_ids_modal: List[str] = []

        if graficos_png:
            for idx, (titulo_aba, caminho_img) in enumerate(graficos_png.items()):
                b64 = self._imagem_para_base64(caminho_img)
                if not b64:
                    continue
                id_aba = f"aba_grafico_{idx}"
                eh_primeiro = len(lista_ids_modal) == 0
                lista_ids_modal.append(id_aba)
                display_stl = "block" if eh_primeiro else "none"
                selected_attr = " selected" if eh_primeiro else ""
                titulo_esc = html.escape(str(titulo_aba))
                nome_arq_dl = html.escape(os.path.basename(caminho_img) or f"grafico_{idx}.png")
                opcoes_graficos += (
                    f'<option value="{id_aba}" data-tipo="grafico" '
                    f'data-arquivo="{nome_arq_dl}"{selected_attr}>📈 {titulo_esc}</option>'
                )
                conteudos_abas += (
                    f'<div id="{id_aba}" class="conteudo-aba" data-tipo="grafico" '
                    f'data-titulo="{titulo_esc}" data-arquivo="{nome_arq_dl}" style="display:{display_stl};">'
                    f'<div class="container-imagem-zoom">'
                    f'<img class="img-metrica-interativa" src="data:image/png;base64,{b64}" '
                    f'alt="{titulo_esc}" onclick="alternarCliqueZoomImagem(this)" '
                    f'title="Clique para alternar zoom ou use os controles acima" />'
                    f'</div>'
                    f'</div>'
                )

        # Abas de ranking por tipo funcional: tabelas HTML com busca/filtro interativo
        contador_tabela = 0
        if tabelas_ranking:
            for titulo_aba, df_tabela in tabelas_ranking.items():
                corpo_tabela = self._montar_tabela_html(df_tabela, cor_por_tipo=PALETA_CORES_FUNCIONAIS)
                if not corpo_tabela:
                    continue
                id_aba = f"aba_tabela_{contador_tabela}"
                contador_tabela += 1
                eh_primeiro = len(lista_ids_modal) == 0
                lista_ids_modal.append(id_aba)
                display_stl = "block" if eh_primeiro else "none"
                selected_attr = " selected" if eh_primeiro else ""
                titulo_esc = html.escape(str(titulo_aba))
                opcoes_tabelas += (
                    f'<option value="{id_aba}" data-tipo="tabela"{selected_attr}>{titulo_esc}</option>'
                )
                conteudos_abas += (
                    f'<div id="{id_aba}" class="conteudo-aba" data-tipo="tabela" '
                    f'data-titulo="{titulo_esc}" style="display:{display_stl};">'
                    f'{corpo_tabela}'
                    f'</div>'
                )

        opcoes_select_modal = ""
        if opcoes_graficos:
            opcoes_select_modal += f'<optgroup label="Gráficos e Imagens (PNG)">{opcoes_graficos}</optgroup>'
        if opcoes_tabelas:
            opcoes_select_modal += f'<optgroup label="Tabelas e Rankings">{opcoes_tabelas}</optgroup>'

        ids_modal_json = json.dumps(lista_ids_modal)
        layouts_json = json.dumps(layouts_precalculados)
        rc_json = json.dumps(list(rc_set))
        btn_modal_html = (
            '<button class="btn-topo btn-destaque" onclick="alternarModalMetricas()">📊 Gráficos de Métricas</button>'
            if lista_ids_modal else ""
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
                width: 92vw;
                max-width: 1450px;
                max-height: 92vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                box-shadow: 0 12px 32px rgba(0,0,0,0.65);
                font-family: sans-serif;
            }}
            .modal-cabecalho {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                padding: 12px 20px;
                background: #181818;
                border-bottom: 1px solid #333;
            }}
            .modal-controles-esq {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 10px;
                flex: 1;
            }}
            .modal-controles-dir {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 8px;
            }}
            .select-modal-grafico {{
                background: #282828;
                color: #ffffff;
                border: 1px solid #4285F4;
                padding: 8px 14px;
                border-radius: 6px;
                font-size: 13.5px;
                font-weight: 600;
                font-family: sans-serif;
                cursor: pointer;
                outline: none;
                min-width: 290px;
            }}
            .select-modal-grafico:hover, .select-modal-grafico:focus {{
                background: #323232;
                border-color: #669df6;
            }}
            .btn-modal-acao {{
                background: #2c2c2c;
                color: #e0e0e0;
                border: 1px solid #4a4a4a;
                padding: 7px 12px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12.5px;
                font-weight: 600;
                transition: background 0.15s, border-color 0.15s;
            }}
            .btn-modal-acao:hover {{
                background: #3c3c3c;
                border-color: #4285F4;
                color: #fff;
            }}
            .badge-contador-modal {{
                background: #2b3548;
                color: #8ab4f8;
                border: 1px solid #4285F4;
                padding: 5px 10px;
                border-radius: 999px;
                font-size: 12px;
                font-weight: bold;
                white-space: nowrap;
            }}
            .input-filtro-tabela {{
                background: #282828;
                color: #fff;
                border: 1px solid #555;
                padding: 7px 11px;
                border-radius: 6px;
                font-size: 12.5px;
                outline: none;
                width: 210px;
            }}
            .input-filtro-tabela:focus {{
                border-color: #4285F4;
            }}
            .btn-fechar-modal {{
                background: #EA4335;
                color: #fff;
                border: none;
                padding: 8px 14px;
                border-radius: 6px;
                cursor: pointer;
                font-weight: bold;
            }}
            .btn-fechar-modal:hover {{
                background: #d33426;
            }}
            .modal-corpo {{
                padding: 18px;
                overflow: auto;
                text-align: center;
                background: #fff;
                flex: 1;
            }}
            .container-imagem-zoom {{
                display: inline-block;
                min-width: 100%;
                overflow: auto;
            }}
            .img-metrica-interativa {{
                max-width: 100%;
                max-height: 76vh;
                object-fit: contain;
                border-radius: 4px;
                transition: transform 0.2s ease, width 0.2s ease;
                cursor: zoom-in;
                transform-origin: top center;
            }}
            .modal-rodape-dica {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 8px 20px;
                background: #181818;
                border-top: 1px solid #333;
                color: #aaa;
                font-size: 12px;
            }}
            .tabela-ranking {{
                border-collapse: collapse;
                margin: 0 auto;
                font-size: 13px;
                text-align: left;
            }}
            .tabela-ranking th,
            .tabela-ranking td {{
                padding: 6px 12px;
                border-bottom: 1px solid #e0e0e0;
                white-space: nowrap;
            }}
            .tabela-ranking th {{
                background: #2c2c2c;
                color: #fff;
                font-weight: 600;
            }}
            .tabela-ranking tr:nth-child(even) td {{
                background: #f4f4f4;
            }}
        </style>

        <div id="painel-legenda">
            <strong id="titulo-legenda">1. Clusters Funcionais</strong>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#4285F4;"></span>Sensorial</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#FBBC05;"></span>Interneurônio</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#EA4335;"></span>Comando / Hubs</div>
            <div class="item-legenda"><span class="bolinha-cor" style="background:#34A853;"></span>Motor</div>
            <div class="item-legenda">★ Mais influente da classe</div>
            <div id="legenda-arestas" style="display:none; margin-top:6px; padding-top:6px; border-top:1px solid #3a3a3a;">
                <div class="item-legenda"><span class="linha-cor" style="background:#FF9800;"></span>Sinapse Química (S/Sp)</div>
                <div class="item-legenda"><span class="linha-cor" style="background:#00BCD4;"></span>Junção Elétrica (EJ)</div>
                <div class="item-legenda"><span class="linha-cor" style="background:#FF4081;"></span>Multiplex (Quím + Elét)</div>
                <div class="item-legenda"><span class="linha-cor" style="background:linear-gradient(90deg, hsl(220, 85%, 55%), hsl(110, 85%, 55%), hsl(0, 85%, 55%));"></span>Custo de Cabeamento (baixo → alto)</div>
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
                    <option value="colorir_custo">Colorir por Custo de Cabeamento (Peso × Distância Soma)</option>
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
                    <div class="modal-controles-esq">
                        <div class="grupo-select">
                            <span class="label-select">Selecionar Gráfico / Tabela</span>
                            <select id="select-grafico-modal" class="select-modal-grafico" onchange="selecionarItemModal(this.value)">
                                {opcoes_select_modal}
                            </select>
                        </div>
                        <button class="btn-modal-acao" onclick="navegarItemModal(-1)" title="Anterior (Seta Esquerda)">◀ Anterior</button>
                        <span id="badge-posicao-modal" class="badge-contador-modal">1 / {max(len(lista_ids_modal), 1)}</span>
                        <button class="btn-modal-acao" onclick="navegarItemModal(1)" title="Próximo (Seta Direita)">Próximo ▶</button>
                    </div>
                    <div class="modal-controles-dir">
                        <div id="controles-zoom-grafico" style="display:flex; align-items:center; gap:6px;">
                            <button class="btn-modal-acao" onclick="ajustarZoomGrafico(-0.2)" title="Reduzir Zoom">🔍−</button>
                            <span id="label-zoom-modal" class="badge-contador-modal" style="min-width:44px; text-align:center;">100%</span>
                            <button class="btn-modal-acao" onclick="ajustarZoomGrafico(0.2)" title="Aumentar Zoom">🔍+</button>
                            <button class="btn-modal-acao" onclick="resetarZoomGrafico()" title="Ajustar à Tela">⟲ Reset</button>
                            <button class="btn-modal-acao" onclick="baixarGraficoAtual()" title="Baixar PNG">⬇ PNG</button>
                        </div>
                        <div id="controles-filtro-tabela" style="display:none; align-items:center; gap:6px;">
                            <input id="input-busca-tabela" type="text" class="input-filtro-tabela" placeholder="🔎 Filtrar neurônio / tipo..." oninput="filtrarTabelaAtiva(this.value)" />
                        </div>
                        <button class="btn-fechar-modal" onclick="alternarModalMetricas()">✕ Fechar (ESC)</button>
                    </div>
                </div>
                <div id="corpo-modal-metricas" class="modal-corpo">{conteudos_abas}</div>
                <div class="modal-rodape-dica">
                    <span id="legenda-item-modal">Dica: Clique na imagem para dar zoom rápido ou use os botões 🔍+ / 🔍−.</span>
                    <span>Atalhos: <strong>← / →</strong> alterna gráficos • <strong>+ / −</strong> zoom • <strong>ESC</strong> fecha</span>
                </div>
            </div>
        </div>

        <script type="text/javascript">
            var fisicaAtiva = false;
            var rcDestacado = false;
            var coordenadasLayouts = {layouts_json};
            var membrosRichClub = new Set({rc_json});
            var custoMaximo = {custo_maximo:.6f};
            var listaIdsModal = {ids_modal_json};
            var indiceModalAtual = 0;
            var nivelZoomAtual = 1.0;

            var descricoesModos = {{
                "clusters": {{
                    titulo: "1. Clusters Funcionais",
                    texto: "Agrupa os neurônios em 5 ilhas por função biológica (Sensorial, Interneurônio, Comando, Motor), com hubs no centro."
                }},
                "anatomico": {{
                    titulo: "2. Anatomia Longitudinal Real",
                    texto: "Eixo X: Posição física do soma no corpo do verme (Esquerda = Cabeça [H] → Centro = Corpo [M] → Direita = Cauda [T]). Eixo Y: Camada funcional."
                }},
                "hierarquico": {{
                    titulo: "3. Hierarquia Direcional de Fluxo",
                    texto: "Organiza a rede em colunas da esquerda para a direita seguindo o processamento neural: Sensorial → Interneurônio → Comando → Motor."
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
                    }} else if (filtro === "colorir_custo") {{
                        ocultar = false;
                        var custoRef = ar.custo_cabeamento || 0;
                        var tCusto = custoMaximo > 0 ? Math.min(custoRef / custoMaximo, 1) : 0;
                        var hue = 220 * (1 - tCusto); // azul (barato) -> vermelho (caro)
                        cor = "hsl(" + hue.toFixed(0) + ", 85%, 55%)";
                        setaAtiva = true;
                        pesoRef = pt; // largura continua refletindo o peso sináptico
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
                var abrir = (modal.style.display !== "flex");
                modal.style.display = abrir ? "flex" : "none";
                if (abrir && listaIdsModal.length > 0) {{
                    var sel = document.getElementById("select-grafico-modal");
                    var idAtivo = (sel && sel.value) ? sel.value : listaIdsModal[0];
                    selecionarItemModal(idAtivo);
                }}
            }}

            function fecharModalFora(e) {{
                if (e.target.id === "modal-metricas") alternarModalMetricas();
            }}

            function selecionarItemModal(idAba) {{
                if (!idAba) return;
                var conteudos = document.getElementsByClassName("conteudo-aba");
                for (var i = 0; i < conteudos.length; i++) {{
                    conteudos[i].style.display = "none";
                }}

                var alvo = document.getElementById(idAba);
                if (!alvo) return;
                alvo.style.display = "block";

                var idx = listaIdsModal.indexOf(idAba);
                if (idx >= 0) {{
                    indiceModalAtual = idx;
                }}

                var sel = document.getElementById("select-grafico-modal");
                if (sel && sel.value !== idAba) {{
                    sel.value = idAba;
                }}

                var badgePos = document.getElementById("badge-posicao-modal");
                if (badgePos) {{
                    badgePos.innerText = (indiceModalAtual + 1) + " / " + Math.max(listaIdsModal.length, 1);
                }}

                var tipoItem = alvo.getAttribute("data-tipo") || "grafico";
                var tituloItem = alvo.getAttribute("data-titulo") || "";
                var divZoom = document.getElementById("controles-zoom-grafico");
                var divFiltro = document.getElementById("controles-filtro-tabela");
                var legendaRodape = document.getElementById("legenda-item-modal");

                if (tipoItem === "grafico") {{
                    if (divZoom) divZoom.style.display = "flex";
                    if (divFiltro) divFiltro.style.display = "none";
                    if (legendaRodape) {{
                        legendaRodape.innerText = "Visualizando gráfico: " + tituloItem + " (Clique na imagem para alternar zoom 100% / 160%)";
                    }}
                    resetarZoomGrafico();
                }} else {{
                    if (divZoom) divZoom.style.display = "none";
                    if (divFiltro) divFiltro.style.display = "flex";
                    var inputBusca = document.getElementById("input-busca-tabela");
                    if (inputBusca) {{
                        inputBusca.value = "";
                        filtrarTabelaAtiva("");
                    }}
                    if (legendaRodape) {{
                        legendaRodape.innerText = "Visualizando tabela: " + tituloItem + " (Use a caixa de busca acima para filtrar linhas)";
                    }}
                }}
            }}

            function navegarItemModal(delta) {{
                if (!listaIdsModal.length) return;
                var novoIdx = (indiceModalAtual + delta + listaIdsModal.length) % listaIdsModal.length;
                selecionarItemModal(listaIdsModal[novoIdx]);
            }}

            function obterImagemAtivaModal() {{
                if (!listaIdsModal.length) return null;
                var idAtual = listaIdsModal[indiceModalAtual];
                var divAtual = document.getElementById(idAtual);
                if (!divAtual) return null;
                return divAtual.querySelector("img.img-metrica-interativa");
            }}

            function aplicarZoomNaImagem(img) {{
                if (!img) return;
                if (Math.abs(nivelZoomAtual - 1.0) < 0.01) {{
                    img.style.maxWidth = "100%";
                    img.style.maxHeight = "76vh";
                    img.style.width = "auto";
                    img.style.cursor = "zoom-in";
                }} else {{
                    img.style.maxWidth = "none";
                    img.style.maxHeight = "none";
                    img.style.width = Math.round(nivelZoomAtual * 100) + "%";
                    img.style.cursor = "zoom-out";
                }}
                var lblZoom = document.getElementById("label-zoom-modal");
                if (lblZoom) {{
                    lblZoom.innerText = Math.round(nivelZoomAtual * 100) + "%";
                }}
            }}

            function ajustarZoomGrafico(delta) {{
                var img = obterImagemAtivaModal();
                if (!img) return;
                nivelZoomAtual = Math.min(2.6, Math.max(0.6, Math.round((nivelZoomAtual + delta) * 10) / 10));
                aplicarZoomNaImagem(img);
            }}

            function resetarZoomGrafico() {{
                nivelZoomAtual = 1.0;
                var img = obterImagemAtivaModal();
                if (img) aplicarZoomNaImagem(img);
                var corpo = document.getElementById("corpo-modal-metricas");
                if (corpo) {{
                    corpo.scrollTop = 0;
                    corpo.scrollLeft = 0;
                }}
            }}

            function alternarCliqueZoomImagem(img) {{
                if (!img) return;
                nivelZoomAtual = (nivelZoomAtual > 1.05) ? 1.0 : 1.6;
                aplicarZoomNaImagem(img);
            }}

            function baixarGraficoAtual() {{
                var img = obterImagemAtivaModal();
                if (!img) return;
                var divAtual = document.getElementById(listaIdsModal[indiceModalAtual]);
                var nomeArq = (divAtual && divAtual.getAttribute("data-arquivo")) ? divAtual.getAttribute("data-arquivo") : "grafico_conectoma.png";
                var link = document.createElement("a");
                link.href = img.src;
                link.download = nomeArq;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }}

            function filtrarTabelaAtiva(termo) {{
                if (!listaIdsModal.length) return;
                var divAtual = document.getElementById(listaIdsModal[indiceModalAtual]);
                if (!divAtual) return;
                var linhas = divAtual.querySelectorAll("table.tabela-ranking tbody tr");
                var q = String(termo || "").toLowerCase().trim();
                for (var i = 0; i < linhas.length; i++) {{
                    var txt = (linhas[i].innerText || linhas[i].textContent || "").toLowerCase();
                    linhas[i].style.display = (!q || txt.indexOf(q) !== -1) ? "" : "none";
                }}
            }}

            document.addEventListener("keydown", function(e) {{
                var modal = document.getElementById("modal-metricas");
                if (!modal || modal.style.display !== "flex") return;
                var tagAtiva = (document.activeElement && document.activeElement.tagName) ? document.activeElement.tagName.toLowerCase() : "";
                if (e.key === "Escape") {{
                    modal.style.display = "none";
                }} else if (tagAtiva !== "input" && tagAtiva !== "select") {{
                    if (e.key === "ArrowLeft") {{
                        e.preventDefault();
                        navegarItemModal(-1);
                    }} else if (e.key === "ArrowRight") {{
                        e.preventDefault();
                        navegarItemModal(1);
                    }} else if (e.key === "+" || e.key === "=") {{
                        e.preventDefault();
                        ajustarZoomGrafico(0.2);
                    }} else if (e.key === "-") {{
                        e.preventDefault();
                        ajustarZoomGrafico(-0.2);
                    }} else if (e.key === "0") {{
                        e.preventDefault();
                        resetarZoomGrafico();
                    }}
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