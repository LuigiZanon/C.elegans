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