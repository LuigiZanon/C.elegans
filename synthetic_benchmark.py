"""
Módulo de benchmark comparativo entre o conectoma real do C. elegans e modelos
de redes sintéticas equivalentes (Erdős-Rényi, Watts-Strogatz e Barabási-Albert),
incluindo índices formais de Small-Worldness (sigma e omega) e análise da cauda
da distribuição de grau via Monte Carlo.
"""

import warnings
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx


class SyntheticNetworkBenchmark:
    """
    Projeta o conectoma em grafo simples não-direcionado, calibra modelos sintéticos
    equivalentes (ER, WS, BA e Lattice), executa simulações Monte Carlo e gera o
    diagnóstico topológico formal (Small-World vs. Scale-Free vs. Aleatório).
    """

    MODELO_REAL = "C. elegans (Real)"
    MODELO_ER = "Erdős-Rényi (ER)"
    MODELO_WS = "Watts-Strogatz (WS)"
    MODELO_BA = "Barabási-Albert (BA)"

    CHAVES_METRICAS = (
        "N",
        "M",
        "Densidade",
        "Grau_Medio",
        "Grau_Desvio",
        "Grau_Max",
        "Heterogeneidade",
        "Agrupamento_C",
        "Transitividade_T",
        "Caminho_Medio_L",
        "Diametro_D",
        "Eficiencia_Global",
        "Assortatividade_r",
        "Fracao_GCC",
    )

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        n_runs: int = 30,
        seed_base: int = 42,
        ws_rewiring_p: float = 0.10,
    ) -> None:
        self.graph = graph
        self.n_runs = int(n_runs)
        self.seed_base = int(seed_base)
        self.ws_rewiring_p = float(ws_rewiring_p)

        self.g_real: nx.Graph = self._projetar_grafo_simples()
        self.n: int = self.g_real.number_of_nodes()
        self.m: int = self.g_real.number_of_edges()
        self.k_medio: float = (2.0 * self.m / self.n) if self.n > 0 else 0.0
        self.densidade: float = (
            (2.0 * self.m / (self.n * (self.n - 1))) if self.n > 1 else 0.0
        )

        # Parâmetros calibrados para modelos sintéticos equivalentes
        self.k_ws: int = max(2, 2 * int(round(self.k_medio / 2.0)))
        if self.k_ws >= self.n and self.n >= 3:
            self.k_ws = self.n - 1 if (self.n - 1) % 2 == 0 else self.n - 2
        self.m_ba: int = max(1, int(round(self.k_medio / 2.0)))
        if self.m_ba >= self.n and self.n >= 2:
            self.m_ba = self.n - 1

    def _projetar_grafo_simples(self) -> nx.Graph:
        """
        Converte o MultiDiGraph (camadas Química e Elétrica) em um grafo simples
        não-direcionado preservando todos os vértices e removendo auto-laços.
        """
        g_simples = nx.Graph()
        g_simples.add_nodes_from(self.graph.nodes())
        for u, v in self.graph.edges():
            if u != v:
                g_simples.add_edge(u, v)
        g_simples.remove_edges_from(nx.selfloop_edges(g_simples))
        return g_simples

    def _gerar_er(self, seed: int) -> nx.Graph:
        """Gera rede Erdős-Rényi G(N, M) com equivalência exata de nós e arestas."""
        return nx.gnm_random_graph(n=self.n, m=self.m, seed=seed)

    def _gerar_ws(self, seed: int) -> nx.Graph:
        """Gera rede Watts-Strogatz com grau par calibrado k_ws e probabilidade p."""
        return nx.watts_strogatz_graph(
            n=self.n, k=self.k_ws, p=self.ws_rewiring_p, seed=seed
        )

    def _gerar_ba(self, seed: int) -> nx.Graph:
        """Gera rede Barabási-Albert com parâmetro de ligação preferencial m_ba."""
        return nx.barabasi_albert_graph(n=self.n, m=self.m_ba, seed=seed)

    def _gerar_lattice(self) -> nx.Graph:
        """Gera anel regular (lattice, p=0) equivalente para o cálculo do índice omega."""
        return nx.watts_strogatz_graph(
            n=self.n, k=self.k_ws, p=0.0, seed=self.seed_base
        )

    @staticmethod
    def _calcular_metricas_grafo(g: nx.Graph) -> Dict[str, float]:
        """
        Extrai métricas topológicas com segurança para grafos potencialmente desconexos,
        calculando caminho mínimo médio e diâmetro exclusivamente na Maior Componente Conexa (GCC).
        """
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()

        if n_nodes == 0:
            return {chave: 0.0 for chave in SyntheticNetworkBenchmark.CHAVES_METRICAS}

        if nx.is_connected(g):
            gcc = g
        else:
            maior_comp = max(nx.connected_components(g), key=len)
            gcc = g.subgraph(maior_comp).copy()

        graus = np.array([deg for _, deg in g.degree()], dtype=float)
        grau_medio = float(np.mean(graus))
        grau_desvio = float(np.std(graus, ddof=1)) if n_nodes > 1 else 0.0
        grau_max = float(np.max(graus))
        heterogeneidade = (
            float(np.mean(graus ** 2) / (grau_medio ** 2)) if grau_medio > 0.0 else 0.0
        )

        agrupamento_c = float(nx.average_clustering(g))
        transitividade_t = float(nx.transitivity(g))

        if gcc.number_of_nodes() > 1:
            caminho_medio_l = float(nx.average_shortest_path_length(gcc))
            diametro_d = float(nx.diameter(gcc))
        else:
            caminho_medio_l = 0.0
            diametro_d = 0.0

        eficiencia_global = float(nx.global_efficiency(g))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                assortatividade = float(nx.degree_assortativity_coefficient(g))
            except Exception:
                assortatividade = 0.0
        if np.isnan(assortatividade) or np.isinf(assortatividade):
            assortatividade = 0.0

        fracao_gcc = float(gcc.number_of_nodes() / n_nodes)

        return {
            "N": float(n_nodes),
            "M": float(n_edges),
            "Densidade": float(nx.density(g)),
            "Grau_Medio": grau_medio,
            "Grau_Desvio": grau_desvio,
            "Grau_Max": grau_max,
            "Heterogeneidade": heterogeneidade,
            "Agrupamento_C": agrupamento_c,
            "Transitividade_T": transitividade_t,
            "Caminho_Medio_L": caminho_medio_l,
            "Diametro_D": diametro_d,
            "Eficiencia_Global": eficiencia_global,
            "Assortatividade_r": assortatividade,
            "Fracao_GCC": fracao_gcc,
        }

    def _agregar_execucoes(
        self, nome_modelo: str, lista_metricas: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Agrega média e desvio padrão amostral das métricas de um conjunto de execuções."""
        linha: Dict[str, Any] = {"Modelo": nome_modelo}
        for chave in self.CHAVES_METRICAS:
            valores = np.array([m[chave] for m in lista_metricas], dtype=float)
            linha[f"{chave}_Media"] = float(np.mean(valores))
            if len(valores) > 1:
                linha[f"{chave}_Std"] = float(np.std(valores, ddof=1))
            else:
                linha[f"{chave}_Std"] = 0.0
        return linha

    def executar_benchmark(self) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Executa o benchmark Monte Carlo comparando o conectoma real com n_runs
        realizações independentes de ER, WS e BA.
        """
        linhas_resultado: List[Dict[str, Any]] = []
        graus_por_modelo: Dict[str, np.ndarray] = {}

        # 1. Rede real (1 avaliação determinística, desvio padrão 0.0)
        metricas_real = self._calcular_metricas_grafo(self.g_real)
        linhas_resultado.append(
            self._agregar_execucoes(self.MODELO_REAL, [metricas_real])
        )
        graus_por_modelo[self.MODELO_REAL] = np.array(
            [deg for _, deg in self.g_real.degree()], dtype=int
        )

        # 2. Modelos sintéticos equivalentes via Monte Carlo
        geradores: List[Tuple[str, Callable[[int], nx.Graph]]] = [
            (self.MODELO_ER, self._gerar_er),
            (self.MODELO_WS, self._gerar_ws),
            (self.MODELO_BA, self._gerar_ba),
        ]

        for nome_modelo, fn_gerador in geradores:
            metricas_rodadas: List[Dict[str, float]] = []
            graus_acumulados: List[np.ndarray] = []

            for i in range(self.n_runs):
                seed = self.seed_base + i
                g_sint = fn_gerador(seed)
                metricas_rodadas.append(self._calcular_metricas_grafo(g_sint))
                graus_acumulados.append(
                    np.array([deg for _, deg in g_sint.degree()], dtype=int)
                )

            linhas_resultado.append(
                self._agregar_execucoes(nome_modelo, metricas_rodadas)
            )
            graus_por_modelo[nome_modelo] = np.concatenate(graus_acumulados)

        # Ordena as colunas garantindo todos os campos do contrato de dados
        colunas_ordenadas = [
            "Modelo",
            "N_Media",
            "N_Std",
            "M_Media",
            "M_Std",
            "Densidade_Media",
            "Densidade_Std",
            "Grau_Medio_Media",
            "Grau_Medio_Std",
            "Grau_Desvio_Media",
            "Grau_Desvio_Std",
            "Grau_Max_Media",
            "Grau_Max_Std",
            "Heterogeneidade_Media",
            "Heterogeneidade_Std",
            "Agrupamento_C_Media",
            "Agrupamento_C_Std",
            "Transitividade_T_Media",
            "Transitividade_T_Std",
            "Caminho_Medio_L_Media",
            "Caminho_Medio_L_Std",
            "Diametro_D_Media",
            "Diametro_D_Std",
            "Eficiencia_Global_Media",
            "Eficiencia_Global_Std",
            "Assortatividade_r_Media",
            "Assortatividade_r_Std",
            "Fracao_GCC_Media",
            "Fracao_GCC_Std",
        ]
        df_comparacao = pd.DataFrame(linhas_resultado)[colunas_ordenadas]
        return df_comparacao, graus_por_modelo

    def calcular_diagnostico_topologico(
        self,
        df_comparacao: pd.DataFrame,
        graus_por_modelo: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Calcula índices formais de Small-Worldness (sigma e omega) e realiza o teste
        estatístico da cauda da distribuição de grau (Lei de Potência vs. Exponencial)
        em NumPy puro.
        """
        por_modelo = df_comparacao.set_index("Modelo")

        c_real = float(por_modelo.loc[self.MODELO_REAL, "Agrupamento_C_Media"])
        c_er = float(por_modelo.loc[self.MODELO_ER, "Agrupamento_C_Media"])
        c_ws = float(por_modelo.loc[self.MODELO_WS, "Agrupamento_C_Media"])
        c_ba = float(por_modelo.loc[self.MODELO_BA, "Agrupamento_C_Media"])

        l_real = float(por_modelo.loc[self.MODELO_REAL, "Caminho_Medio_L_Media"])
        l_er = float(por_modelo.loc[self.MODELO_ER, "Caminho_Medio_L_Media"])
        l_ws = float(por_modelo.loc[self.MODELO_WS, "Caminho_Medio_L_Media"])
        l_ba = float(por_modelo.loc[self.MODELO_BA, "Caminho_Medio_L_Media"])

        het_real = float(por_modelo.loc[self.MODELO_REAL, "Heterogeneidade_Media"])
        het_er = float(por_modelo.loc[self.MODELO_ER, "Heterogeneidade_Media"])
        het_ba = float(por_modelo.loc[self.MODELO_BA, "Heterogeneidade_Media"])

        # Lattice equivalente para o índice omega (Telesford et al., 2011)
        g_lattice = self._gerar_lattice()
        c_lattice = float(nx.average_clustering(g_lattice))

        razao_c_er = float(c_real / c_er) if c_er > 0.0 else 0.0
        razao_l_er = float(l_real / l_er) if l_er > 0.0 else 0.0
        sigma_sw = float(razao_c_er / razao_l_er) if razao_l_er > 0.0 else 0.0

        termo_l = (l_er / l_real) if l_real > 0.0 else 0.0
        termo_c = (c_real / c_lattice) if c_lattice > 0.0 else 0.0
        omega_sw = float(termo_l - termo_c)

        # Análise da cauda superior da distribuição de grau do conectoma real
        graus_real = np.asarray(graus_por_modelo[self.MODELO_REAL], dtype=float)
        k_min_cauda = int(np.ceil(self.k_medio))
        cauda = graus_real[graus_real >= k_min_cauda]
        if len(cauda) < 5:
            k_min_cauda = max(1, int(np.percentile(graus_real, 50)))
            cauda = graus_real[graus_real >= k_min_cauda]

        # CCDF empírica P(K >= k) para k >= k_min
        k_unicos = np.unique(cauda)
        ccdf = np.array([np.mean(graus_real >= k) for k in k_unicos], dtype=float)

        log_k = np.log(k_unicos)
        log_p = np.log(ccdf)
        ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))

        if len(k_unicos) >= 2 and ss_tot > 0.0:
            # Ajuste Lei de Potência (log-log)
            coef_pl = np.polyfit(log_k, log_p, 1)
            pred_pl = coef_pl[0] * log_k + coef_pl[1]
            r2_log_log = float(1.0 - np.sum((log_p - pred_pl) ** 2) / ss_tot)

            # Ajuste Exponencial (semi-log)
            coef_exp = np.polyfit(k_unicos, log_p, 1)
            pred_exp = coef_exp[0] * k_unicos + coef_exp[1]
            r2_semi_log = float(1.0 - np.sum((log_p - pred_exp) ** 2) / ss_tot)
        else:
            r2_log_log = 0.0
            r2_semi_log = 0.0

        # Estimadores de Máxima Verossimilhança (MLE) na cauda com correção de continuidade
        n_tail = int(len(cauda))
        x_min_cont = max(0.5, float(k_min_cauda) - 0.5)
        soma_log_razao = float(np.sum(np.log(cauda / x_min_cont)))
        gamma_mle = float(1.0 + n_tail / soma_log_razao) if soma_log_razao > 0.0 else 0.0

        deslocamento_medio = float(np.mean(cauda) - x_min_cont)
        lambda_exp_mle = float(1.0 / deslocamento_medio) if deslocamento_medio > 0.0 else 0.0

        # Log-verossimilhança aproximada na cauda contínua [x_min_cont, +inf)
        if gamma_mle > 1.0 and lambda_exp_mle > 0.0:
            ll_pl = float(
                n_tail * np.log((gamma_mle - 1.0) / x_min_cont)
                - gamma_mle * soma_log_razao
            )
            ll_exp = float(
                n_tail * np.log(lambda_exp_mle)
                - lambda_exp_mle * float(np.sum(cauda - x_min_cont))
            )
            delta_ll = float(ll_pl - ll_exp)
        else:
            delta_ll = 0.0

        # Veredito formal baseado nas evidências matemáticas
        eh_small_world = (
            sigma_sw > 1.0 and razao_c_er > 2.0 and abs(omega_sw) <= 0.5
        )
        favorece_lei_potencia = (delta_ll > 0.0) and (r2_log_log > r2_semi_log)

        if eh_small_world and favorece_lei_potencia:
            veredito = (
                f"Rede Small-World (sigma={sigma_sw:.2f} > 1, omega={omega_sw:+.2f} ~ 0) "
                f"com assinatura compatível com Livre de Escala na cauda "
                f"(gamma_MLE={gamma_mle:.2f}, R2_log-log={r2_log_log:.3f} > "
                f"R2_semi-log={r2_semi_log:.3f}, kappa={het_real:.2f})."
            )
        elif eh_small_world:
            veredito = (
                f"Rede fortemente Small-World (sigma={sigma_sw:.2f} > 1, "
                f"omega={omega_sw:+.2f} ~ 0) com cauda pesada exponencialmente truncada "
                f"(R2_semi-log={r2_semi_log:.3f} vs. R2_log-log={r2_log_log:.3f}, "
                f"delta_LL={delta_ll:+.2f}, kappa={het_real:.2f} > kappa_ER={het_er:.2f}) — "
                f"não é aleatória nem estritamente Livre de Escala pura."
            )
        elif sigma_sw > 1.0 and razao_c_er > 1.5:
            veredito = (
                f"Rede com traços Small-World (sigma={sigma_sw:.2f}, omega={omega_sw:+.2f}) "
                f"e heterogeneidade kappa={het_real:.2f}."
            )
        else:
            veredito = (
                f"Topologia próxima ao regime aleatório (sigma={sigma_sw:.2f}, "
                f"omega={omega_sw:+.2f}, kappa={het_real:.2f})."
            )

        return {
            "N": int(self.n),
            "M": int(self.m),
            "k_medio": float(self.k_medio),
            "k_ws": int(self.k_ws),
            "m_ba": int(self.m_ba),
            "p_ws": float(self.ws_rewiring_p),
            "C_real": float(c_real),
            "C_ER": float(c_er),
            "C_WS": float(c_ws),
            "C_BA": float(c_ba),
            "C_lattice": float(c_lattice),
            "L_real": float(l_real),
            "L_ER": float(l_er),
            "L_WS": float(l_ws),
            "L_BA": float(l_ba),
            "razao_C_ER": float(razao_c_er),
            "razao_L_ER": float(razao_l_er),
            "sigma_small_world": float(sigma_sw),
            "omega_small_world": float(omega_sw),
            "k_min_cauda": int(k_min_cauda),
            "gamma_mle": float(gamma_mle),
            "lambda_exp_mle": float(lambda_exp_mle),
            "r2_log_log": float(r2_log_log),
            "r2_semi_log": float(r2_semi_log),
            "delta_log_likelihood": float(delta_ll),
            "heterogeneidade_real": float(het_real),
            "heterogeneidade_ER": float(het_er),
            "heterogeneidade_BA": float(het_ba),
            "veredito_classificacao": str(veredito),
        }

    def exibir_relatorio_console(
        self,
        df_comparacao: pd.DataFrame,
        diagnostico: Dict[str, Any],
    ) -> None:
        """Exibe tabela comparativa formatada (Media +/- Std) e o diagnóstico topológico."""
        print("\n" + "=" * 96)
        print(
            f"  BENCHMARK TOPOLÓGICO: CONECTOMA REAL VS. MODELOS SINTÉTICOS "
            f"(Monte Carlo, n={self.n_runs}, seed={self.seed_base})"
        )
        print("=" * 96)
        print(
            f"Calibração: N={diagnostico['N']} | M={diagnostico['M']} | "
            f"<k>={diagnostico['k_medio']:.2f} | WS(k={diagnostico['k_ws']}, "
            f"p={diagnostico['p_ws']:.2f}) | BA(m={diagnostico['m_ba']})"
        )
        print("-" * 96)

        metricas_tabela = [
            ("Agrupamento (C)", "Agrupamento_C", 4),
            ("Caminho (L)", "Caminho_Medio_L", 4),
            ("Efic. Glob.", "Eficiencia_Global", 4),
            ("Assort. (r)", "Assortatividade_r", 4),
            ("Heterog. (k)", "Heterogeneidade", 3),
            ("Grau Max", "Grau_Max", 1),
        ]

        linhas_fmt = []
        for _, row in df_comparacao.iterrows():
            registro = {"Modelo": row["Modelo"]}
            eh_real = row["Modelo"] == self.MODELO_REAL
            for rotulo, base, casas in metricas_tabela:
                media = row[f"{base}_Media"]
                std = row[f"{base}_Std"]
                if eh_real:
                    registro[rotulo] = f"{media:.{casas}f}"
                else:
                    registro[rotulo] = f"{media:.{casas}f} +/- {std:.{casas}f}"
            linhas_fmt.append(registro)

        df_print = pd.DataFrame(linhas_fmt)
        print(df_print.to_string(index=False))

        print("-" * 96)
        print("DIAGNÓSTICO TOPOLÓGICO FORMAL:")
        print(
            f"  • Razão de Agrupamento (C_real / C_ER):       {diagnostico['razao_C_ER']:.2f}x "
            f"(C_real={diagnostico['C_real']:.4f} vs C_ER={diagnostico['C_ER']:.4f}, "
            f"C_latt={diagnostico['C_lattice']:.4f})"
        )
        print(
            f"  • Razão de Caminho Médio (L_real / L_ER):     {diagnostico['razao_L_ER']:.2f}x "
            f"(L_real={diagnostico['L_real']:.4f} vs L_ER={diagnostico['L_ER']:.4f})"
        )
        print(
            f"  • Small-Worldness sigma (Humphries & Gurney): {diagnostico['sigma_small_world']:.3f}  "
            f"[critério: sigma > 1]"
        )
        print(
            f"  • Small-Worldness omega (Telesford et al.):   {diagnostico['omega_small_world']:+.3f}  "
            f"[critério: |omega| <= 0.5; -1=Lattice, +1=ER]"
        )
        print(
            f"  • Cauda de Grau (k >= {diagnostico['k_min_cauda']}): "
            f"gamma_MLE={diagnostico['gamma_mle']:.3f} | "
            f"lambda_EXP={diagnostico['lambda_exp_mle']:.4f} | "
            f"R²_log-log={diagnostico['r2_log_log']:.4f} vs "
            f"R²_semi-log={diagnostico['r2_semi_log']:.4f} | "
            f"ΔLL={diagnostico['delta_log_likelihood']:+.2f}"
        )
        print(
            f"  • Heterogeneidade (kappa = <k²>/<k>²):        "
            f"Real={diagnostico['heterogeneidade_real']:.3f} | "
            f"ER={diagnostico['heterogeneidade_ER']:.3f} | "
            f"BA={diagnostico['heterogeneidade_BA']:.3f}"
        )
        print(f"  • Veredito: {diagnostico['veredito_classificacao']}")
        print("=" * 96 + "\n")
