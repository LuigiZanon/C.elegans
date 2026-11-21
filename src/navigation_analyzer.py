"""Módulo de Navegabilidade (Greedy Routing) e Assortatividade do conectoma C. elegans."""

import math
import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from graph_utils import projetar_grafo, resumir_contra_nulo, sanitizar_json

ESTRATEGIAS_ORDEM = ["Espacial (soma_pos)", "Grau (Hubs)", "Híbrido (Espacial + Grau)", "Caminhada Aleatória"]


def _float_finito(valor: Any, default: Optional[float] = 0.0) -> Optional[float]:
    if valor is None:
        return default
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(v) or math.isinf(v)) else v


def _classificar_assortatividade(r: float) -> str:
    if r >= 0.25:
        return "Fortemente Assortativo / Homofílico"
    if r >= 0.05:
        return "Moderadamente Assortativo"
    if r > -0.05:
        return "Neutro (Sem Correlação Relevante)"
    return "Moderadamente Disassortativo" if r > -0.25 else "Fortemente Disassortativo"


class NavigationAssortativityAnalyzer:
    """Analisa navegabilidade via Roteamento Guloso e assortatividade estrutural, funcional e espacial."""

    def __init__(
        self, graph: nx.MultiDiGraph, membros_rich_club: Optional[Set[str]] = None,
        n_runs_null: int = 20, n_pares_nulo: int = 5000, max_hops_fator: int = 3,
        alpha_hibrido: float = 0.55, seed_base: int = 42,
    ) -> None:
        self.graph = graph
        self.n_runs_null = max(1, int(n_runs_null))
        self.n_pares_nulo = max(100, int(n_pares_nulo))
        self.max_hops_fator = max(1, int(max_hops_fator))
        self.alpha_hibrido = float(min(max(alpha_hibrido, 0.0), 1.0))
        self.seed_base = int(seed_base)
        self.soma_pos_map, self.n_soma_pos_ausentes = self._sanitizar_soma_pos()
        self.g_comb_undir = self._projetar_com_soma("Combinada", False)
        self.g_comb_dir = self._projetar_com_soma("Combinada", True)
        self.g_quim_dir = self._projetar_com_soma("Quimica", True)
        self.g_elet_undir = self._projetar_com_soma("Eletrica", False)
        self.membros_rich_club: Set[str] = (
            set(membros_rich_club) & set(self.graph.nodes)
            if membros_rich_club else {n for n, _ in sorted(self.g_comb_undir.degree(), key=lambda x: (-x[1], str(x[0])))[:6]}
        )

    def _sanitizar_soma_pos(self) -> Tuple[Dict[str, float], int]:
        validos, brutos = [], {}
        for no, d in self.graph.nodes(data=True):
            v = _float_finito(d.get("soma_pos"), default=None)
            ok = v if (v is not None and 0.0 <= v <= 1.0) else None
            brutos[no] = ok
            if ok is not None:
                validos.append(ok)
        med = min(max(float(np.median(validos)) if validos else 0.5, 0.0), 1.0)
        return {no: (med if v is None else min(max(v, 0.0), 1.0)) for no, v in brutos.items()}, sum(v is None for v in brutos.values())

    def _projetar_com_soma(self, camada: str, direcionado: bool) -> nx.Graph:
        g = projetar_grafo(self.graph, camada=camada, direcionado=direcionado, preservar_atributos_nos=True)
        for no in g.nodes:
            g.nodes[no]["soma_pos"] = self.soma_pos_map[no]
        return g

    def _preparar_estruturas_roteamento(self, g: nx.Graph, soma_pos_override: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        nos = sorted(g.nodes(), key=str)
        n, no_idx = len(nos), {no: i for i, no in enumerate(nos)}
        mpos = soma_pos_override if soma_pos_override is not None else self.soma_pos_map
        graus = [int(g.out_degree(no) if g.is_directed() else g.degree(no)) for no in nos]
        kmax = max(graus) if graus else 0
        adj_list = [[no_idx[v] for v in g.neighbors(no) if v != no] for no in nos]
        pares_alc, max_dist, soma_inv = [], 0, 0.0
        for src, dists in nx.all_pairs_shortest_path_length(g):
            si = no_idx[src]
            for dst, dsp in dists.items():
                if dst != src and dsp >= 1:
                    pares_alc.append((si, no_idx[dst], int(dsp)))
                    max_dist = max(max_dist, int(dsp))
                    soma_inv += 1.0 / float(dsp)
        tot = n * (n - 1)
        return {
            "n": n, "nos": nos, "x_pos": [float(mpos[no]) for no in nos],
            "tipos": [str(g.nodes[no].get("tipo_funcional", "N/A")) for no in nos],
            "is_rc": [no in self.membros_rich_club for no in nos], "graus": graus,
            "graus_norm": [float(k) / float(kmax) if kmax > 0 else 0.0 for k in graus],
            "adj_list": adj_list, "adj_set": [set(v) for v in adj_list],
            "adj_by_deg": [sorted(v, key=lambda idx: (-graus[idx], idx)) for v in adj_list],
            "pares_alc": pares_alc, "max_hops": max(25, self.max_hops_fator * max(1, max_dist)),
            "frac_alcancavel": float(len(pares_alc)) / float(tot) if tot > 0 else 0.0,
            "e_glob": soma_inv / float(tot) if tot > 0 else 0.0,
        }

    @staticmethod
    def _executar_caminhada(s: int, t: int, adj_set: List[Set[int]], max_hops: int, is_rc: List[bool], escolher: Callable[[int, int, Set[int]], int]) -> Tuple[bool, int, bool]:
        u, vis, passos, via_rc = s, {s}, 0, False
        while passos < max_hops:
            if t in adj_set[u]:
                return True, passos + 1, via_rc
            prox = escolher(u, t, vis)
            if prox < 0 or prox in vis:
                return False, passos, via_rc
            u = prox; vis.add(u); passos += 1
            if is_rc[u]:
                via_rc = True
        return False, passos, via_rc

    def _criar_seletor(self, est: Dict[str, Any], estrategia: str, py_rng: random.Random) -> Callable[[int, int, Set[int]], int]:
        xp, gr, gn, al, ad = est["x_pos"], est["graus"], est["graus_norm"], est["adj_list"], est["adj_by_deg"]
        a, ua = self.alpha_hibrido, 1.0 - self.alpha_hibrido
        if estrategia == "Espacial (soma_pos)":
            return lambda u, t, _v: min(al[u], key=lambda v: (abs(xp[v] - xp[t]), -gr[v], v)) if al[u] else -1
        if estrategia == "Grau (Hubs)":
            return lambda u, _t, vis: next((v for v in ad[u] if v not in vis), -1)
        if estrategia == "Híbrido (Espacial + Grau)":
            return lambda u, t, vis: min((v for v in al[u] if v not in vis), key=lambda v: (a * abs(xp[v] - xp[t]) - ua * gn[v], -gr[v], v), default=-1)
        if estrategia == "Caminhada Aleatória":
            def _rnd(u: int, _t: int, vis: Set[int]) -> int:
                c = [v for v in al[u] if v not in vis]
                return c[py_rng.randrange(len(c))] if c else -1
            return _rnd
        raise ValueError(f"Estratégia desconhecida: {estrategia}")

    def _simular_estrategia(self, est: Dict[str, Any], estrategia: str, pares: List[Tuple[int, int, int]], seed: int = 42, coletar_fluxo: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        n, tipos, is_rc, adj_set, mh = est["n"], est["tipos"], est["is_rc"], est["adj_set"], est["max_hops"]
        seletor = self._criar_seletor(est, estrategia, random.Random(int(np.random.default_rng(seed).integers(0, 2**32 - 1))))
        n_pares, stretches, hops_suc, fluxo = len(pares), [], [], {}
        soma_sp = n_suc = n_ot = n_d2 = n_rc = 0
        soma_inv_g = 0.0

        for s, t, d_sp in pares:
            soma_sp += d_sp
            ok, passos, via_rc = self._executar_caminhada(s, t, adj_set, mh, is_rc, seletor)
            if ok:
                n_suc += 1; stretches.append(float(passos) / float(d_sp)); hops_suc.append(passos)
                soma_inv_g += 1.0 / float(passos); n_ot += int(passos == d_sp)
                if d_sp >= 2:
                    n_d2 += 1; n_rc += int(via_rc)
            if coletar_fluxo:
                b = fluxo.setdefault((tipos[s], tipos[t]), [0, 0, []])
                b[0] += 1
                if ok:
                    b[1] += 1; b[2].append(float(passos) / float(d_sp))

        tot = n * (n - 1)
        if n_pares == len(est["pares_alc"]):
            e_nav, e_glob = (soma_inv_g / float(tot) if tot > 0 else 0.0), est["e_glob"]
        else:
            fator = est["frac_alcancavel"] / float(n_pares) if n_pares > 0 else 0.0
            e_nav, e_glob = soma_inv_g * fator, sum(1.0 / float(d) for _, _, d in pares) * fator

        arr_t = np.asarray(stretches, dtype=float)
        met = {
            "N_Pares_Alcancaveis": n_pares, "Fracao_Alcancavel": float(est["frac_alcancavel"]),
            "Taxa_Sucesso_Pct": (100.0 * n_suc / n_pares) if n_pares > 0 else 0.0,
            "Stretch_Medio": float(np.mean(arr_t)) if stretches else float("nan"),
            "Stretch_Mediana": float(np.median(arr_t)) if stretches else float("nan"),
            "Stretch_P90": float(np.percentile(arr_t, 90)) if stretches else float("nan"),
            "Frac_Otimo_Pct": (100.0 * n_ot / n_suc) if n_suc > 0 else 0.0,
            "Hops_Greedy_Medio": float(np.mean(hops_suc)) if stretches else float("nan"),
            "Hops_SP_Medio": (float(soma_sp) / float(n_pares)) if n_pares > 0 else float("nan"),
            "Eficiencia_Nav": float(e_nav), "Eficiencia_Global": float(e_glob),
            "Razao_Eficiencia": float(min(max(e_nav / e_glob if e_glob > 0 else 0.0, 0.0), 1.0)),
            "Pct_Rotas_Via_RichClub": (100.0 * n_rc / n_d2) if n_d2 > 0 else 0.0,
        }
        lf = [
            {"Estrategia": estrategia, "Tipo_Origem": to, "Tipo_Destino": td, "N_Pares": npf, "N_Sucesso": nsf,
             "Taxa_Sucesso_Pct": (100.0 * nsf / npf) if npf > 0 else 0.0,
             "Stretch_Medio": float(np.mean(stf)) if stf else float("nan"),
             "Stretch_Mediana": float(np.median(stf)) if stf else float("nan")}
            for (to, td), (npf, nsf, stf) in sorted(fluxo.items())
        ] if coletar_fluxo else []
        return met, lf

    @staticmethod
    def _amostrar_pares(pares: List[Tuple[int, int, int]], n_amostra: int, seed: int) -> List[Tuple[int, int, int]]:
        if len(pares) <= n_amostra:
            return pares
        return [pares[int(i)] for i in np.random.default_rng(seed).choice(len(pares), size=n_amostra, replace=False)]

    def _executar_modelos_nulos(self, est_undir: Dict[str, Any]) -> Dict[str, Any]:
        nos = est_undir["nos"]
        x_orig = np.asarray([self.soma_pos_map[no] for no in nos], dtype=float)
        arestas, m = list(self.g_comb_undir.edges()), self.g_comb_undir.number_of_edges()

        sr_esp, dist_esp = [], []
        for i in range(self.n_runs_null):
            si = self.seed_base + i
            xp = np.random.default_rng(si).permutation(x_orig)
            mp = {no: float(xp[idx]) for idx, no in enumerate(nos)}
            if arestas:
                dist_esp.append(float(np.mean([abs(mp[u] - mp[v]) for u, v in arestas])))
            met, _ = self._simular_estrategia(dict(est_undir, x_pos=[float(v) for v in xp]), "Espacial (soma_pos)", self._amostrar_pares(est_undir["pares_alc"], self.n_pares_nulo, si + 1000), seed=si + 2000)
            sr_esp.append(met["Taxa_Sucesso_Pct"])

        sr_rew: Dict[str, List[float]] = {e: [] for e in ESTRATEGIAS_ORDEM}
        r_nulo, knn_acum = [], {}
        for i in range(self.n_runs_null):
            si = self.seed_base + 500 + i
            g_rew = self.g_comb_undir.copy()
            if m >= 4:
                try:
                    nx.double_edge_swap(g_rew, nswap=2 * m, max_tries=20 * m, seed=si)
                except nx.NetworkXAlgorithmError:
                    pass
            rv = _float_finito(nx.degree_assortativity_coefficient(g_rew), default=None)
            if rv is not None:
                r_nulo.append(rv)
            for kv, knnv in nx.average_degree_connectivity(g_rew).items():
                if kv > 0 and knnv is not None and not math.isnan(float(knnv)):
                    knn_acum.setdefault(int(kv), []).append(float(knnv))
            est_r = self._preparar_estruturas_roteamento(g_rew)
            pa = self._amostrar_pares(est_r["pares_alc"], self.n_pares_nulo, si + 3000)
            for idx_e, en in enumerate(ESTRATEGIAS_ORDEM):
                sr_rew[en].append(self._simular_estrategia(est_r, en, pa, seed=si + 4000 + idx_e * 97)[0]["Taxa_Sucesso_Pct"])

        return {"nulo_espacial_sr": sr_esp, "dist_media_nula": float(np.mean(dist_esp)) if dist_esp else 0.0,
                "nulo_rewiring_sr": sr_rew, "r_grau_nulo_lista": r_nulo, "knn_nulo_acum": knn_acum}

    def _correlacao_espacial(self, g: nx.Graph) -> Tuple[float, float]:
        if g.number_of_edges() == 0:
            return 0.0, 0.0
        xs, ys, dists = [], [], []
        for u, v in g.edges():
            if u != v:
                xu, xv = self.soma_pos_map[u], self.soma_pos_map[v]
                dists.append(abs(xu - xv)); xs.append(xu); ys.append(xv)
                if not g.is_directed():
                    xs.append(xv); ys.append(xu)
        if len(xs) < 2:
            return 0.0, float(np.mean(dists)) if dists else 0.0
        dx, dy = np.asarray(xs) - np.mean(xs), np.asarray(ys) - np.mean(ys)
        den = math.sqrt(float(np.sum(dx * dx) * np.sum(dy * dy)))
        r = float(np.sum(dx * dy) / den) if den > 1e-12 else 0.0
        return max(-1.0, min(1.0, _float_finito(r, 0.0) or 0.0)), float(np.mean(dists))

    @staticmethod
    def _assort_attr(g: nx.Graph, attr: str) -> float:
        val = [n for n, d in g.nodes(data=True) if str(d.get(attr, "")).strip() not in {"", "N/A", "nan", "None"}]
        if len(val) < 2 or len({str(g.nodes[n][attr]).strip() for n in val}) < 2 or g.subgraph(val).number_of_edges() == 0:
            return 0.0
        try:
            return max(-1.0, min(1.0, _float_finito(nx.attribute_assortativity_coefficient(g.subgraph(val), attr), 0.0) or 0.0))
        except Exception:
            return 0.0

    @staticmethod
    def _assort_grau(g: nx.Graph, x: str = "out", y: str = "in", weight: Optional[str] = None) -> float:
        if g.number_of_edges() < 2:
            return 0.0
        try:
            r = nx.degree_assortativity_coefficient(g, x=x, y=y, weight=weight) if g.is_directed() else nx.degree_assortativity_coefficient(g, weight=weight)
            return max(-1.0, min(1.0, _float_finito(r, 0.0) or 0.0))
        except Exception:
            return 0.0

    def calcular_assortatividade(self, nulos: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        specs = [
            ("Grau (r_k)", "Combinada (Não-Direcionada)", self.g_comb_undir, {}, "r_grau_combinada"),
            ("Grau Ponderado (r_k^w)", "Combinada (Não-Direcionada)", self.g_comb_undir, {"weight": "weight"}, "r_grau_combinada_ponderada"),
            ("Grau (r_k)", "Eletrica (Não-Direcionada)", self.g_elet_undir, {}, "r_eletrica_grau"),
            ("Grau Ponderado (r_k^w)", "Eletrica (Não-Direcionada)", self.g_elet_undir, {"weight": "weight"}, "r_eletrica_grau_ponderada"),
            ("Grau Direcionado (out -> in)", "Quimica (Direcionada)", self.g_quim_dir, {"x": "out", "y": "in"}, "r_quimica_out_in"),
            ("Grau Direcionado (out -> out)", "Quimica (Direcionada)", self.g_quim_dir, {"x": "out", "y": "out"}, "r_quimica_out_out"),
            ("Grau Direcionado (in -> in)", "Quimica (Direcionada)", self.g_quim_dir, {"x": "in", "y": "in"}, "r_quimica_in_in"),
            ("Grau Direcionado (in -> out)", "Quimica (Direcionada)", self.g_quim_dir, {"x": "in", "y": "out"}, "r_quimica_in_out"),
        ]
        res, linhas = {}, []
        for met, cam, g, kw, ch in specs:
            r = self._assort_grau(g, **kw); res[ch] = r
            st = resumir_contra_nulo(r, nulos["r_grau_nulo_lista"]) if ch == "r_grau_combinada" else None
            if st:
                res.update({"r_grau_nulo_media": st["nulo_media"], "r_grau_nulo_std": st["nulo_std"], "z_score_r_grau": st["z_score"]})
            linhas.append({"Metrica": met, "Camada": cam, "Valor_r": r, "Referencia_Nula": st["nulo_media"] if st else np.nan,
                           "Std_Nula": st["nulo_std"] if st else np.nan, "Z_Score": st["z_score"] if st else np.nan, "Interpretacao": _classificar_assortatividade(r)})

        knn_real = nx.average_degree_connectivity(self.g_comb_undir)
        l_knn, ks, knns = [], [], []
        for kv in sorted(k for k in knn_real if k > 0):
            kr = _float_finito(knn_real[kv], default=None)
            if kr is not None and kr > 0:
                vn = nulos["knn_nulo_acum"].get(int(kv), [])
                l_knn.append({"k": int(kv), "knn_real": kr, "knn_nulo_media": float(np.mean(vn)) if vn else float("nan"), "knn_nulo_std": float(np.std(vn, ddof=1)) if len(vn) > 1 else 0.0})
                ks.append(float(kv)); knns.append(kr)
        sl, ic = np.polyfit(np.log(ks), np.log(knns), 1) if len(ks) >= 2 else (0.0, 0.0)
        res["beta_knn"], res["intercept_knn"] = _float_finito(sl, 0.0) or 0.0, _float_finito(ic, 0.0) or 0.0

        camadas = [("Combinada", self.g_comb_undir), ("Quimica", self.g_quim_dir), ("Eletrica", self.g_elet_undir)]
        for attr, rot in (("tipo_funcional", "Homofilia Funcional (tipo_funcional)"), ("regiao_soma", "Homofilia Anatômica (regiao_soma)"), ("span", "Homofilia de Alcance (span)")):
            for nc, gc in camadas:
                r = self._assort_attr(gc, attr)
                if nc == "Combinada":
                    res[f"r_homofilia_{attr}"] = r
                linhas.append({"Metrica": rot, "Camada": nc, "Valor_r": r, "Referencia_Nula": np.nan, "Std_Nula": np.nan, "Z_Score": np.nan, "Interpretacao": _classificar_assortatividade(r)})
        for nc, gc in camadas:
            re, dm = self._correlacao_espacial(gc)
            if nc == "Combinada":
                res.update({"r_espacial_soma_pos": re, "distancia_fisica_media_aresta": dm, "distancia_fisica_media_nula": float(nulos["dist_media_nula"])})
            linhas.append({"Metrica": "Correlação Espacial (soma_pos)", "Camada": nc, "Valor_r": re, "Referencia_Nula": 0.0 if nc == "Combinada" else np.nan, "Std_Nula": np.nan, "Z_Score": np.nan, "Interpretacao": _classificar_assortatividade(re)})
        return pd.DataFrame(linhas), pd.DataFrame(l_knn), res

    def executar_analise_completa(self) -> Dict[str, Any]:
        est_u, est_d = self._preparar_estruturas_roteamento(self.g_comb_undir), self._preparar_estruturas_roteamento(self.g_comb_dir)
        nulos = self._executar_modelos_nulos(est_u)
        l_nav, l_fluxo = [], []
        for i, en in enumerate(ESTRATEGIAS_ORDEM):
            met, lf = self._simular_estrategia(est_u, en, est_u["pares_alc"], self.seed_base + 100 + i * 31, True)
            l_fluxo.extend(lf)
            nt = "Nulo_Espacial" if en == "Espacial (soma_pos)" else "Nulo_Rewiring"
            st = resumir_contra_nulo(met["Taxa_Sucesso_Pct"], nulos["nulo_espacial_sr"] if en == "Espacial (soma_pos)" else nulos["nulo_rewiring_sr"][en])
            l_nav.append({"Visao": "Combinada (Não-Direcionada)", "Estrategia": en, **met, "Nulo_Tipo": nt, "Nulo_Sucesso_Media": st["nulo_media"], "Nulo_Sucesso_Std": st["nulo_std"], "Z_Score_Sucesso": st["z_score"]})
        for i, en in enumerate(ESTRATEGIAS_ORDEM):
            met, _ = self._simular_estrategia(est_d, en, est_d["pares_alc"], self.seed_base + 200 + i * 31, False)
            l_nav.append({"Visao": "Combinada (Direcionada)", "Estrategia": en, **met, "Nulo_Tipo": "N/A", "Nulo_Sucesso_Media": np.nan, "Nulo_Sucesso_Std": np.nan, "Z_Score_Sucesso": np.nan})

        cols = ["Visao", "Estrategia", "N_Pares_Alcancaveis", "Fracao_Alcancavel", "Taxa_Sucesso_Pct", "Stretch_Medio", "Stretch_Mediana", "Stretch_P90", "Frac_Otimo_Pct", "Hops_Greedy_Medio", "Hops_SP_Medio", "Eficiencia_Nav", "Eficiencia_Global", "Razao_Eficiencia", "Pct_Rotas_Via_RichClub", "Nulo_Tipo", "Nulo_Sucesso_Media", "Nulo_Sucesso_Std", "Z_Score_Sucesso"]
        df_nav, df_fluxo = pd.DataFrame(l_nav)[cols], pd.DataFrame(l_fluxo)
        df_assort, df_knn, r_assort = self.calcular_assortatividade(nulos)

        df_u = df_nav[df_nav["Visao"] == "Combinada (Não-Direcionada)"]
        l_esp, l_gr, l_hib = (df_u[df_u["Estrategia"] == e].iloc[0] for e in ESTRATEGIAS_ORDEM[:3])
        df_gul = df_u[df_u["Estrategia"] != "Caminhada Aleatória"]

        def _sm(e: str) -> Tuple[float, float]:
            sub = df_fluxo[(df_fluxo["Estrategia"] == e) & (df_fluxo["Tipo_Origem"] == "Sensorial") & (df_fluxo["Tipo_Destino"] == "Motor")]
            return (0.0, 1.0) if sub.empty else (_float_finito(sub.iloc[0]["Taxa_Sucesso_Pct"], 0.0) or 0.0, _float_finito(sub.iloc[0]["Stretch_Medio"], 1.0) or 1.0)

        sme, smg, smh = _sm("Espacial (soma_pos)"), _sm("Grau (Hubs)"), _sm("Híbrido (Espacial + Grau)")
        resumo = sanitizar_json({
            "metadados": {"N": self.g_comb_undir.number_of_nodes(), "M_combinada": self.g_comb_undir.number_of_edges(), "n_soma_pos_ausentes": self.n_soma_pos_ausentes, "membros_rich_club": sorted(self.membros_rich_club)},
            "navegacao": {
                "melhor_estrategia_undir": str(df_gul.loc[df_gul["Taxa_Sucesso_Pct"].idxmax(), "Estrategia"]),
                "taxa_sucesso_espacial_pct": float(l_esp["Taxa_Sucesso_Pct"]), "stretch_medio_espacial": float(l_esp["Stretch_Medio"]),
                "taxa_sucesso_grau_pct": float(l_gr["Taxa_Sucesso_Pct"]), "stretch_medio_grau": float(l_gr["Stretch_Medio"]),
                "taxa_sucesso_hibrido_pct": float(l_hib["Taxa_Sucesso_Pct"]), "stretch_medio_hibrido": float(l_hib["Stretch_Medio"]),
                "sensorial_para_motor": {"espacial_sucesso_pct": sme[0], "espacial_stretch_medio": sme[1], "grau_sucesso_pct": smg[0], "grau_stretch_medio": smg[1], "hibrido_sucesso_pct": smh[0], "hibrido_stretch_medio": smh[1]},
            },
            "assortatividade": r_assort,
        }, nan_como_none=True)
        return {"df_navegacao": df_nav, "df_nav_fluxo": df_fluxo, "df_assortatividade": df_assort, "df_knn": df_knn, "resumo_json": resumo}
