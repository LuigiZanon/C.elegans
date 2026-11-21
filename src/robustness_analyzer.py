"""Módulo de Robustez a Falhas e Ataques (Percolação, Schneider R, ER/WS/BA e Fluxo Direcionado)."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from graph_utils import calibrar_parametros_sinteticos, gerar_rede_sintetica, projetar_grafo, sanitizar_json


class RobustnessAnalyzer:
    """Avalia robustez estrutural, comparativa (ER/WS/BA) e funcional direcionada."""

    def __init__(
        self, graph: nx.MultiDiGraph, membros_rich_club: Optional[Set[str]] = None,
        n_runs_aleatorio: int = 30, n_runs_sinteticas: int = 15, ws_rewiring_p: float = 0.10,
        passo_eficiencia: int = 5, limiar_colapso_50: float = 0.50, limiar_colapso_20: float = 0.20, seed_base: int = 42,
    ) -> None:
        self.graph = graph
        self.n_runs_aleatorio, self.n_runs_sinteticas = int(n_runs_aleatorio), int(n_runs_sinteticas)
        self.ws_rewiring_p, self.passo_eficiencia = float(ws_rewiring_p), max(1, int(passo_eficiencia))
        self.limiar_colapso_50, self.limiar_colapso_20, self.seed_base = float(limiar_colapso_50), float(limiar_colapso_20), int(seed_base)

        self.g_undir = projetar_grafo(graph, "Combinada", direcionado=False)
        self.g_dir_comb = projetar_grafo(graph, "Combinada", direcionado=True)
        self.g_dir_quim = projetar_grafo(graph, "Quimica", direcionado=True)
        self.nos = sorted(self.g_undir.nodes())
        self.n = len(self.nos)
        self.nos_sensoriais = {n for n, d in self.g_undir.nodes(data=True) if d.get("tipo_funcional") == "Sensorial"}
        self.nos_motores = {n for n, d in self.g_undir.nodes(data=True) if d.get("tipo_funcional") == "Motor"}
        rc = [n for n in (membros_rich_club or set()) if n in self.g_undir] or sorted(self.nos, key=lambda n: (-self.g_undir.degree(n), n))[:6]
        self.membros_rich_club = sorted(rc, key=lambda n: (-self.g_undir.degree(n), n))

    def _grade_q(self, n: int) -> List[int]:
        return sorted(set(range(min(11, n + 1))) | set(range(0, n + 1, self.passo_eficiencia)) | {n})

    @staticmethod
    def _gcc_por_remocao_reversa(g: nx.Graph, ordem: Sequence[Any]) -> np.ndarray:
        n = len(ordem)
        s = np.zeros(n + 1, dtype=float)
        if n == 0:
            return s
        parent: Dict[Any, Any] = {}
        size: Dict[Any, int] = {}
        mc = 0

        def find(x: Any) -> Any:
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        for idx in range(n - 1, -1, -1):
            u = ordem[idx]; parent[u], size[u], mc = u, 1, max(mc, 1)
            for v in g.neighbors(u):
                if v in parent:
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        if size[ru] < size[rv]:
                            ru, rv = rv, ru
                        parent[rv] = ru; size[ru] += size[rv]; mc = max(mc, size[ru])
            s[idx] = mc / n
        return s

    @staticmethod
    def _eficiencia_global(g: nx.Graph, n0: int) -> float:
        if n0 <= 1 or g.number_of_nodes() <= 1 or g.number_of_edges() == 0:
            return 0.0
        return sum(sum(1.0 / d for d in ds.values() if d > 0) for _, ds in nx.all_pairs_shortest_path_length(g)) / (n0 * (n0 - 1))

    @staticmethod
    def _trapz(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x))) if len(x) >= 2 else 0.0

    @staticmethod
    def _limiar_fc(f: np.ndarray, v: np.ndarray, lim: float) -> float:
        idx = np.where(v <= lim)[0]
        return float(f[idx[0]]) if len(idx) > 0 else 1.0

    @staticmethod
    def _ordem_adaptativa_grau(g: nx.Graph, prefixo: Sequence[Any] = ()) -> List[Any]:
        gw, ord_l = g.copy(), []
        for u in prefixo:
            if u in gw:
                ord_l.append(u); gw.remove_node(u)
        while gw.number_of_nodes() > 0:
            if gw.number_of_edges() == 0:
                ord_l.extend(sorted(gw.nodes())); break
            alvo = min(gw.nodes(), key=lambda x: (-gw.degree(x), x))
            ord_l.append(alvo); gw.remove_node(alvo)
        return ord_l

    @staticmethod
    def _ordem_adaptativa_intermediacao(g: nx.Graph) -> List[Any]:
        gw, ord_l = g.copy(), []
        while gw.number_of_nodes() > 0:
            if gw.number_of_edges() == 0:
                ord_l.extend(sorted(gw.nodes())); break
            bc = nx.betweenness_centrality(gw, normalized=True)
            alvo = min(gw.nodes(), key=lambda x: (-bc[x], -gw.degree(x), x))
            ord_l.append(alvo); gw.remove_node(alvo)
        return ord_l

    def _avaliar_eficiencia_grade(self, g: nx.Graph, ordem: Sequence[Any], qg: Sequence[int], e0: float) -> np.ndarray:
        en = np.zeros(len(qg), dtype=float)
        if e0 <= 0:
            return en
        gw, rem, n = g.copy(), 0, len(ordem)
        for i, q in enumerate(qg):
            if q > rem:
                gw.remove_nodes_from(ordem[rem:q]); rem = q
            en[i] = 1.0 if q == 0 else (self._eficiencia_global(gw, n) / e0)
        return en

    def _simular_pilar1_real(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        g, n = self.g_undir, self.n
        f_all, qg = np.arange(n + 1) / n, self._grade_q(n)
        fg, e0, q_rc = np.asarray(qg, dtype=float) / n, self._eficiencia_global(g, n), min(len(self.membros_rich_club), n)
        bc_e = nx.betweenness_centrality(g, normalized=True)
        ordens = {
            "Ataque Estático (Grau)": sorted(self.nos, key=lambda x: (-g.degree(x), x)),
            "Ataque Adaptativo (Grau)": self._ordem_adaptativa_grau(g),
            "Ataque Estático (Intermediação)": sorted(self.nos, key=lambda x: (-bc_e[x], -g.degree(x), x)),
            "Ataque Adaptativo (Intermediação)": self._ordem_adaptativa_intermediacao(g),
            "Rich-Club -> Grau Adaptativo": self._ordem_adaptativa_grau(g, prefixo=self.membros_rich_club),
        }
        lp, ls, cache = [], [], {}

        sr, er = np.zeros((self.n_runs_aleatorio, n + 1)), np.zeros((self.n_runs_aleatorio, len(qg)))
        rr, rer, f50, f20, fe50 = [], [], [], [], []
        for r in range(self.n_runs_aleatorio):
            ordem = [self.nos[i] for i in np.random.default_rng(self.seed_base + r).permutation(n)]
            sc, ec = self._gcc_por_remocao_reversa(g, ordem), self._avaliar_eficiencia_grade(g, ordem, qg, e0)
            sr[r], er[r] = sc, ec
            rr.append(float(np.sum(sc[1:]) / n)); rer.append(self._trapz(fg, ec))
            f50.append(self._limiar_fc(f_all, sc, self.limiar_colapso_50)); f20.append(self._limiar_fc(f_all, sc, self.limiar_colapso_20)); fe50.append(self._limiar_fc(fg, ec, 0.50))

        sm, ss = sr.mean(0), sr.std(0)
        emf, esf = np.interp(f_all, fg, er.mean(0)), np.interp(f_all, fg, er.std(0))
        for q in range(n + 1):
            lp.append({"Regime": "Falha Aleatória", "Nos_Removidos": q, "Fracao_Removida": float(f_all[q]), "GCC_Media": float(sm[q]), "GCC_Std": float(ss[q]), "Eficiencia_Norm_Media": float(emf[q]), "Eficiencia_Norm_Std": float(esf[q])})
        ls.append({"Regime": "Falha Aleatória", "Schneider_R": float(np.mean(rr)), "Schneider_R_Std": float(np.std(rr)), "Schneider_RE": float(np.mean(rer)), "fc_50": float(np.mean(f50)), "fc_20": float(np.mean(f20)), "fc_E50": float(np.mean(fe50)), "GCC_apos_RichClub_6": float(sm[q_rc])})
        cache["Falha Aleatória"] = {"s_mean": sm, "s_std": ss, "r_mean": float(np.mean(rr)), "r_std": float(np.std(rr)), "fc_50": float(np.mean(f50)), "fc_20": float(np.mean(f20))}

        for reg, ordem in ordens.items():
            sc, eg = self._gcc_por_remocao_reversa(g, ordem), self._avaliar_eficiencia_grade(g, ordem, qg, e0)
            ef = np.interp(f_all, fg, eg)
            rv, rev = float(np.sum(sc[1:]) / n), self._trapz(fg, eg)
            c50, c20, ce50 = self._limiar_fc(f_all, sc, self.limiar_colapso_50), self._limiar_fc(f_all, sc, self.limiar_colapso_20), self._limiar_fc(fg, eg, 0.50)
            for q in range(n + 1):
                lp.append({"Regime": reg, "Nos_Removidos": q, "Fracao_Removida": float(f_all[q]), "GCC_Media": float(sc[q]), "GCC_Std": 0.0, "Eficiencia_Norm_Media": float(ef[q]), "Eficiencia_Norm_Std": 0.0})
            ls.append({"Regime": reg, "Schneider_R": rv, "Schneider_R_Std": 0.0, "Schneider_RE": rev, "fc_50": c50, "fc_20": c20, "fc_E50": ce50, "GCC_apos_RichClub_6": float(sc[q_rc])})
            if reg == "Ataque Adaptativo (Grau)":
                cache[reg] = {"s_mean": sc, "s_std": np.zeros_like(sc), "r_mean": rv, "r_std": 0.0, "fc_50": c50, "fc_20": c20}
        return pd.DataFrame(lp), pd.DataFrame(ls), cache

    def _simular_pilar2_sinteticas(self, cache_real: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        p = calibrar_parametros_sinteticos(self.g_undir)
        n, m, k_ws, m_ba = int(p["N"]), int(p["M"]), int(p["k_ws"]), int(p["m_ba"])
        f_all, lc, lr = np.arange(n + 1) / n, [], []

        def _reg(mod: str, al: Tuple[Any, ...], at: Tuple[Any, ...]) -> None:
            for reg, sm, ss in (("Falha Aleatória", al[0], al[1]), ("Ataque Adaptativo (Grau)", at[0], at[1])):
                for q in range(n + 1):
                    lc.append({"Modelo": mod, "Regime": reg, "Nos_Removidos": q, "Fracao_Removida": float(f_all[q]), "GCC_Media": float(sm[q]), "GCC_Std": float(ss[q])})
            lr.append({"Modelo": mod, "R_Aleatorio_Media": al[2], "R_Aleatorio_Std": al[3], "R_Ataque_Adapt_Media": at[2], "R_Ataque_Adapt_Std": at[3], "Delta_R": al[2] - at[2], "Razao_RYF": al[2] / max(at[2], 1e-9), "fc_50_Aleatorio": al[4], "fc_50_Ataque_Adapt": at[4], "fc_20_Ataque_Adapt": at[5]})

        ar, at = cache_real["Falha Aleatória"], cache_real["Ataque Adaptativo (Grau)"]
        _reg("C. elegans (Real)", (ar["s_mean"], ar["s_std"], ar["r_mean"], ar["r_std"], ar["fc_50"]), (at["s_mean"], at["s_std"], at["r_mean"], at["r_std"], at["fc_50"], at["fc_20"]))
        for im, (nm, cod) in enumerate((("Erdős-Rényi (ER)", "ER"), ("Watts-Strogatz (WS)", "WS"), ("Barabási-Albert (BA)", "BA"))):
            sal, sat = np.zeros((self.n_runs_sinteticas, n + 1)), np.zeros((self.n_runs_sinteticas, n + 1))
            ral, rat, f50a, f50t, f20t = [], [], [], [], []
            for r in range(self.n_runs_sinteticas):
                seed = self.seed_base + 100 * (im + 1) + r
                gs = gerar_rede_sintetica(cod, n, m, k_ws, m_ba, self.ws_rewiring_p, seed)
                cal = self._gcc_por_remocao_reversa(gs, np.random.default_rng(seed + 1000).permutation(n).tolist())
                cat = self._gcc_por_remocao_reversa(gs, self._ordem_adaptativa_grau(gs))
                sal[r], sat[r] = cal, cat
                ral.append(float(np.sum(cal[1:]) / n)); rat.append(float(np.sum(cat[1:]) / n))
                f50a.append(self._limiar_fc(f_all, cal, self.limiar_colapso_50)); f50t.append(self._limiar_fc(f_all, cat, self.limiar_colapso_50)); f20t.append(self._limiar_fc(f_all, cat, self.limiar_colapso_20))
            _reg(nm, (sal.mean(0), sal.std(0), float(np.mean(ral)), float(np.std(ral)), float(np.mean(f50a))), (sat.mean(0), sat.std(0), float(np.mean(rat)), float(np.std(rat)), float(np.mean(f50t)), float(np.mean(f20t))))
        return pd.DataFrame(lc), pd.DataFrame(lr)

    def _metricas_direcionadas(self, gd: nx.DiGraph) -> Tuple[float, float, float]:
        if gd.number_of_nodes() == 0:
            return 0.0, 0.0, 0.0
        lscc = max((len(c) for c in nx.strongly_connected_components(gd)), default=0) / self.n
        den, ma = len(self.nos_sensoriais) * len(self.nos_motores), self.nos_motores.intersection(gd.nodes())
        if den == 0 or not ma:
            return lscc, 0.0, 0.0
        alc, eff = 0, 0.0
        for s in self.nos_sensoriais.intersection(gd.nodes()):
            ds = nx.single_source_shortest_path_length(gd, s)
            for m in ma:
                d = ds.get(m)
                if d is not None and d > 0:
                    alc += 1; eff += 1.0 / d
        return lscc, (alc / den) * 100.0, eff / den

    def _avaliar_curva_dir(self, gd: nx.DiGraph, ordem: Sequence[str], qg: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lv, av, ev = (np.zeros(len(qg)) for _ in range(3))
        gw, rem = gd.copy(), 0
        for i, q in enumerate(qg):
            if q > rem:
                gw.remove_nodes_from(ordem[rem:q]); rem = q
            lv[i], av[i], ev[i] = self._metricas_direcionadas(gw)
        return lv, av, ev

    def _ablacao_rc(self, gd: nx.DiGraph, soff: int) -> Dict[str, float]:
        krc = len(self.membros_rich_club)
        l0, a0, e0, g0 = (*self._metricas_direcionadas(gd), self._eficiencia_global(gd, self.n))
        grc = gd.copy(); grc.remove_nodes_from(self.membros_rich_club)
        lr, ar, er, gr = (*self._metricas_direcionadas(grc), self._eficiencia_global(grc, self.n))
        la, aa, ea, ga = [], [], [], []
        rng = np.random.default_rng(self.seed_base + soff)
        for _ in range(self.n_runs_aleatorio):
            gt = gd.copy(); gt.remove_nodes_from([self.nos[i] for i in rng.choice(self.n, size=krc, replace=False)])
            l_, a_, e_ = self._metricas_direcionadas(gt)
            la.append(l_); aa.append(a_); ea.append(e_); ga.append(self._eficiencia_global(gt, self.n))

        def _q(b: float, v: float) -> float:
            return ((b - v) / b) * 100.0 if b > 0 else 0.0

        def _z(o: float, arr: List[float]) -> float:
            mu, sd = float(np.mean(arr)), float(np.std(arr))
            return (o - mu) / sd if sd > 1e-12 else 0.0

        qlr, qla = _q(l0, lr), [_q(l0, v) for v in la]
        qar, qaa = _q(a0, ar), [_q(a0, v) for v in aa]
        qer, qea = _q(e0, er), [_q(e0, v) for v in ea]
        qgr, qga = _q(g0, gr), [_q(g0, v) for v in ga]
        return {
            "lscc_inicial_pct": l0 * 100, "lscc_apos_rc_pct": lr * 100, "lscc_apos_6_aleatorios_pct": float(np.mean(la)) * 100,
            "lscc_queda_rc_pct": qlr, "lscc_queda_aleat_pct": float(np.mean(qla)), "z_score_queda_lscc": _z(qlr, qla),
            "alcance_sm_inicial_pct": a0, "alcance_sm_apos_rc_pct": ar, "alcance_sm_apos_6_aleatorios_pct": float(np.mean(aa)),
            "alcance_sm_queda_rc_pct": qar, "alcance_sm_queda_aleat_pct": float(np.mean(qaa)), "z_score_queda_alcance_sm": _z(qar, qaa),
            "eficiencia_sm_inicial": e0, "eficiencia_sm_apos_rc": er, "eficiencia_sm_queda_rc_pct": qer, "eficiencia_sm_queda_aleat_pct": float(np.mean(qea)), "z_score_queda_eficiencia_sm": _z(qer, qea),
            "eficiencia_glob_queda_rc_pct": qgr, "eficiencia_glob_queda_aleat_pct": float(np.mean(qga)), "z_score_queda_eficiencia_glob": _z(qgr, qga),
        }

    def simular_robustez_funcional_direcionada(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        qg, linhas = self._grade_q(self.n), []
        for reg, ordem in (("Rich-Club -> Grau Adaptativo (Dir.)", self._ordem_adaptativa_grau(self.g_dir_comb, prefixo=self.membros_rich_club)), ("Ataque Adaptativo (Intermediação Dir.)", self._ordem_adaptativa_intermediacao(self.g_dir_comb))):
            lv, av, ev = self._avaliar_curva_dir(self.g_dir_comb, ordem, qg)
            for i, q in enumerate(qg):
                linhas.append({"Regime": reg, "Nos_Removidos": q, "Fracao_Removida": float(q / self.n), "Frac_LSCC": float(lv[i]), "Frac_LSCC_Std": 0.0, "Alcance_S_para_M_Pct": float(av[i]), "Alcance_S_para_M_Std": 0.0, "Eficiencia_S_para_M": float(ev[i]), "Eficiencia_S_para_M_Std": 0.0})
        lr, ar, er = (np.zeros((self.n_runs_aleatorio, len(qg))) for _ in range(3))
        for r in range(self.n_runs_aleatorio):
            lr[r], ar[r], er[r] = self._avaliar_curva_dir(self.g_dir_comb, [self.nos[i] for i in np.random.default_rng(self.seed_base + 500 + r).permutation(self.n)], qg)
        for i, q in enumerate(qg):
            linhas.append({"Regime": "Falha Aleatória (Dir.)", "Nos_Removidos": q, "Fracao_Removida": float(q / self.n), "Frac_LSCC": float(lr[:, i].mean()), "Frac_LSCC_Std": float(lr[:, i].std()), "Alcance_S_para_M_Pct": float(ar[:, i].mean()), "Alcance_S_para_M_Std": float(ar[:, i].std()), "Eficiencia_S_para_M": float(er[:, i].mean()), "Eficiencia_S_para_M_Std": float(er[:, i].std())})
        return pd.DataFrame(linhas), {"combinada_dir": self._ablacao_rc(self.g_dir_comb, 777), "quimica_dir": self._ablacao_rc(self.g_dir_quim, 888)}

    def executar_analise_completa(self) -> Dict[str, Any]:
        df_p, df_s, cache = self._simular_pilar1_real()
        df_sc, df_sr = self._simular_pilar2_sinteticas(cache)
        df_fd, ab = self.simular_robustez_funcional_direcionada()
        resumo = sanitizar_json({
            "metadados": {"N": self.n, "M_undir": self.g_undir.number_of_edges(), "n_rich_club": len(self.membros_rich_club), "membros_rich_club": list(self.membros_rich_club)},
            "limiares_fc": {r["Regime"]: {"fc_50": float(r["fc_50"]), "fc_20": float(r["fc_20"]), "fc_E50": float(r["fc_E50"])} for _, r in df_s.iterrows()},
            "schneider_R": {r["Regime"]: {"R": float(r["Schneider_R"]), "R_std": float(r["Schneider_R_Std"]), "R_E": float(r["Schneider_RE"])} for _, r in df_s.iterrows()},
            "comparacao_sinteticas": {r["Modelo"]: {"R_aleat": float(r["R_Aleatorio_Media"]), "R_ataque": float(r["R_Ataque_Adapt_Media"]), "razao_ryf": float(r["Razao_RYF"])} for _, r in df_sr.iterrows()},
            "ablacao_rich_club": ab,
        })
        return {"df_percolacao": df_p, "df_schneider": df_s, "df_sinteticas_curvas": df_sc, "df_sinteticas_resumo": df_sr, "df_funcional_dir": df_fd, "resumo_json": resumo}
