"""Classificação topológica formal (Aleatório, Small-World e Scale-Free) em NumPy/NetworkX."""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
from graph_utils import (
    calibrar_parametros_sinteticos, extrair_componente_principal,
    gerar_rede_sintetica, projetar_grafo, sanitizar_json,
)


class TopologyClassifier:
    """Executa testes estatísticos formais para diagnosticar a classe topológica do conectoma."""

    def __init__(
        self, graph: nx.MultiDiGraph, n_runs_null: int = 30, n_runs_ws_sweep: int = 15,
        n_bootstrap_ks: int = 200, min_tail_size: int = 30, seed_base: int = 42,
        baselines_externos: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.graph = graph
        self.n_runs_null, self.n_runs_ws_sweep = int(n_runs_null), int(n_runs_ws_sweep)
        self.n_bootstrap_ks, self.min_tail_size, self.seed_base = int(n_bootstrap_ks), int(min_tail_size), int(seed_base)
        self._baselines_externos = baselines_externos

        self.g_simple: nx.Graph = projetar_grafo(graph, "Combinada", direcionado=False, preservar_atributos_nos=False)
        p = calibrar_parametros_sinteticos(self.g_simple)
        self.n, self.m, self.grau_medio, self.densidade, self.k_ws = int(p["N"]), int(p["M"]), float(p["k_medio"]), float(p["densidade"]), int(p["k_ws"])
        self.g_quimico_dir: nx.DiGraph = projetar_grafo(graph, "Quimica", direcionado=True, preservar_atributos_nos=False)

        self.graus_real = np.array([deg for _, deg in self.g_simple.degree()], dtype=float)
        self.c_real = float(nx.average_clustering(self.g_simple)) if self.n > 0 else 0.0
        gcc = extrair_componente_principal(self.g_simple)
        self.l_real = float(nx.average_shortest_path_length(gcc)) if gcc.number_of_nodes() > 1 else 0.0
        self.kappa_real = float(np.mean(self.graus_real ** 2) / (self.grau_medio ** 2)) if self.grau_medio > 0 else 0.0
        self._cache_nulo_er: Optional[Dict[str, Any]] = None

    @staticmethod
    def _ks_duas_amostras(a: np.ndarray, b: np.ndarray) -> float:
        a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
        if len(a) == 0 or len(b) == 0:
            return 0.0
        u = np.unique(np.concatenate((a, b)))
        return float(np.max(np.abs(np.searchsorted(a, u, side="right") / float(len(a)) - np.searchsorted(b, u, side="right") / float(len(b)))))

    def avaliar_hipotese_aleatoria(self) -> Dict[str, Any]:
        ext = self._baselines_externos
        if ext and ext.get("n_runs") == self.n_runs_null and ext.get("seed_base") == self.seed_base:
            c_arr, l_arr, k_arr, g_arr = (np.asarray(ext[k], dtype=float) for k in ("c_er_arr", "l_er_arr", "kappa_er_arr", "graus_er_arr"))
        else:
            cl, ll, kl, gt = [], [], [], []
            for i in range(self.n_runs_null):
                g_er = gerar_rede_sintetica("ER", self.n, self.m, seed=self.seed_base + i)
                cl.append(float(nx.average_clustering(g_er)))
                gi_c = extrair_componente_principal(g_er)
                ll.append(float(nx.average_shortest_path_length(gi_c)) if gi_c.number_of_nodes() > 1 else 0.0)
                gi = np.array([deg for _, deg in g_er.degree()], dtype=float)
                km = float(np.mean(gi))
                kl.append(float(np.mean(gi ** 2) / (km ** 2)) if km > 0 else 0.0); gt.extend(gi.tolist())
            c_arr, l_arr, k_arr, g_arr = np.asarray(cl), np.asarray(ll), np.asarray(kl), np.asarray(gt)

        cm, cs = float(np.mean(c_arr)), float(np.std(c_arr, ddof=1)) if len(c_arr) > 1 else 1e-9
        lm, ls = float(np.mean(l_arr)), float(np.std(l_arr, ddof=1)) if len(l_arr) > 1 else 1e-9
        km = float(np.mean(k_arr))
        z_c = float((self.c_real - cm) / cs) if cs > 1e-12 else 0.0
        id_p = float(np.var(self.graus_real, ddof=1) / self.grau_medio) if (self.n > 1 and self.grau_medio > 0) else 0.0
        ks_er = self._ks_duas_amostras(self.graus_real, g_arr)
        self._cache_nulo_er = {"c_er_arr": c_arr, "l_er_arr": l_arr, "c_er_media": cm, "c_er_std": cs, "l_er_media": lm, "l_er_std": ls, "kappa_er_media": km}
        return {
            "C_real": self.c_real, "C_ER_media": cm, "C_ER_std": cs, "z_score_C": z_c, "indice_dispersao_ID": id_p,
            "kappa_real": self.kappa_real, "kappa_ER_media": km, "razao_heterogeneidade": float(self.kappa_real / km) if km > 0 else 0.0,
            "ks_dist_vs_ER": ks_er, "veredito": "Rejeitado" if ((z_c > 3.0) or (id_p > 2.0) or (ks_er > 0.15)) else "Não Rejeitado",
        }

    def avaliar_hipotese_small_world(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        if self._cache_nulo_er is None:
            self.avaliar_hipotese_aleatoria()
        cache = self._cache_nulo_er; assert cache is not None
        c_arr, l_arr, c_er, l_er = cache["c_er_arr"], cache["l_er_arr"], cache["c_er_media"], cache["l_er_media"]
        if self._baselines_externos and "c_latt" in self._baselines_externos:
            c_latt, l_latt = float(self._baselines_externos["c_latt"]), float(self._baselines_externos["l_latt"])
        else:
            gl = gerar_rede_sintetica("LATTICE", self.n, self.m, self.k_ws, seed=self.seed_base)
            c_latt = float(nx.average_clustering(gl))
            gccl = extrair_componente_principal(gl)
            l_latt = float(nx.average_shortest_path_length(gccl)) if gccl.number_of_nodes() > 1 else 0.0

        sigma = float((self.c_real / c_er) / (self.l_real / l_er)) if (c_er > 0 and l_er > 0 and self.l_real > 0) else 0.0
        sigs = np.where((c_arr > 0) & (self.l_real > 0), (self.c_real / c_arr) / (self.l_real / l_arr), 0.0)
        sig_std = float(np.std(sigs, ddof=1)) if len(sigs) > 1 else 0.0
        omega = float((l_er / self.l_real if self.l_real > 0 else 0.0) - (self.c_real / c_latt if c_latt > 0 else 0.0))
        dc = float(np.clip((c_latt - self.c_real) / (c_latt - c_er), 0, 1)) if abs(c_latt - c_er) > 1e-12 else 0.0
        dl = float(np.clip((self.l_real - l_er) / (l_latt - l_er), 0, 1)) if abs(l_latt - l_er) > 1e-12 else 0.0
        phi = float(np.clip(1.0 - math.sqrt((dc ** 2 + dl ** 2) / 2.0), 0, 1))
        crn, lrn = (self.c_real / c_latt if c_latt > 0 else 0.0), (self.l_real / l_latt if l_latt > 0 else 0.0)

        linhas, mdist, peq = [], float("inf"), 0.10
        for idx_p, pv in enumerate(np.concatenate(([0.0], np.logspace(-3, 0, 12)))):
            pf = float(pv)
            if pf == 0.0:
                cm, cs, lm, ls = 1.0, 0.0, 1.0, 0.0
            else:
                cn, ln = [], []
                for r in range(self.n_runs_ws_sweep):
                    gw = gerar_rede_sintetica("WS", self.n, self.m, self.k_ws, ws_p=pf, seed=self.seed_base + idx_p * 100 + r)
                    cn.append(float(nx.average_clustering(gw)) / c_latt if c_latt > 0 else 0.0)
                    gccw = extrair_componente_principal(gw)
                    ln.append((float(nx.average_shortest_path_length(gccw)) if gccw.number_of_nodes() > 1 else 0.0) / l_latt if l_latt > 0 else 0.0)
                cm, cs = float(np.mean(cn)), float(np.std(cn, ddof=1)) if len(cn) > 1 else 0.0
                lm, ls = float(np.mean(ln)), float(np.std(ln, ddof=1)) if len(ln) > 1 else 0.0
                d = math.sqrt((cm - crn) ** 2 + (lm - lrn) ** 2)
                if d < mdist:
                    mdist, peq = d, pf
            linhas.append({"p_rewiring": pf, "C_norm_media": cm, "C_norm_std": cs, "L_norm_media": lm, "L_norm_std": ls})

        ver = "Confirmado" if ((sigma > 1.0) and (-0.5 <= omega <= 0.5) and (phi >= 0.595)) else ("Parcial" if sigma > 1.0 else "Rejeitado")
        return {
            "C_real": self.c_real, "C_ER_media": c_er, "C_latt": c_latt, "L_real": self.l_real, "L_ER_media": l_er, "L_latt": l_latt,
            "C_real_norm_latt": crn, "L_real_norm_latt": lrn, "sigma": sigma, "sigma_std": sig_std, "omega": omega,
            "phi_swp": phi, "delta_C": dc, "delta_L": dl, "p_equiv_ws": peq,
            "z_score_C_vs_ER": float((self.c_real - c_er) / cache["c_er_std"]) if cache["c_er_std"] > 1e-12 else 0.0,
            "indice_dispersao_ID": float(np.var(self.graus_real, ddof=1) / self.grau_medio) if self.grau_medio > 0 else 0.0,
            "veredito": ver,
        }, pd.DataFrame(linhas)

    @staticmethod
    def _log_pmf_normalizada(k_grid: np.ndarray, familia: str, params: Tuple[float, ...]) -> np.ndarray:
        if familia == "PL":
            lu = -params[0] * np.log(k_grid)
        elif familia == "TPL":
            lu = -params[0] * np.log(k_grid) - params[1] * k_grid
        elif familia == "EXP":
            lu = -params[0] * k_grid
        elif familia == "LN":
            lk = np.log(k_grid)
            lu = -lk - ((lk - params[0]) ** 2) / (2.0 * max(params[1], 1e-4) ** 2)
        else:
            raise ValueError(f"Família desconhecida: {familia}")
        mx = float(np.max(lu))
        return lu - (mx + float(np.log(np.sum(np.exp(lu - mx)))))

    @classmethod
    def _ajustar_familia_cauda(cls, cauda: np.ndarray, k_min: int, k_grid: np.ndarray, familia: str) -> Tuple[Tuple[float, ...], float, np.ndarray]:
        idx = cauda - k_min

        def _b1(grade: np.ndarray, mp: float, mll: float) -> Tuple[float, float]:
            for v in grade:
                ll = float(np.sum(cls._log_pmf_normalizada(k_grid, familia, (float(v),))[idx]))
                if ll > mll:
                    mll, mp = ll, float(v)
            return mp, mll

        def _b2(g1: np.ndarray, g2: np.ndarray, p1: float, p2: float, mll: float) -> Tuple[float, float, float]:
            for a in g1:
                for b in g2:
                    ll = float(np.sum(cls._log_pmf_normalizada(k_grid, familia, (float(a), float(b)))[idx]))
                    if ll > mll:
                        mll, p1, p2 = ll, float(a), float(b)
            return p1, p2, mll

        if familia == "PL":
            sl = float(np.sum(np.log(cauda / max(0.5, float(k_min) - 0.5))))
            g0 = float(np.clip(1.0 + len(cauda) / sl if sl > 0 else 2.5, 1.15, 5.40))
            mg, mll = _b1(np.unique(np.append(np.linspace(1.10, 5.50, 40), g0)), g0, -float("inf"))
            mg, mll = _b1(np.linspace(max(1.05, mg - 0.25), min(6.50, mg + 0.25), 40), mg, mll)
            params: Tuple[float, ...] = (mg,)
        elif familia == "EXP":
            l0 = float(np.clip(1.0 / max(0.5, float(np.mean(cauda) - k_min + 0.5)), 0.002, 0.50))
            ml, mll = _b1(np.unique(np.append(np.linspace(0.002, 0.50, 40), l0)), l0, -float("inf"))
            ml, mll = _b1(np.linspace(max(0.0005, ml - 0.025), min(0.80, ml + 0.025), 40), ml, mll)
            params = (ml,)
        elif familia == "TPL":
            mg, ml, mll = _b2(np.linspace(0.10, 4.00, 22), np.linspace(0.002, 0.25, 22), 1.0, 0.05, -float("inf"))
            mg, ml, mll = _b2(np.linspace(max(0.01, mg - 0.30), min(5.00, mg + 0.30), 20), np.linspace(max(0.0005, ml - 0.025), min(0.45, ml + 0.025), 20), mg, ml, mll)
            params = (mg, ml)
        else:
            lx = np.log(cauda.astype(float))
            mu0, s0 = float(np.mean(lx)), max(0.15, float(np.std(lx)))
            mm, ms, mll = _b2(np.linspace(mu0 - 1.5, mu0 + 1.5, 22), np.linspace(max(0.10, s0 - 0.8), s0 + 1.5, 22), mu0, s0, -float("inf"))
            mm, ms, mll = _b2(np.linspace(mm - 0.25, mm + 0.25, 20), np.linspace(max(0.05, ms - 0.20), ms + 0.20, 20), mm, ms, mll)
            params = (mm, ms)
        return params, mll, cls._log_pmf_normalizada(k_grid, familia, params)

    @staticmethod
    def _ks_cauda(cauda: np.ndarray, k_min: int, pmf: np.ndarray) -> float:
        co = np.sort(cauda)
        ks = np.arange(k_min, int(co[-1]) + 1, dtype=int)
        return float(np.max(np.abs(np.searchsorted(co, ks, side="right") / float(len(co)) - np.cumsum(pmf)[: len(ks)])))

    @staticmethod
    def _vuong(lp1: np.ndarray, lp2: np.ndarray) -> Tuple[float, float]:
        d = lp1 - lp2
        s2 = float(np.mean((d - float(np.mean(d))) ** 2)) if len(d) > 1 else 0.0
        if s2 <= 1e-12:
            return 0.0, 1.0
        z = float(np.sum(d) / math.sqrt(len(d) * s2))
        return z, float(math.erfc(abs(z) / math.sqrt(2.0)))

    @staticmethod
    def _perfil_dir(g_arr: np.ndarray) -> Dict[str, float]:
        gp = np.asarray(g_arr, dtype=float)
        km = float(np.mean(gp)) if len(gp) else 0.0
        kmin = max(1, int(np.ceil(km)))
        cauda = gp[gp >= kmin]
        if len(cauda) < 10:
            gnz = gp[gp > 0]
            kmin = max(1, int(np.percentile(gnz, 50))) if len(gnz) else 1
            cauda = gp[gp >= kmin]
        sl = float(np.sum(np.log(cauda / max(0.5, float(kmin) - 0.5)))) if len(cauda) else 0.0
        ku = np.unique(cauda)
        r2l = r2s = 0.0
        if len(ku) >= 3:
            lp = np.log(np.array([np.mean(gp >= k) for k in ku]))
            st = float(np.sum((lp - np.mean(lp)) ** 2))
            if st > 0:
                c1, c2 = np.polyfit(np.log(ku), lp, 1), np.polyfit(ku, lp, 1)
                r2l = float(1.0 - np.sum((lp - (c1[0] * np.log(ku) + c1[1])) ** 2) / st)
                r2s = float(1.0 - np.sum((lp - (c2[0] * ku + c2[1])) ** 2) / st)
        return {"kappa": float(np.mean(gp ** 2) / (km ** 2)) if km > 0 else 0.0, "gamma_mle": float(1.0 + len(cauda) / sl) if sl > 0 else 0.0, "r2_log_log": r2l, "r2_semi_log": r2s}

    def avaliar_hipotese_scale_free(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        gi = np.asarray(self.graus_real, dtype=int)
        gp = gi[gi >= 1]
        teto = max(5 * int(np.max(gp)), 2000)

        kmin, mks = int(max(1, np.ceil(self.grau_medio))), float("inf")
        for kc in np.unique(gp):
            kci = int(kc)
            if kci < 2 or len(gp[gp >= kci]) < self.min_tail_size:
                continue
            _, _, lpc = self._ajustar_familia_cauda(gp[gp >= kci], kci, np.arange(kci, teto + 1, dtype=float), "PL")
            ks_c = self._ks_cauda(gp[gp >= kci], kci, np.exp(lpc))
            if ks_c < mks:
                mks, kmin = ks_c, kci

        cauda = gp[gp >= kmin]
        nt, k_grid = len(cauda), np.arange(kmin, teto + 1, dtype=float)
        (g_pl,), ll_pl, lp_pl = self._ajustar_familia_cauda(cauda, kmin, k_grid, "PL")
        pmf_pl = np.exp(lp_pl); pmf_pl /= np.sum(pmf_pl)
        d_emp = self._ks_cauda(cauda, kmin, pmf_pl)

        rng, corpo, p_c, kgi = np.random.default_rng(self.seed_base), gi[gi < kmin], float(nt) / float(len(gi)), np.arange(kmin, teto + 1, dtype=int)
        ge = 0
        for _ in range(self.n_bootstrap_ks):
            ncs = int(rng.binomial(len(gi), p_c))
            am = rng.choice(kgi, size=max(10, nt) if (len(corpo) == 0 or ncs < 10) else ncs, replace=True, p=pmf_pl)
            sb = float(np.sum(np.log(am / max(0.5, float(kmin) - 0.5))))
            gb0 = float(np.clip(1.0 + len(am) / sb if sb > 0 else g_pl, 1.10, 6.00))
            ib = np.clip(am - kmin, 0, len(k_grid) - 1)
            mllb, mpb = -float("inf"), pmf_pl
            for gc in np.linspace(max(1.05, gb0 - 0.25), min(6.50, gb0 + 0.25), 11):
                lpb = self._log_pmf_normalizada(k_grid, "PL", (float(gc),))
                llb = float(np.sum(lpb[ib]))
                if llb > mllb:
                    mllb, mpb = llb, np.exp(lpb)
            ge += int(self._ks_cauda(am, kmin, mpb) >= d_emp)
        p_ks = float(ge / float(max(1, self.n_bootstrap_ks)))

        (gt, lt), ll_t, lp_t = self._ajustar_familia_cauda(cauda, kmin, k_grid, "TPL")
        (ml, sl), ll_l, lp_l = self._ajustar_familia_cauda(cauda, kmin, k_grid, "LN")
        (le,), ll_e, lp_e = self._ajustar_familia_cauda(cauda, kmin, k_grid, "EXP")
        io = cauda - kmin
        zt, pt = self._vuong(lp_pl[io], lp_t[io])
        zl, pl = self._vuong(lp_pl[io], lp_l[io])
        ze, pe = self._vuong(lp_pl[io], lp_e[io])

        specs = [
            ("Lei de Potência (PL)", f"gamma={g_pl:.3f}", 1, ll_pl, 0.0, 1.0),
            ("PL Truncada (TPL)", f"gamma={gt:.3f}, lambda={lt:.4f}", 2, ll_t, zt, pt),
            ("Log-Normal (LN)", f"mu={ml:.3f}, sigma={sl:.3f}", 2, ll_l, zl, pl),
            ("Exponencial (EXP)", f"lambda={le:.4f}", 1, ll_e, ze, pe),
        ]
        aics = [2.0 * q - 2.0 * ll for _, _, q, ll, _, _ in specs]
        amin = float(min(aics))
        df_m = pd.DataFrame([
            {"Modelo": nm, "Parametros": ps, "LogLik": float(ll), "AIC": float(am), "Delta_AIC": float(am - amin),
             "BIC": float(qm * math.log(max(1, nt)) - 2.0 * ll), "Vuong_R_vs_PL": float(zv), "Vuong_p_vs_PL": float(pv),
             "Melhor_Que_PL": "Referência (PL)" if nm == "Lei de Potência (PL)" else ("Sim (Vuong p < 0.10)" if (zv < 0 and pv < 0.10) else ("Sim (Delta AIC < -2)" if (am - amin) < (aics[0] - amin - 2.0) else "Não / Inconclusivo"))}
            for (nm, ps, qm, ll, zv, pv), am in zip(specs, aics)
        ])
        m_aic = str(specs[int(np.argmin(aics))][0])
        ku = np.unique(cauda)
        ccdf_e = np.array([np.mean(cauda >= k) for k in ku], dtype=float)
        lpc = np.log(ccdf_e); sst = float(np.sum((lpc - np.mean(lpc)) ** 2))
        r2l = r2s = 0.0
        if len(ku) >= 3 and sst > 0:
            c1, c2 = np.polyfit(np.log(ku.astype(float)), lpc, 1), np.polyfit(ku.astype(float), lpc, 1)
            r2l = float(1.0 - np.sum((lpc - (c1[0] * np.log(ku.astype(float)) + c1[1])) ** 2) / sst)
            r2s = float(1.0 - np.sum((lpc - (c2[0] * ku.astype(float) + c2[1])) ** 2) / sst)

        def _ck(lp: np.ndarray) -> List[float]:
            pv = np.exp(lp); pv /= np.sum(pv)
            cf = np.clip(1.0 - np.concatenate(([0.0], np.cumsum(pv)[:-1])), 1e-12, 1.0)
            return [float(v) for v in cf[np.clip(ku.astype(int) - kmin, 0, len(cf) - 1)]]

        lp_u = np.array([float(k) * math.log(max(1.0, self.grau_medio)) - max(1.0, self.grau_medio) - math.lgamma(float(k) + 1.0) for k in k_grid])
        lp_pois = lp_u - (float(np.max(lp_u)) + float(np.log(np.sum(np.exp(lp_u - float(np.max(lp_u)))))))
        pin = self._perfil_dir(np.array([d for _, d in self.g_quimico_dir.in_degree()], dtype=float))
        pout = self._perfil_dir(np.array([d for _, d in self.g_quimico_dir.out_degree()], dtype=float))
        rival = (m_aic != "Lei de Potência (PL)") or (zt < 0 and pt < 0.10) or (zl < 0 and pl < 0.10) or (r2s > r2l)
        ver = "Rejeitado (Cauda Pesada Truncada — KS p < 0.10)" if p_ks < 0.10 else (f"Não-Scale-Free Puro (Cauda Pesada Truncada — {m_aic} vence AIC)" if rival else "Compatível com Scale-Free na Cauda")

        return {
            "k_min": kmin, "n_tail": nt, "gamma_pl": float(g_pl), "ks_dist_pl": d_emp, "p_valor_ks_bootstrap": p_ks,
            "melhor_modelo_aic": m_aic, "r2_log_log": r2l, "r2_semi_log": r2s,
            "vuong_pl_vs_tpl": {"R": float(zt), "p": float(pt)}, "vuong_pl_vs_ln": {"R": float(zl), "p": float(pl)}, "vuong_pl_vs_exp": {"R": float(ze), "p": float(pe)},
            "direcionado_quimico": {
                "in_degree_kappa": pin["kappa"], "out_degree_kappa": pout["kappa"], "in_degree_gamma": pin["gamma_mle"], "out_degree_gamma": pout["gamma_mle"],
                "in_degree_r2_log_log": pin["r2_log_log"], "in_degree_r2_semi_log": pin["r2_semi_log"], "out_degree_r2_log_log": pout["r2_log_log"], "out_degree_r2_semi_log": pout["r2_semi_log"],
            },
            "curvas_ccdf": {"k_vals": [int(k) for k in ku], "ccdf_empirica": [float(v) for v in ccdf_e], "ccdf_pl": _ck(lp_pl), "ccdf_tpl": _ck(lp_t), "ccdf_ln": _ck(lp_l), "ccdf_exp": _ck(lp_e), "ccdf_poisson": _ck(lp_pois)},
            "veredito": ver,
        }, df_m

    def executar_diagnostico_completo(self) -> Dict[str, Any]:
        res_er = self.avaliar_hipotese_aleatoria()
        res_sw, ws_df = self.avaliar_hipotese_small_world()
        res_sf, mod_df = self.avaliar_hipotese_scale_free()
        scorecard = pd.DataFrame([
            {"Hipotese": "Aleatório (Erdős-Rényi)", "Veredito": res_er["veredito"], "Evidencia_Principal": f"Z_C={res_er['z_score_C']:.1f}, I_D={res_er['indice_dispersao_ID']:.2f}", "Metricas_Chave": f"C_real={res_er['C_real']:.4f}; C_ER={res_er['C_ER_media']:.4f}; D_KS={res_er['ks_dist_vs_ER']:.3f}"},
            {"Hipotese": "Small-World (Mundo Pequeno)", "Veredito": res_sw["veredito"], "Evidencia_Principal": f"sigma={res_sw['sigma']:.2f}, omega={res_sw['omega']:+.2f}, phi_SWP={res_sw['phi_swp']:.2f}", "Metricas_Chave": f"sigma={res_sw['sigma']:.2f}+/-{res_sw['sigma_std']:.2f}; p_equiv_WS={res_sw['p_equiv_ws']:.3f}"},
            {"Hipotese": "Livre de Escala (Scale-Free)", "Veredito": res_sf["veredito"], "Evidencia_Principal": f"k_min={res_sf['k_min']}, n={res_sf['n_tail']}, Melhor_AIC={res_sf['melhor_modelo_aic']}", "Metricas_Chave": f"gamma_PL={res_sf['gamma_pl']:.2f}; p_KS={res_sf['p_valor_ks_bootstrap']:.2f}; Vuong_TPL={res_sf['vuong_pl_vs_tpl']['R']:+.2f}"},
        ])
        resumo = sanitizar_json({"rede_base": {"N": self.n, "M": self.m, "grau_medio": self.grau_medio, "densidade": self.densidade, "k_ws": self.k_ws}, "aleatorio": res_er, "small_world": res_sw, "scale_free": res_sf})
        return {"scorecard_df": scorecard, "modelos_cauda_df": mod_df, "ws_sweep_df": ws_df, "resumo_json": resumo}
