"""Módulo de difusão de informação (estrutural + simulações SI e IC/SIR) no conectoma C. elegans."""

import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import networkx as nx
import numpy as np
import pandas as pd
from graph_utils import extrair_componente_principal, gerar_rede_sintetica, projetar_grafo, sanitizar_json

VISAO_COMB_UNDIR = "Combinada (Não-Direcionada)"
VISAO_COMB_DIR = "Combinada (Direcionada)"
VISAO_QUIM_DIR = "Química (Direcionada)"
CENARIO_REAL_HUBS = "C. elegans (Hubs / Rich-Club)"
CENARIO_REAL_SENS = "C. elegans (Sensorial)"
CENARIO_REAL_ALEAT = "C. elegans (Aleatório)"
CENARIO_ER_HUBS = "Erdős-Rényi (Top Hubs)"
CENARIO_ER_ALEAT = "Erdős-Rényi (Aleatório)"
ORDEM_CENARIOS = [CENARIO_REAL_HUBS, CENARIO_REAL_SENS, CENARIO_REAL_ALEAT, CENARIO_ER_HUBS, CENARIO_ER_ALEAT]


def _nanstat(v: np.ndarray, fn) -> float:
    ok = v[~np.isnan(v)] if v.size else v
    return float(fn(ok)) if ok.size else float("nan")


class InformationDiffusionAnalyzer:
    """Analisa a facilidade de difusão de informação estrutural e dinâmica no conectoma."""

    def __init__(
        self, graph: nx.MultiDiGraph, membros_rich_club: Optional[Set[str]] = None,
        n_runs_er_estrutural: int = 20, n_runs_si: int = 150, n_runs_ic: int = 150,
        beta_si_padrao: float = 0.08, max_steps_si: int = 25,
        betas_ic: Optional[List[float]] = None, beta_ic_referencia: float = 0.08, seed_base: int = 42,
    ) -> None:
        self.graph = graph
        self.n_runs_er_estrutural, self.n_runs_si, self.n_runs_ic = int(n_runs_er_estrutural), int(n_runs_si), int(n_runs_ic)
        self.beta_si_padrao, self.max_steps_si, self.beta_ic_referencia = float(beta_si_padrao), int(max_steps_si), float(beta_ic_referencia)
        self.betas_ic = [float(b) for b in (betas_ic or [0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35])]
        self.seed_base = int(seed_base)

        self.g_comb_undir = projetar_grafo(graph, "Combinada", direcionado=False, adicionar_distancia=True)
        self.g_comb_dir = projetar_grafo(graph, "Combinada", direcionado=True, adicionar_distancia=True)
        self.g_quim_dir = projetar_grafo(graph, "Quimica", direcionado=True, adicionar_distancia=True)
        self.nos_sensoriais = sorted(n for n, d in self.g_comb_undir.nodes(data=True) if d.get("tipo_funcional") == "Sensorial")
        self.nos_motores = sorted(n for n, d in self.g_comb_undir.nodes(data=True) if d.get("tipo_funcional") == "Motor")
        hubs_ok = sorted(n for n in (membros_rich_club or set()) if n in self.g_comb_undir)
        self.nos_hubs = hubs_ok or sorted(n for n, _ in sorted(self.g_comb_undir.degree(), key=lambda x: (-x[1], x[0]))[:6])
        self._pools_er: Dict[str, List[Union[nx.Graph, nx.DiGraph]]] = {}

    def _obter_pool_er(self, visao: str, n: int, m: int, directed: bool, offset: int) -> List[Union[nx.Graph, nx.DiGraph]]:
        if visao not in self._pools_er:
            self._pools_er[visao] = [
                gerar_rede_sintetica("ER", n, m, seed=self.seed_base + offset + r, direcionado=directed)
                for r in range(self.n_runs_er_estrutural)
            ]
        return self._pools_er[visao]

    @staticmethod
    def _metricas_estruturais_grafo(g: Union[nx.Graph, nx.DiGraph]) -> Dict[str, Any]:
        n, m = g.number_of_nodes(), g.number_of_edges()
        if n == 0:
            return {"N": 0, "M": 0, "Fracao_Componente": 0.0, "L": 0.0, "D": 0.0, "E_Glob": 0.0,
                    "Frac_Pares_Alcancaveis": 0.0, "Lambda_c": float("nan"), "Dist_Fracoes": {}, "Pct_Ate_3_Saltos": 0.0}
        comp = extrair_componente_principal(g)
        nc = comp.number_of_nodes()
        l_val = float(nx.average_shortest_path_length(comp)) if nc > 1 else 0.0
        d_val = float(nx.diameter(comp)) if nc > 1 else 0.0
        tot, soma_inv, alc, cont_d = n * (n - 1), 0.0, 0, {}
        for u, dists in nx.all_pairs_shortest_path_length(g):
            for v, d in dists.items():
                if u != v and d > 0:
                    soma_inv += 1.0 / d; alc += 1; cont_d[d] = cont_d.get(d, 0) + 1
        df = {d: c / float(tot) for d, c in cont_d.items()} if tot > 0 else {}
        lam_c = float("nan")
        if not g.is_directed():
            gr = np.array([deg for _, deg in g.degree()], dtype=float)
            k2 = float(np.mean(gr ** 2))
            lam_c = float(np.mean(gr)) / k2 if k2 > 0 else float("nan")
        return {
            "N": n, "M": m, "Fracao_Componente": nc / float(n), "L": l_val, "D": d_val,
            "E_Glob": soma_inv / float(tot) if tot > 0 else 0.0,
            "Frac_Pares_Alcancaveis": alc / float(tot) if tot > 0 else 0.0,
            "Lambda_c": lam_c, "Dist_Fracoes": df, "Pct_Ate_3_Saltos": sum(df.get(d, 0.0) for d in (1, 2, 3)) * 100.0,
        }

    def _calcular_canal_biologico(self, g_dir: nx.DiGraph) -> Dict[str, float]:
        mot = set(self.nos_motores)

        def _medir(origens: List[str]) -> Tuple[float, float, float]:
            dt, dp, tot = [], [], 0
            for s in origens:
                if s not in g_dir:
                    continue
                spt = nx.single_source_shortest_path_length(g_dir, s)
                spp = nx.single_source_dijkstra_path_length(g_dir, s, weight="distance")
                for m in mot:
                    if m != s and m in g_dir:
                        tot += 1
                        if m in spt:
                            dt.append(float(spt[m]))
                        if m in spp:
                            dp.append(float(spp[m]))
            return (float(np.mean(dt)) if dt else float("nan"), (len(dt) / float(tot) * 100.0) if tot > 0 else 0.0, float(np.mean(dp)) if dp else float("nan"))

        lsm, asm, lpsm = _medir(self.nos_sensoriais)
        lhm, ahm, lphm = _medir(self.nos_hubs)
        return {"L_sensorial_para_motor": lsm, "alcance_sensorial_para_motor_pct": asm, "L_ponderado_sensorial_para_motor": lpsm,
                "L_hubs_para_motor": lhm, "alcance_hubs_para_motor_pct": ahm, "L_ponderado_hubs_para_motor": lphm}

    def _executar_analise_estrutural(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        l_est, l_dist, det = [], [], {}
        for visao, gr, directed, off in ((VISAO_COMB_UNDIR, self.g_comb_undir, False, 1000), (VISAO_COMB_DIR, self.g_comb_dir, True, 2000), (VISAO_QUIM_DIR, self.g_quim_dir, True, 3000)):
            mr = self._metricas_estruturais_grafo(gr)
            er_m = [self._metricas_estruturais_grafo(g) for g in self._obter_pool_er(visao, mr["N"], mr["M"], directed, off)]
            lm, ls = float(np.mean([x["L"] for x in er_m])), float(np.std([x["L"] for x in er_m]))
            dm, ds = float(np.mean([x["D"] for x in er_m])), float(np.std([x["D"] for x in er_m]))
            em, es = float(np.mean([x["E_Glob"] for x in er_m])), float(np.std([x["E_Glob"] for x in er_m]))
            lcm = _nanstat(np.asarray([x["Lambda_c"] for x in er_m], dtype=float), np.mean)
            l_est.append({
                "Visao": visao, "N": mr["N"], "M": mr["M"], "Fracao_Componente_Real": mr["Fracao_Componente"],
                "L_Real": mr["L"], "L_ER_Media": lm, "L_ER_Std": ls, "Razao_L_ER": mr["L"] / lm if lm > 0 else float("nan"),
                "D_Real": mr["D"], "D_ER_Media": dm, "D_ER_Std": ds,
                "E_Glob_Real": mr["E_Glob"], "E_Glob_ER_Media": em, "E_Glob_ER_Std": es,
                "Razao_E_ER": mr["E_Glob"] / em if em > 0 else float("nan"),
                "Z_Score_E_Glob": (mr["E_Glob"] - em) / es if es > 1e-12 else 0.0,
                "Lambda_c_Real": mr["Lambda_c"], "Lambda_c_ER_Media": lcm,
            })
            det[visao] = {
                "L_real": mr["L"], "L_er_media": lm, "D_real": mr["D"], "D_er_media": dm,
                "E_glob_real": mr["E_Glob"], "E_glob_er_media": em, "lambda_c_real": mr["Lambda_c"], "lambda_c_er_media": lcm,
                "pct_pares_ate_3_saltos_real": mr["Pct_Ate_3_Saltos"], "pct_pares_ate_3_saltos_er": float(np.mean([x["Pct_Ate_3_Saltos"] for x in er_m])),
            }
            if visao in {VISAO_COMB_UNDIR, VISAO_COMB_DIR}:
                for rot in ("1", "2", "3", "4", "5+"):
                    pr = (sum(v for k, v in mr["Dist_Fracoes"].items() if k >= 5) if rot == "5+" else mr["Dist_Fracoes"].get(int(rot), 0.0)) * 100.0
                    vals = np.array([(sum(v for k, v in x["Dist_Fracoes"].items() if k >= 5) if rot == "5+" else x["Dist_Fracoes"].get(int(rot), 0.0)) * 100.0 for x in er_m])
                    l_dist.append({"Visao": visao, "Distancia": rot, "Pct_Real": pr, "Pct_ER_Media": float(np.mean(vals)), "Pct_ER_Std": float(np.std(vals))})
        det[VISAO_COMB_DIR].update(self._calcular_canal_biologico(self.g_comb_dir))
        det[VISAO_QUIM_DIR].update(self._calcular_canal_biologico(self.g_quim_dir))
        return pd.DataFrame(l_est), pd.DataFrame(l_dist), det

    @staticmethod
    def _grafo_para_adj_indexada(g: Union[nx.Graph, nx.DiGraph]) -> Tuple[List[Any], Dict[Any, int], List[List[int]], List[List[float]], List[int]]:
        nos = list(g.nodes())
        mapa = {no: i for i, no in enumerate(nos)}
        adj, adj_w = [], []
        for ui, u in enumerate(nos):
            lv, lw = [], []
            for v in (g.successors(u) if g.is_directed() else g.neighbors(u)):
                vi = mapa[v]
                if vi != ui:
                    lv.append(vi); lw.append(max(float(g[u][v].get("weight", 1.0)), 0.0))
            adj.append(lv); adj_w.append(lw)
        return nos, mapa, adj, adj_w, sorted(range(len(nos)), key=lambda i: (-len(adj[i]), i))[:6]

    @staticmethod
    def _ativar_passo(front: List[int], ativos: np.ndarray, adj: List[List[int]], beta: float, rng: np.random.Generator, adj_w: Optional[List[List[float]]] = None) -> List[int]:
        bc, q = min(max(float(beta), 0.0), 1.0), 1.0 - min(max(float(beta), 0.0), 1.0)
        if adj_w is None:
            cont: Dict[int, int] = {}
            for u in front:
                for v in adj[u]:
                    if not ativos[v]:
                        cont[v] = cont.get(v, 0) + 1
            if not cont:
                return []
            cands = list(cont.keys())
            mv = np.fromiter(cont.values(), dtype=np.float64, count=len(cands))
            probs = np.ones_like(mv) if bc >= 1.0 else (1.0 - np.power(q, mv))
        else:
            sob: Dict[int, float] = {}
            for u in front:
                for v, w in zip(adj[u], adj_w[u]):
                    if not ativos[v]:
                        sob[v] = sob.get(v, 1.0) * ((0.0 if w > 0 else 1.0) if bc >= 1.0 else math.pow(q, w))
            if not sob:
                return []
            cands = list(sob.keys())
            probs = np.array([1.0 - sob[v] for v in cands], dtype=np.float64)
        sorteios = rng.random(len(cands))
        novos = [v for i, v in enumerate(cands) if sorteios[i] < probs[i]]
        for v in novos:
            ativos[v] = True
        return novos

    def _rodar_si(self, n: int, adj: List[List[int]], si: int, beta: float, steps: int, rng: np.random.Generator, mm: Optional[np.ndarray], nm: int) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        ativos = np.zeros(n, dtype=bool); ativos[si] = True
        al, nat, nma = [si], 1, (1 if (mm is not None and mm[si]) else 0)
        ct, cm = np.zeros(steps + 1), np.full(steps + 1, np.nan)
        ct[0] = nat / float(n)
        if mm is not None and nm > 0:
            cm[0] = nma / float(nm)
        t50, t80 = (0.0 if ct[0] >= 0.5 else float("nan")), (0.0 if ct[0] >= 0.8 else float("nan"))
        tm50 = 0.0 if (mm is not None and nm > 0 and cm[0] >= 0.5) else float("nan")
        for p in range(1, steps + 1):
            if nat < n:
                nv = self._ativar_passo(al, ativos, adj, beta, rng)
                al.extend(nv); nat += len(nv)
                if mm is not None and nv:
                    nma += sum(1 for v in nv if mm[v])
            ct[p] = nat / float(n)
            if math.isnan(t50) and ct[p] >= 0.5:
                t50 = float(p)
            if math.isnan(t80) and ct[p] >= 0.8:
                t80 = float(p)
            if mm is not None and nm > 0:
                cm[p] = nma / float(nm)
                if math.isnan(tm50) and cm[p] >= 0.5:
                    tm50 = float(p)
        return ct, cm, t50, t80, float(np.mean(ct[1:])) if steps > 0 else ct[0], tm50

    def _rodar_ic(self, n: int, adj: List[List[int]], adj_w: Optional[List[List[float]]], si: int, beta: float, rng: np.random.Generator, mm: Optional[np.ndarray], nm: int) -> Tuple[float, int, float, float]:
        ativos = np.zeros(n, dtype=bool); ativos[si] = True
        front, nat, nma, passos = [si], 1, (1 if (mm is not None and mm[si]) else 0), 0
        while front:
            nv = self._ativar_passo(front, ativos, adj, beta, rng, adj_w=adj_w)
            if nv:
                passos += 1; nat += len(nv)
                if mm is not None:
                    nma += sum(1 for v in nv if mm[v])
            front = nv
        alc = (nat / float(n)) * 100.0
        return alc, passos, (100.0 if alc >= 20.0 else 0.0), ((nma / float(nm)) * 100.0 if (mm is not None and nm > 0) else float("nan"))

    def _selecionar_instancia(self, cen: str, r: int, rng: np.random.Generator, rctx, er_ests):
        nr, ar, mmr, nmr, hr, sr = rctx
        if cen.startswith("C. elegans"):
            si = hr[r % len(hr)] if cen == CENARIO_REAL_HUBS else (int(sr[int(rng.integers(0, len(sr)))]) if cen == CENARIO_REAL_SENS else int(rng.integers(0, nr)))
            return nr, ar, si, mmr, nmr
        ne, _, ae, _, he = er_ests[r % len(er_ests)]
        return len(ne), ae, (he[r % len(he)] if cen == CENARIO_ER_HUBS else int(rng.integers(0, len(ne)))), None, 0

    def _executar_simulacoes_visao(self, visao: str, gr: Union[nx.Graph, nx.DiGraph], directed: bool, off: int):
        nr, mr, ar, _, _ = self._grafo_para_adj_indexada(gr)
        mmr = np.array([no in set(self.nos_motores) for no in nr], dtype=bool)
        rctx = (len(nr), ar, mmr, int(mmr.sum()), [mr[h] for h in self.nos_hubs if h in mr], [mr[s] for s in self.nos_sensoriais if s in mr])
        ers = [self._grafo_para_adj_indexada(g) for g in self._obter_pool_er(visao, gr.number_of_nodes(), gr.number_of_edges(), directed, off)]
        lsic, lsir, lic = [], [], []

        for ci, cen in enumerate(ORDEM_CENARIOS):
            rng = np.random.default_rng(self.seed_base + off + 500 + ci)
            mc, mm = np.zeros((self.n_runs_si, self.max_steps_si + 1)), np.full((self.n_runs_si, self.max_steps_si + 1), np.nan)
            t50s, t80s, aucs, tm50s = np.full(self.n_runs_si, np.nan), np.full(self.n_runs_si, np.nan), np.zeros(self.n_runs_si), np.full(self.n_runs_si, np.nan)
            for r in range(self.n_runs_si):
                nc, ac, si, mk, nm = self._selecionar_instancia(cen, r, rng, rctx, ers)
                ct, cm, t50s[r], t80s[r], aucs[r], tm50s[r] = self._rodar_si(nc, ac, si, self.beta_si_padrao, self.max_steps_si, rng, mk, nm)
                mc[r], mm[r] = ct * 100.0, cm * 100.0
            for t in range(self.max_steps_si + 1):
                lsic.append({"Passo_t": t, "Cenario": cen, "Visao": visao, "Infectados_Pct_Media": float(np.mean(mc[:, t])), "Infectados_Pct_Std": float(np.std(mc[:, t])), "Motores_Pct_Media": _nanstat(mm[:, t], np.mean)})
            lsir.append({"Visao": visao, "Cenario": cen, "Beta_SI": self.beta_si_padrao, "t50_Media": _nanstat(t50s, np.mean), "t50_Std": _nanstat(t50s, np.std), "t80_Media": _nanstat(t80s, np.mean), "Frac_Atingiu_50_Pct": float(np.mean(~np.isnan(t50s)) * 100.0), "AUC_SI_Media": float(np.mean(aucs)), "AUC_SI_Std": float(np.std(aucs)), "Infectados_Final_Pct": float(np.mean(mc[:, -1])), "t_motor_50_Media": _nanstat(tm50s, np.mean)})

        for ci, cen in enumerate(ORDEM_CENARIOS):
            for bi, beta in enumerate(self.betas_ic):
                rng = np.random.default_rng(self.seed_base + off + 700 + ci * 50 + bi)
                alcs, durs, glbs, mots = np.zeros(self.n_runs_ic), np.zeros(self.n_runs_ic), np.zeros(self.n_runs_ic), np.full(self.n_runs_ic, np.nan)
                for r in range(self.n_runs_ic):
                    nc, ac, si, mk, nm = self._selecionar_instancia(cen, r, rng, rctx, ers)
                    alcs[r], durs[r], glbs[r], mots[r] = self._rodar_ic(nc, ac, None, si, beta, rng, mk, nm)
                lic.append({"Visao": visao, "Cenario": cen, "Beta": beta, "Alcance_Final_Pct_Media": float(np.mean(alcs)), "Alcance_Final_Pct_Std": float(np.std(alcs)), "Duracao_Media": float(np.mean(durs)), "Prob_Cascata_Global_Pct": float(np.mean(glbs)), "Alcance_Motores_Pct_Media": _nanstat(mots, np.mean)})
        return lsic, lsir, lic

    def _executar_ic_ponderada_real(self) -> pd.DataFrame:
        nr, mr, ar, aw, _ = self._grafo_para_adj_indexada(self.g_comb_dir)
        mm = np.array([no in set(self.nos_motores) for no in nr], dtype=bool)
        nm, linhas = int(mm.sum()), []
        for ci, (cen, pool, rot) in enumerate(((CENARIO_REAL_HUBS, [mr[h] for h in self.nos_hubs if h in mr], True), (CENARIO_REAL_SENS, [mr[s] for s in self.nos_sensoriais if s in mr], False))):
            for bi, beta in enumerate(self.betas_ic):
                rng = np.random.default_rng(self.seed_base + 9000 + ci * 50 + bi)
                alcs, durs, glbs, mots = np.zeros(self.n_runs_ic), np.zeros(self.n_runs_ic), np.zeros(self.n_runs_ic), np.full(self.n_runs_ic, np.nan)
                for r in range(self.n_runs_ic):
                    si = pool[r % len(pool)] if rot else int(pool[int(rng.integers(0, len(pool)))])
                    alcs[r], durs[r], glbs[r], mots[r] = self._rodar_ic(len(nr), ar, aw, si, beta, rng, mm, nm)
                linhas.append({"Visao": VISAO_COMB_DIR, "Cenario": cen, "Beta": beta, "Alcance_Final_Pct_Media": float(np.mean(alcs)), "Alcance_Final_Pct_Std": float(np.std(alcs)), "Duracao_Media": float(np.mean(durs)), "Prob_Cascata_Global_Pct": float(np.mean(glbs)), "Alcance_Motores_Pct_Media": _nanstat(mots, np.mean)})
        return pd.DataFrame(linhas)

    @staticmethod
    def _buscar(df: pd.DataFrame, filtros: Dict[str, Any], col: str) -> float:
        mask = pd.Series(True, index=df.index)
        for c, v in filtros.items():
            mask &= np.isclose(df[c].to_numpy(dtype=float), v, atol=1e-9) if isinstance(v, float) else (df[c] == v)
        sub = df.loc[mask, col]
        return float(sub.iloc[0]) if not sub.empty else float("nan")

    def executar_analise_completa(self) -> Dict[str, Any]:
        df_est, df_dist, det = self._executar_analise_estrutural()
        si_cu, si_ru, ic_cu = self._executar_simulacoes_visao(VISAO_COMB_UNDIR, self.g_comb_undir, False, 1000)
        si_cd, si_rd, ic_cd = self._executar_simulacoes_visao(VISAO_COMB_DIR, self.g_comb_dir, True, 2000)
        df_si_c, df_si_r, df_ic_c, df_ic_p = pd.DataFrame(si_cu + si_cd), pd.DataFrame(si_ru + si_rd), pd.DataFrame(ic_cu + ic_cd), self._executar_ic_ponderada_real()
        br, eu, ed, eq = self.beta_ic_referencia, det[VISAO_COMB_UNDIR], det[VISAO_COMB_DIR], det[VISAO_QUIM_DIR]
        resumo = sanitizar_json({
            "metadados": {"N": self.g_comb_undir.number_of_nodes(), "n_sensoriais": len(self.nos_sensoriais), "n_motores": len(self.nos_motores), "membros_hubs": list(self.nos_hubs), "beta_si_padrao": self.beta_si_padrao, "beta_ic_referencia": br},
            "estrutural": {
                "combinada_undir": eu,
                "combinada_dir": {k: ed[k] for k in ("L_real", "L_er_media", "D_real", "D_er_media", "E_glob_real", "E_glob_er_media", "L_sensorial_para_motor", "alcance_sensorial_para_motor_pct", "L_hubs_para_motor")},
                "quimica_dir": {k: eq[k] for k in ("L_real", "D_real", "E_glob_real", "L_sensorial_para_motor")},
            },
            "simulacao_si": {
                "combinada_undir": {
                    "hubs_t50": self._buscar(df_si_r, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_HUBS}, "t50_Media"),
                    "hubs_auc": self._buscar(df_si_r, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_HUBS}, "AUC_SI_Media"),
                    "sensorial_t50": self._buscar(df_si_r, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_SENS}, "t50_Media"),
                    "sensorial_auc": self._buscar(df_si_r, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_SENS}, "AUC_SI_Media"),
                    "aleatorio_real_t50": self._buscar(df_si_r, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_REAL_ALEAT}, "t50_Media"),
                    "er_hubs_t50": self._buscar(df_si_r, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_ER_HUBS}, "t50_Media"),
                    "er_aleatorio_t50": self._buscar(df_si_r, {"Visao": VISAO_COMB_UNDIR, "Cenario": CENARIO_ER_ALEAT}, "t50_Media"),
                },
                "combinada_dir": {
                    "hubs_t_motor_50": self._buscar(df_si_r, {"Visao": VISAO_COMB_DIR, "Cenario": CENARIO_REAL_HUBS}, "t_motor_50_Media"),
                    "sensorial_t_motor_50": self._buscar(df_si_r, {"Visao": VISAO_COMB_DIR, "Cenario": CENARIO_REAL_SENS}, "t_motor_50_Media"),
                },
            },
            "simulacao_ic": {
                "beta_referencia": br,
                "combinada_undir": {k: self._buscar(df_ic_c, {"Visao": VISAO_COMB_UNDIR, "Cenario": c, "Beta": br}, "Alcance_Final_Pct_Media") for k, c in (("hubs_alcance_pct", CENARIO_REAL_HUBS), ("sensorial_alcance_pct", CENARIO_REAL_SENS), ("aleatorio_real_alcance_pct", CENARIO_REAL_ALEAT), ("er_hubs_alcance_pct", CENARIO_ER_HUBS), ("er_aleatorio_alcance_pct", CENARIO_ER_ALEAT))},
                "combinada_dir_ponderada": {"hubs_alcance_pct": self._buscar(df_ic_p, {"Visao": VISAO_COMB_DIR, "Cenario": CENARIO_REAL_HUBS, "Beta": br}, "Alcance_Final_Pct_Media"), "sensorial_alcance_pct": self._buscar(df_ic_p, {"Visao": VISAO_COMB_DIR, "Cenario": CENARIO_REAL_SENS, "Beta": br}, "Alcance_Final_Pct_Media")},
            },
        }, nan_como_none=True)
        return {"df_estrutural": df_est, "df_distancias": df_dist, "df_si_curvas": df_si_c, "df_si_resumo": df_si_r, "df_ic_curvas": df_ic_c, "df_ic_ponderada": df_ic_p, "resumo_json": resumo}
