"""Módulo de preservação de informação (Reciprocidade, SCC/Bow-Tie, Modularidade e Ciclos C2/C3)."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from config import ORDEM_CLASSES_PADRAO
from graph_utils import gerar_nulo_direcionado, projetar_grafo, resumir_contra_nulo, sanitizar_json

SETORES_BOWTIE = ["IN (Entrada / Feeders)", "SCC (Núcleo Recorrente)", "OUT (Saída / Sinks)", "Outros (Isolados/Tubes)"]


class InformationPreservationAnalyzer:
    """Quantifica a facilidade de preservação de informação no conectoma."""

    def __init__(
        self, graph: nx.MultiDiGraph, membros_rich_club: Optional[Set[str]] = None,
        n_runs_null: int = 30, swaps_fator: int = 3, louvain_resolution: float = 1.0, seed_base: int = 42,
    ) -> None:
        self.graph = graph
        self.n_runs_null, self.swaps_fator = max(1, int(n_runs_null)), max(1, int(swaps_fator))
        self.louvain_resolution, self.seed_base = float(louvain_resolution), int(seed_base)

        self.g_quim_dir = projetar_grafo(graph, "Quimica", direcionado=True)
        self.g_comb_dir = projetar_grafo(graph, "Combinada", direcionado=True)
        self.g_comb_undir = projetar_grafo(graph, "Combinada", direcionado=False)
        self.membros_rich_club: Set[str] = (
            set(membros_rich_club) if membros_rich_club is not None
            else {no for no, _ in sorted(self.g_comb_undir.degree(), key=lambda x: (-x[1], x[0]))[:6]}
        )
        pres = {d.get("tipo_funcional", "Desconhecido") for _, d in self.g_quim_dir.nodes(data=True)}
        self.ordem_classes = [c for c in ORDEM_CLASSES_PADRAO if c in pres] + sorted(c for c in pres if c not in ORDEM_CLASSES_PADRAO)

    @staticmethod
    def _calcular_nmi(la: List[str], lb: List[str]) -> float:
        n = len(la)
        if n == 0 or len(lb) != n:
            return 0.0
        _, ia = np.unique(la, return_inverse=True)
        _, ib = np.unique(lb, return_inverse=True)
        cont = np.zeros((int(ia.max()) + 1, int(ib.max()) + 1), dtype=np.float64)
        np.add.at(cont, (ia, ib), 1.0)
        am, bm = cont.sum(1), cont.sum(0)
        pa, pb = am[am > 0] / n, bm[bm > 0] / n
        ha, hb = float(-np.sum(pa * np.log(pa))), float(-np.sum(pb * np.log(pb)))
        if (ha + hb) <= 1e-12:
            return 0.0
        nz = cont > 0
        mi = float(np.sum((cont[nz] / n) * np.log((n * cont[nz]) / np.outer(am, bm)[nz])))
        return float(min(1.0, max(0.0, (2.0 * mi) / (ha + hb))))

    @staticmethod
    def _detectar_comunidades(g: nx.Graph, weight: Optional[str], res: float, seed: int) -> List[Set[str]]:
        if g.number_of_nodes() == 0:
            return []
        if g.number_of_edges() == 0:
            return [{n} for n in g.nodes()]
        try:
            return [set(c) for c in nx.community.louvain_communities(g, weight=weight, resolution=res, seed=seed)]
        except Exception:
            try:
                return [set(c) for c in nx.community.greedy_modularity_communities(g, weight=weight, resolution=res)]
            except TypeError:
                return [set(c) for c in nx.community.greedy_modularity_communities(g, weight=weight)]

    @staticmethod
    def _mod_segura(g: nx.Graph, coms: List[Set[str]], weight: Optional[str]) -> float:
        return float(nx.community.modularity(g, coms, weight=weight)) if (g.number_of_edges() > 0 and coms) else 0.0

    @staticmethod
    def _retencao_intra(g: nx.Graph, coms: List[Set[str]]) -> float:
        mapa = {no: i for i, com in enumerate(coms) for no in com}
        tot = intra = 0.0
        for u, v, d in g.edges(data=True):
            w = float(d.get("weight", 1.0)); tot += w
            if mapa.get(u) == mapa.get(v):
                intra += w
        return float((intra / tot) * 100.0) if tot > 0 else 0.0

    @staticmethod
    def _reciprocidade_grafo(g: nx.DiGraph) -> Tuple[float, float]:
        if g.number_of_edges() == 0:
            return 0.0, 0.0
        tot = rec = 0.0
        for u, v, d in g.edges(data=True):
            w = float(d.get("weight", 1.0)); tot += w
            if g.has_edge(v, u):
                rec += min(w, float(g[v][u].get("weight", 1.0)))
        return float(nx.overall_reciprocity(g)), float(rec / tot) if tot > 0 else 0.0

    @staticmethod
    def _contar_ciclos(g: nx.DiGraph, nos: List[str]) -> Tuple[int, int, np.ndarray, np.ndarray]:
        n = len(nos)
        if n == 0:
            z = np.zeros(0, dtype=np.int64)
            return 0, 0, z, z
        idx = {no: i for i, no in enumerate(nos)}
        a = np.zeros((n, n), dtype=np.int64)
        for u, v in g.edges():
            if u != v:
                a[idx[u], idx[v]] = 1
        a2 = a @ a
        c2, c3 = np.diag(a2).astype(np.int64), np.diag(a2 @ a).astype(np.int64)
        return int(np.sum(c2) // 2), int(np.sum(c3) // 3), c2, c3

    @staticmethod
    def _linha(pilar: str, met: str, cam: str, val: float, st: Optional[Dict[str, float]]) -> Dict[str, Any]:
        return {"Pilar": pilar, "Metrica": met, "Camada": cam, "Valor_Real": val,
                "Nulo_Media": st["nulo_media"] if st else np.nan, "Nulo_Std": st["nulo_std"] if st else np.nan,
                "Razao_Real_Nulo": st["razao"] if st else np.nan, "Z_Score": st["z_score"] if st else np.nan}

    def _metricas_reciprocidade(self, nr: Sequence[float], nrw: Sequence[float], nrc: Sequence[float]) -> Dict[str, Any]:
        rq, rwq = self._reciprocidade_grafo(self.g_quim_dir)
        rc, rwc = self._reciprocidade_grafo(self.g_comb_dir)
        sr, srw, src = resumir_contra_nulo(rq, nr), resumir_contra_nulo(rwq, nrw), resumir_contra_nulo(rc, nrc or nr)
        ctot = pd.DataFrame(0, index=self.ordem_classes, columns=self.ordem_classes, dtype=int)
        crec = pd.DataFrame(0, index=self.ordem_classes, columns=self.ordem_classes, dtype=int)
        intra = inter = 0
        vistos: Set[Tuple[str, str]] = set()
        for u, v in self.g_quim_dir.edges():
            cu, cv = self.g_quim_dir.nodes[u].get("tipo_funcional", "Desconhecido"), self.g_quim_dir.nodes[v].get("tipo_funcional", "Desconhecido")
            if cu in ctot.index and cv in ctot.columns:
                ctot.loc[cu, cv] += 1
                if self.g_quim_dir.has_edge(v, u):
                    crec.loc[cu, cv] += 1
                    par = (u, v) if u < v else (v, u)
                    if par not in vistos:
                        vistos.add(par); intra += int(cu == cv); inter += int(cu != cv)
        mat = pd.DataFrame([[float(crec.loc[a, b] / ctot.loc[a, b]) if ctot.loc[a, b] > 0 else 0.0 for b in self.ordem_classes] for a in self.ordem_classes], index=self.ordem_classes, columns=self.ordem_classes)
        return {
            "json": {
                "quimica_topologica": rq, "quimica_ponderada": rwq, "quimica_nulo_topologica_media": sr["nulo_media"], "quimica_nulo_topologica_std": sr["nulo_std"], "z_score_reciprocidade": sr["z_score"],
                "quimica_nulo_ponderada_media": srw["nulo_media"], "quimica_nulo_ponderada_std": srw["nulo_std"], "z_score_reciprocidade_ponderada": srw["z_score"],
                "combinada_topologica": rc, "combinada_ponderada": rwc, "combinada_nulo_topologica_media": src["nulo_media"], "combinada_nulo_topologica_std": src["nulo_std"], "z_score_combinada_topologica": src["z_score"],
                "pares_reciprocos_intra_classe": intra, "pares_reciprocos_inter_classe": inter,
            },
            "linhas": [
                self._linha("1. Reciprocidade", "Reciprocidade Topológica (R)", "Quimica", rq, sr),
                self._linha("1. Reciprocidade", "Reciprocidade Ponderada (R_w)", "Quimica", rwq, srw),
                self._linha("1. Reciprocidade", "Reciprocidade Topológica (R)", "Combinada", rc, src),
                self._linha("1. Reciprocidade", "Reciprocidade Ponderada (R_w)", "Combinada", rwc, None),
            ],
            "mat_reciprocidade": mat,
        }

    def _metricas_scc_bowtie(self, null_lscc: Sequence[float]) -> Dict[str, Any]:
        nn = self.g_quim_dir.number_of_nodes()
        sq = sorted(nx.strongly_connected_components(self.g_quim_dir), key=len, reverse=True)
        lq, lc = (set(sq[0]) if sq else set()), (set(max(nx.strongly_connected_components(self.g_comb_dir), key=len, default=set())))
        rc_in = len(self.membros_rich_club & lq)
        rep = next(iter(lq)) if lq else None
        anc, desc = (nx.ancestors(self.g_quim_dir, rep) if rep else set()), (nx.descendants(self.g_quim_dir, rep) if rep else set())
        cont, c_cls = {s: 0 for s in SETORES_BOWTIE}, {s: {c: 0 for c in self.ordem_classes} for s in SETORES_BOWTIE}
        for no, d in self.g_quim_dir.nodes(data=True):
            s = "SCC (Núcleo Recorrente)" if no in lq else ("IN (Entrada / Feeders)" if no in anc else ("OUT (Saída / Sinks)" if no in desc else "Outros (Isolados/Tubes)"))
            cont[s] += 1
            cls = d.get("tipo_funcional", "Desconhecido")
            if cls in c_cls[s]:
                c_cls[s][cls] += 1
        df_bt = pd.DataFrame([{"Setor_BowTie": s, **{c: c_cls[s][c] for c in self.ordem_classes}, "Total": cont[s], "Fracao_Pct": round((cont[s] / nn) * 100.0, 2) if nn > 0 else 0.0} for s in SETORES_BOWTIE])
        st = resumir_contra_nulo(float(len(lq)), null_lscc)
        return {
            "json": {
                "quimica_lscc_tamanho": len(lq), "quimica_lscc_fracao_pct": (len(lq) / nn) * 100.0 if nn > 0 else 0.0,
                "quimica_n_sccs_nao_triviais": sum(1 for c in sq if len(c) >= 2), "quimica_lscc_nulo_media": st["nulo_media"], "quimica_lscc_nulo_std": st["nulo_std"], "quimica_lscc_z_score": st["z_score"],
                "combinada_lscc_tamanho": len(lc), "combinada_lscc_fracao_pct": (len(lc) / nn) * 100.0 if nn > 0 else 0.0,
                "rich_club_na_lscc_quimica_qtd": rc_in, "rich_club_total": len(self.membros_rich_club),
                "rich_club_na_lscc_quimica_pct": (rc_in / len(self.membros_rich_club)) * 100.0 if self.membros_rich_club else 0.0, "bowtie_contagens": cont,
            },
            "linhas": [self._linha("2. SCC e Bow-Tie", "Tamanho LSCC (|V_LSCC|)", "Quimica", float(len(lq)), st), self._linha("2. SCC e Bow-Tie", "Tamanho LSCC (|V_LSCC|)", "Combinada", float(len(lc)), None)],
            "df_bowtie_classes": df_bt,
        }

    def _metricas_modularidade(self, null_q: Sequence[float]) -> Dict[str, Any]:
        res = self.louvain_resolution
        cqw = self._detectar_comunidades(self.g_quim_dir, "weight", res, self.seed_base)
        qqw, qqu = self._mod_segura(self.g_quim_dir, cqw, "weight"), self._mod_segura(self.g_quim_dir, self._detectar_comunidades(self.g_quim_dir, None, res, self.seed_base), None)
        ccw = self._detectar_comunidades(self.g_comb_undir, "weight", res, self.seed_base)
        qcw, qcu = self._mod_segura(self.g_comb_undir, ccw, "weight"), self._mod_segura(self.g_comb_undir, self._detectar_comunidades(self.g_comb_undir, None, res, self.seed_base), None)

        gf, ga, nos_a = {}, {}, []
        for no, d in self.g_quim_dir.nodes(data=True):
            gf.setdefault(str(d.get("tipo_funcional", "Desconhecido")), set()).add(no)
            reg = str(d.get("regiao_soma", "N/A")).strip()
            if reg and reg.upper() != "N/A" and reg.lower() != "nan":
                ga.setdefault(reg, set()).add(no); nos_a.append(no)
        pf, pa, sub_qa = list(gf.values()), list(ga.values()), self.g_quim_dir.subgraph(nos_a)
        qfq, qaq = self._mod_segura(self.g_quim_dir, pf, "weight"), self._mod_segura(sub_qa, pa, "weight")
        ret_q, ret_c = self._retencao_intra(self.g_quim_dir, cqw), self._retencao_intra(self.g_comb_undir, ccw)
        mlv, nall = {no: f"C{i}" for i, c in enumerate(cqw) for no in c}, list(self.g_quim_dir.nodes())
        nmif = self._calcular_nmi([mlv[n] for n in nall], [str(self.g_quim_dir.nodes[n].get("tipo_funcional", "Desconhecido")) for n in nall])
        nmia = self._calcular_nmi([mlv[n] for n in nos_a], [str(self.g_quim_dir.nodes[n].get("regiao_soma", "N/A")).strip() for n in nos_a])
        st = resumir_contra_nulo(qqw, null_q)
        df_mod = pd.DataFrame([
            {"Particao": "Louvain (Química)", "Camada": "Quimica", "Q_Ponderado": qqw, "Q_Nao_Ponderado": qqu, "N_Comunidades": len(cqw), "Retencao_Intra_Pct": ret_q, "Nulo_Media": st["nulo_media"], "Nulo_Std": st["nulo_std"], "Z_Score": st["z_score"]},
            {"Particao": "Louvain (Combinada)", "Camada": "Combinada", "Q_Ponderado": qcw, "Q_Nao_Ponderado": qcu, "N_Comunidades": len(ccw), "Retencao_Intra_Pct": ret_c, "Nulo_Media": np.nan, "Nulo_Std": np.nan, "Z_Score": np.nan},
            {"Particao": "Partição Funcional", "Camada": "Quimica", "Q_Ponderado": qfq, "Q_Nao_Ponderado": self._mod_segura(self.g_quim_dir, pf, None), "N_Comunidades": len(pf), "Retencao_Intra_Pct": self._retencao_intra(self.g_quim_dir, pf), "Nulo_Media": np.nan, "Nulo_Std": np.nan, "Z_Score": np.nan},
            {"Particao": "Partição Anatômica", "Camada": "Quimica", "Q_Ponderado": qaq, "Q_Nao_Ponderado": self._mod_segura(sub_qa, pa, None), "N_Comunidades": len(pa), "Retencao_Intra_Pct": self._retencao_intra(sub_qa, pa), "Nulo_Media": np.nan, "Nulo_Std": np.nan, "Z_Score": np.nan},
            {"Particao": "Modelo Nulo (Química)", "Camada": "Quimica", "Q_Ponderado": st["nulo_media"], "Q_Nao_Ponderado": np.nan, "N_Comunidades": np.nan, "Retencao_Intra_Pct": np.nan, "Nulo_Media": st["nulo_media"], "Nulo_Std": st["nulo_std"], "Z_Score": 0.0},
        ])
        return {
            "json": {
                "Q_quimica_louvain_ponderada": qqw, "Q_quimica_louvain_n_comunidades": len(cqw), "Q_quimica_nulo_media": st["nulo_media"], "Q_quimica_nulo_std": st["nulo_std"], "Q_quimica_z_score": st["z_score"],
                "Q_combinada_louvain_ponderada": qcw, "Q_combinada_louvain_n_comunidades": len(ccw), "Q_funcional_quimica": qfq, "Q_anatomica_quimica": qaq,
                "retencao_intra_modulo_quimica_pct": ret_q, "retencao_intra_modulo_combinada_pct": ret_c, "nmi_louvain_vs_funcional": nmif, "nmi_louvain_vs_anatomico": nmia,
            },
            "linhas": [self._linha("3. Modularidade", "Modularidade Louvain Ponderada (Q)", "Quimica", qqw, st), self._linha("3. Modularidade", "Modularidade Louvain Ponderada (Q)", "Combinada", qcw, None), self._linha("3. Modularidade", "Retenção Intra-Módulo (%)", "Quimica", ret_q, None)],
            "df_modularidade": df_mod,
        }

    def _metricas_ciclos_triades(self, nos: List[str], nc2: Sequence[float], nc3: Sequence[float], ntri: Dict[str, List[float]]) -> Dict[str, Any]:
        n = len(nos)
        c2, c3, c2n, c3n = self._contar_ciclos(self.g_quim_dir, nos)
        r2, r3 = (float(2.0 * c2 / (n * (n - 1))) if n >= 2 else 0.0), (float(3.0 * c3 / (n * (n - 1) * (n - 2))) if n >= 3 else 0.0)
        ntrip = float((n * (n - 1) * (n - 2)) / 6.0) if n >= 3 else 1.0
        dfn = pd.DataFrame({"Tipo": [self.g_quim_dir.nodes[no].get("tipo_funcional", "Desconhecido") for no in nos], "RC": [no in self.membros_rich_club for no in nos], "C2": c2n, "C3": c3n})
        l_cls = [{"Grupo": cls, "Categoria": "Classe Funcional", "N_Neuronios": len(s), "C2_Media": float(s["C2"].mean()) if len(s) else 0.0, "C2_Soma": int(s["C2"].sum()), "C3_Media": float(s["C3"].mean()) if len(s) else 0.0, "C3_Soma": int(s["C3"].sum())} for cls in self.ordem_classes for s in (dfn[dfn["Tipo"] == cls],)]
        src, spf = dfn[dfn["RC"]], dfn[~dfn["RC"]]
        for nm, s in (("Núcleo Rich-Club", src), ("Periferia (Demais)", spf)):
            l_cls.append({"Grupo": nm, "Categoria": "Rich-Club vs. Periferia", "N_Neuronios": len(s), "C2_Media": float(s["C2"].mean()) if len(s) else 0.0, "C2_Soma": int(s["C2"].sum()), "C3_Media": float(s["C3"].mean()) if len(s) else 0.0, "C3_Soma": int(s["C3"].sum())})

        censo = nx.triadic_census(self.g_quim_dir)
        tv = {k: int(censo.get(k, 0)) for k in ("030T", "030C", "120C", "210", "300")}
        tf = sum(int(censo.get(k, 0)) for k in ("030T", "030C", "120D", "120U", "120C", "210", "300"))
        tr = tv["030C"] + tv["120C"] + tv["210"] + tv["300"]
        s2, s3 = resumir_contra_nulo(float(c2), nc2), resumir_contra_nulo(float(c3), nc3)
        specs = [("2-Ciclos (C2)", "Retroalimentação Direta", c2, r2, s2), ("3-Ciclos (C3)", "Retroalimentação Cíclica", c3, r3, s3)]
        stri = {}
        for k, rot, tc in (("030T", "030T (Feedforward)", "Feedforward Acíclico"), ("030C", "030C (3-Ciclo Puro)", "Retroalimentação Cíclica"), ("120C", "120C (Ciclo + Díade)", "Retroalimentação Cíclica"), ("210", "210 (2 Díades Recíprocas)", "Retroalimentação Intensa"), ("300", "300 (Clique Bidirecional)", "Retroalimentação Máxima")):
            stri[k] = resumir_contra_nulo(float(tv[k]), ntri.get(k, []))
            specs.append((rot, tc, tv[k], float(tv[k] / ntrip), stri[k]))
        df_m = pd.DataFrame([{"Estrutura": e, "Tipo_Circuito": tc, "Contagem_Real": int(cr), "Densidade_Real": float(dr), "Nulo_Media": st["nulo_media"], "Nulo_Std": st["nulo_std"], "Razao_Real_Nulo": st["razao"], "Z_Score": st["z_score"]} for e, tc, cr, dr, st in specs])
        return {
            "json": {
                "n_ciclos_2": c2, "densidade_ciclos_2": r2, "z_score_ciclos_2": s2["z_score"], "n_ciclos_3": c3, "densidade_ciclos_3": r3, "z_score_ciclos_3": s3["z_score"],
                "triade_030T_feedforward": tv["030T"], "triade_030C_feedback": tv["030C"], "triade_120C": tv["120C"], "triade_210": tv["210"], "triade_300": tv["300"],
                "fracao_triades_recorrentes_pct": (tr / tf) * 100.0 if tf > 0 else 0.0, "razao_030C_030T": float(tv["030C"] / max(tv["030T"], 1)),
                "media_c2_rich_club": float(src["C2"].mean()) if len(src) else 0.0, "media_c2_periferia": float(spf["C2"].mean()) if len(spf) else 0.0,
                "media_c3_rich_club": float(src["C3"].mean()) if len(src) else 0.0, "media_c3_periferia": float(spf["C3"].mean()) if len(spf) else 0.0,
            },
            "linhas": [self._linha("4. Ciclos Curtos", "2-Ciclos Direcionados (N_C2)", "Quimica", float(c2), s2), self._linha("4. Ciclos Curtos", "3-Ciclos Direcionados (N_C3)", "Quimica", float(c3), s3), self._linha("4. Ciclos Curtos", "Tríades 210 (2 Díades Recíprocas)", "Quimica", float(tv["210"]), stri["210"]), self._linha("4. Ciclos Curtos", "Tríades 300 (Clique Bidirecional)", "Quimica", float(tv["300"]), stri["300"])],
            "df_ciclos_motifs": df_m, "df_ciclos_por_classe": pd.DataFrame(l_cls),
        }

    def executar_analise_completa(self) -> Dict[str, Any]:
        nos = sorted(self.g_quim_dir.nodes())
        aq = [(u, v, float(d.get("weight", 1.0))) for u, v, d in self.g_quim_dir.edges(data=True)]
        ac = [(u, v, float(d.get("weight", 1.0))) for u, v, d in self.g_comb_dir.edges(data=True)]
        sq, sc = max(1500, self.swaps_fator * len(aq)), max(1500, self.swaps_fator * len(ac))
        nr, nrw, nrc, nlscc, nq, nc2, nc3 = [], [], [], [], [], [], []
        chaves = ("030T", "030C", "120C", "210", "300")
        ntri: Dict[str, List[float]] = {k: [] for k in chaves}

        for idx in range(self.n_runs_null):
            si = self.seed_base + idx
            gn = gerar_nulo_direcionado(aq, nos, sq, np.random.default_rng(si))
            rn, rwn = self._reciprocidade_grafo(gn); nr.append(rn); nrw.append(rwn)
            nrc.append(self._reciprocidade_grafo(gerar_nulo_direcionado(ac, nos, sc, np.random.default_rng(si + 10000)))[0])
            nlscc.append(float(max((len(c) for c in nx.strongly_connected_components(gn)), default=0)))
            nq.append(self._mod_segura(gn, self._detectar_comunidades(gn, "weight", self.louvain_resolution, si), "weight"))
            c2n, c3n, _, _ = self._contar_ciclos(gn, nos); nc2.append(float(c2n)); nc3.append(float(c3n))
            cn = nx.triadic_census(gn)
            for k in chaves:
                ntri[k].append(float(cn.get(k, 0)))

        rec, scc, mod, cic = self._metricas_reciprocidade(nr, nrw, nrc), self._metricas_scc_bowtie(nlscc), self._metricas_modularidade(nq), self._metricas_ciclos_triades(nos, nc2, nc3, ntri)
        resumo = sanitizar_json({
            "metadados": {"N": self.g_quim_dir.number_of_nodes(), "M_quimica": self.g_quim_dir.number_of_edges(), "M_combinada_dir": self.g_comb_dir.number_of_edges(), "n_runs_null": self.n_runs_null},
            "reciprocidade": rec["json"], "scc_bowtie": scc["json"], "modularidade": mod["json"], "ciclos_curtos": cic["json"],
        })
        return {
            "df_resumo": pd.DataFrame(rec["linhas"] + scc["linhas"] + mod["linhas"] + cic["linhas"]),
            "mat_reciprocidade": rec["mat_reciprocidade"], "df_bowtie_classes": scc["df_bowtie_classes"],
            "df_modularidade": mod["df_modularidade"], "df_ciclos_motifs": cic["df_ciclos_motifs"],
            "df_ciclos_por_classe": cic["df_ciclos_por_classe"], "resumo_json": resumo,
        }
