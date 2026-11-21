"""Exportação do dashboard interativo multi-layout em HTML (Pyvis + template externo)."""

import base64
import html
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from config import CENTROS_GRUPOS_FUNCIONAIS, ORDEM_CLASSES_PADRAO, PALETA_CORES_FUNCIONAIS, PYVIS_PHYSICS_OPTIONS
from plotting_utils import resolver_caminho_saida

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "dashboard_overlay.html")


def _imagem_para_base64(caminho: str) -> str:
    if not caminho or not os.path.exists(caminho):
        return ""
    with open(caminho, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _formatar_celula(col: str, val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    if isinstance(val, bool):
        return "Sim" if val else "Não"
    if isinstance(val, (int, np.integer)):
        return html.escape(str(int(val)))
    if isinstance(val, (float, np.floating)):
        if col in ("PageRank", "Intermediacao", "Score_Chefe", "Score_Global", "Proximidade"):
            return html.escape(f"{float(val):.4f}")
        return html.escape(f"{float(val) * 100:.1f}%" if str(col).startswith("Pct_") else f"{float(val):.2f}")
    return html.escape(str(val))


def _montar_tabela_html(df: pd.DataFrame, cor_por_tipo: Optional[Dict[str, str]] = None) -> str:
    if df is None or df.empty:
        return ""
    cols = list(df.columns)
    bordas = {
        1: ' style="border-left:5px solid #FFD700; font-weight:bold; background-color:rgba(255,215,0,0.08);"',
        2: ' style="border-left:5px solid #C0C0C0; font-weight:bold; background-color:rgba(192,192,192,0.08);"',
        3: ' style="border-left:5px solid #CD7F32; font-weight:bold; background-color:rgba(205,127,50,0.08);"',
    }
    linhas = []
    for _, ln in df.iterrows():
        est0 = ""
        if "Posicao_Global" in cols:
            try:
                est0 = bordas.get(int(ln["Posicao_Global"]), "")
            except (ValueError, TypeError):
                pass
        elif cor_por_tipo:
            eh_c = bool(ln["E_Chefe"]) if "E_Chefe" in cols else (int(ln["Posicao_No_Tipo"]) == 1 if "Posicao_No_Tipo" in cols else True)
            cor = cor_por_tipo.get(str(ln.get("Tipo", "")))
            if eh_c and cor:
                est0 = f' style="border-left:4px solid {html.escape(cor)};"'
        linhas.append("<tr>" + "".join(f"<td{est0 if i == 0 else ''}>{_formatar_celula(c, ln[c])}</td>" for i, c in enumerate(cols)) + "</tr>")
    cab = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    return f'<div style="overflow-x:auto;"><table class="tabela-ranking"><thead><tr>{cab}</tr></thead><tbody>{"".join(linhas)}</tbody></table></div>'


def _calcular_forca_por_camada(graph: nx.MultiDiGraph) -> Tuple[Dict[str, float], Dict[str, float]]:
    fq, fe = {n: 0.0 for n in graph.nodes()}, {n: 0.0 for n in graph.nodes()}
    for u, v, k, d in graph.edges(keys=True, data=True):
        w = float(d.get("weight", 1.0))
        alvo = fe if d.get("synapse_type", k) == "Eletrica" else fq
        alvo[u], alvo[v] = alvo.get(u, 0.0) + w, alvo.get(v, 0.0) + w
    return fq, fe


def _calcular_todos_layouts(graph: nx.MultiDiGraph, rc_set: Set[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    layouts: Dict[str, Dict[str, Dict[str, float]]] = {k: {} for k in ("clusters", "anatomico", "hierarquico", "rich_club", "multiplex")}
    ang = np.pi * (3.0 - np.sqrt(5.0))
    por_tipo: Dict[str, List[Tuple[str, dict]]] = {}
    for n, d in graph.nodes(data=True):
        por_tipo.setdefault(d.get("tipo_funcional", "Desconhecido"), []).append((n, d))
    for lista in por_tipo.values():
        lista.sort(key=lambda x: graph.degree(x[0]), reverse=True)

    for tipo, lista in por_tipo.items():
        cx, cy = CENTROS_GRUPOS_FUNCIONAIS.get(tipo, (0, 0))
        for i, (n, _) in enumerate(lista):
            r, th = 52.0 * np.sqrt(i), i * ang
            layouts["clusters"][n] = {"x": round(float(cx + r * np.cos(th)), 2), "y": round(float(cy + r * np.sin(th)), 2)}

    fy = {"Sensorial": -430.0, "Interneurônio": 20.0, "Interneurônio (Comando)": 230.0, "Motor": 450.0, "Desconhecido": 600.0}
    ord_p = sorted(graph.nodes(data=True), key=lambda x: (float(x[1].get("soma_pos") or 0.5), x[0]))
    tot, cf = max(len(ord_p) - 1, 1), {k: 0 for k in fy}
    for ri, (n, d) in enumerate(ord_p):
        sp = min(max(float(d.get("soma_pos") or 0.5), 0.0), 1.0)
        tp = d.get("tipo_funcional", "Desconhecido")
        il = cf.get(tp, 0); cf[tp] = il + 1
        layouts["anatomico"][n] = {"x": round(float(-1350.0 + (0.45 * sp + 0.55 * (ri / tot)) * 2700.0), 2), "y": round(float(fy.get(tp, 0.0) + ((il % 5) - 2) * 34.0), 2)}

    cols = {"Sensorial": (-680.0, 3), "Interneurônio": (0.0, 3), "Interneurônio (Comando)": (620.0, 1), "Motor": (1200.0, 4), "Desconhecido": (1500.0, 1)}
    for tp, lista in por_tipo.items():
        xc, ns = cols.get(tp, (0.0, 2))
        lt = int(np.ceil(len(lista) / ns)); ey = 54.0 if lt <= 30 else 44.0
        for i, (n, _) in enumerate(lista):
            layouts["hierarquico"][n] = {"x": round(float(xc + ((i % ns) - (ns - 1) / 2.0) * 92.0), 2), "y": round(float(((i // ns) - (lt - 1) / 2.0) * ey), 2)}

    pg = sorted(graph.nodes(data=True), key=lambda x: graph.degree(x[0]), reverse=True)
    elite = [x for x in pg if x[0] in rc_set] or pg[:10]
    rest = [x for x in pg if x[0] not in {e[0] for e in elite}]
    c1, c2 = max(int(len(rest) * 0.22), 1), max(int(len(rest) * 0.58), 2)
    oset = {c: i for i, c in enumerate(ORDEM_CLASSES_PADRAO)}
    a2 = sorted(rest[c1:c2], key=lambda x: (oset.get(x[1].get("tipo_funcional", ""), 99), -graph.degree(x[0])))
    a3 = sorted(rest[c2:], key=lambda x: (oset.get(x[1].get("tipo_funcional", ""), 99), -graph.degree(x[0])))
    for la, rb, alt in ((elite, 115.0, 0.0), (rest[:c1], 330.0, 0.0), (a2, 580.0, 28.0), (a3, 860.0, 38.0)):
        q = max(len(la), 1)
        for i, (n, _) in enumerate(la):
            a, re = (2.0 * np.pi * i) / q - (np.pi / 2.0), rb + (i % 2) * alt
            layouts["rich_club"][n] = {"x": round(float(re * np.cos(a)), 2), "y": round(float(re * np.sin(a)), 2)}

    fq, fe = _calcular_forca_por_camada(graph)
    fx = {"Sensorial": -980.0, "Interneurônio": -320.0, "Interneurônio (Comando)": 220.0, "Motor": 820.0, "Desconhecido": 1250.0}
    for tp, lista in por_tipo.items():
        xc = fx.get(tp, 0.0)
        for i, (n, _) in enumerate(lista):
            s = fq.get(n, 0.0) + fe.get(n, 0.0)
            pol = (fe.get(n, 0.0) - fq.get(n, 0.0)) / s if s > 0 else 0.0
            layouts["multiplex"][n] = {"x": round(float(xc + ((i % 5) - 2) * 72.0 + ((i // 5) % 2) * 22.0), 2), "y": round(float(pol * 620.0 + ((i % 3) - 1) * 24.0), 2)}
    return layouts


def exportar_html_interativo(
    graph: nx.MultiDiGraph, output_dir: str, arquivo_saida: str = "celegans_rede_interativa.html",
    notebook: bool = False, influentes: Optional[Dict[str, float]] = None, top_globais: Optional[Dict[str, int]] = None,
    top_detalhes: Optional[Dict[str, Dict[str, Any]]] = None, membros_rich_club: Optional[Set[str]] = None,
    graficos_png: Optional[Dict[str, str]] = None, tabelas_ranking: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    caminho = resolver_caminho_saida(output_dir, arquivo_saida)
    rc_set = set(membros_rich_club or set())
    layouts = _calcular_todos_layouts(graph, rc_set)
    fq_m, fe_m = _calcular_forca_por_camada(graph)

    net = Network(notebook=notebook, height="100vh", width="100vw", directed=True, bgcolor="#1a1a1a", font_color="white")
    net.set_options(PYVIS_PHYSICS_OPTIONS)
    g_vis = nx.DiGraph()

    por_tipo: Dict[str, List[Tuple[str, dict]]] = {}
    for n, d in graph.nodes(data=True):
        por_tipo.setdefault(d.get("tipo_funcional", "Desconhecido"), []).append((n, d))

    med, bordas = {1: "🥇", 2: "🥈", 3: "🥉"}, {1: "#FFD700", 2: "#C0C0C0", 3: "#CD7F32"}
    for tp, lista in por_tipo.items():
        lista.sort(key=lambda x: graph.degree(x[0]), reverse=True)
        cx, cy = CENTROS_GRUPOS_FUNCIONAIS.get(tp, (0, 0))
        anc = f"__ancora_{tp}__"
        g_vis.add_node(anc, label=" ", x=float(cx), y=float(cy), fixed={"x": True, "y": True}, physics=True, size=0.1, color="rgba(0,0,0,0)", borderWidth=0, chosen=False)
        for n, d in lista:
            ci, gr, eh_rc = layouts["clusters"][n], graph.degree(n), n in rc_set
            tam, rk, eh_inf = 10 + (gr / 5.0), (top_globais.get(n) if top_globais else None), (influentes is not None and n in influentes)
            sp = d.get("soma_pos", 0.0)
            tt = [f"Neurônio: {n}", f"Tipo: {tp}", f"Grau: {gr} (Q:{int(fq_m.get(n, 0))} | E:{int(fe_m.get(n, 0))})", f"Soma: {float(sp):.2f}" if sp is not None else "Soma: N/A", f"Rich-Club: {'Sim' if eh_rc else 'Não'}"]
            cor_b = PALETA_CORES_FUNCIONAIS.get(tp, "grey")
            if rk in med:
                tt.insert(0, f"🏆 TOP {rk} GLOBAL")
                if top_detalhes and n in top_detalhes:
                    tt.append(f"Score Global: {float(top_detalhes[n].get('Score_Global', 0)):.4f}")
                rot, tf, bw, shp, cf = f"{med[rk]} {n}", tam * 2.2, 6.0, "star", {"background": cor_b, "border": bordas[rk], "highlight": {"background": cor_b, "border": "#FFFFFF"}}
            elif eh_inf:
                tt.append(f"Chefe: {influentes[n]:.3f}")
                rot, tf, bw, shp, cf = f"★ {n}", tam * 1.5, 4.0, "star", cor_b
            else:
                rot, tf, bw, shp, cf = n, tam, (4.0 if eh_rc else 1.5), "dot", cor_b
            g_vis.add_node(n, label=rot, title="\n".join(tt), color=cf, size=tf, shape=shp, borderWidth=bw, x=float(ci["x"]), y=float(ci["y"]), physics=True)
            g_vis.add_edge(anc, n, physics=True, color="rgba(0,0,0,0)", width=0.01, hoverWidth=0, selectionWidth=0, arrows={"to": {"enabled": False}})

    for u, v, k, d in graph.edges(keys=True, data=True):
        w, cam = int(d.get("weight", 1)), d.get("synapse_type", k)
        pq, pe = (w, 0) if cam == "Quimica" else (0, w)
        du = min(max(float(graph.nodes[u].get("soma_pos") if graph.nodes[u].get("soma_pos") is not None else 0.5), 0.0), 1.0)
        dv = min(max(float(graph.nodes[v].get("soma_pos") if graph.nodes[v].get("soma_pos") is not None else 0.5), 0.0), 1.0)
        dist, custo = abs(du - dv), float(w) * abs(du - dv)
        if g_vis.has_edge(u, v):
            e = g_vis[u][v]
            e["peso_acumulado"] += w; e["peso_quimico"] += pq; e["peso_eletrico"] += pe
            e["custo_cabeamento"] += custo; e["custo_quimico"] += float(pq) * dist; e["custo_eletrico"] += float(pe) * dist
            e["width"] = float(0.7 + np.log1p(e["peso_acumulado"]) * 0.85); e["title"] += f" | {cam}: {w}"
        else:
            g_vis.add_edge(u, v, peso_acumulado=w, peso_quimico=pq, peso_eletrico=pe, custo_cabeamento=custo, custo_quimico=float(pq) * dist, custo_eletrico=float(pe) * dist, distancia_soma=dist, width=float(0.7 + np.log1p(w) * 0.85), title=f"{u} → {v} | {cam}: {w}", color="rgba(160, 160, 160, 0.22)", physics=False)

    custo_max = max((g_vis[u][v].get("custo_cabeamento", 0.0) for u, v in g_vis.edges()), default=0.0)
    net.from_nx(g_vis); net.save_graph(caminho)

    op_g, op_t, cont, ids = "", "", "", []
    for i, (tit, p_img) in enumerate((graficos_png or {}).items()):
        b64 = _imagem_para_base64(p_img)
        if not b64:
            continue
        ida, prim, te, na = f"aba_grafico_{i}", len(ids) == 0, html.escape(str(tit)), html.escape(os.path.basename(p_img) or f"g_{i}.png")
        ids.append(ida)
        op_g += f'<option value="{ida}" data-tipo="grafico" data-arquivo="{na}"{" selected" if prim else ""}>📈 {te}</option>'
        cont += f'<div id="{ida}" class="conteudo-aba" data-tipo="grafico" data-titulo="{te}" data-arquivo="{na}" style="display:{"block" if prim else "none"};"><div class="container-imagem-zoom"><img class="img-metrica-interativa" src="data:image/png;base64,{b64}" alt="{te}" onclick="alternarCliqueZoomImagem(this)" /></div></div>'
    for i, (tit, df_t) in enumerate((tabelas_ranking or {}).items()):
        corpo = _montar_tabela_html(df_t, cor_por_tipo=PALETA_CORES_FUNCIONAIS)
        if not corpo:
            continue
        ida, prim, te = f"aba_tabela_{i}", len(ids) == 0, html.escape(str(tit))
        ids.append(ida)
        op_t += f'<option value="{ida}" data-tipo="tabela"{" selected" if prim else ""}>{te}</option>'
        cont += f'<div id="{ida}" class="conteudo-aba" data-tipo="tabela" data-titulo="{te}" style="display:{"block" if prim else "none"};">{corpo}</div>'

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        tpl = f.read()
    for ch, val in {
        "{{BTN_MODAL}}": '<button class="btn-topo btn-destaque" onclick="alternarModalMetricas()">📊 Gráficos de Métricas</button>' if ids else "",
        "{{OPCOES_SELECT_MODAL}}": (f'<optgroup label="Gráficos e Imagens (PNG)">{op_g}</optgroup>' if op_g else "") + (f'<optgroup label="Tabelas e Rankings">{op_t}</optgroup>' if op_t else ""),
        "{{TOTAL_MODAL}}": str(max(len(ids), 1)), "{{CONTEUDOS_ABAS}}": cont,
        "{{LAYOUTS_JSON}}": json.dumps(layouts), "{{RC_JSON}}": json.dumps(list(rc_set)),
        "{{CUSTO_MAXIMO}}": f"{custo_max:.6f}", "{{IDS_MODAL_JSON}}": json.dumps(ids),
    }.items():
        tpl = tpl.replace(ch, val)
    with open(caminho, "r", encoding="utf-8") as f:
        html_raw = f.read()
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(html_raw.replace("</body>", tpl))
    return caminho
