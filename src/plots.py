"""Figuras estáticas (PNG, 300 DPI) do conectoma C. elegans em Matplotlib."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from config import (
    ORDEM_CLASSES_PADRAO, PALETA_CENARIOS_DIFUSAO, PALETA_CORES_FUNCIONAIS,
    PALETA_ESTRATEGIAS_NAVEGACAO, PALETA_MODELOS_CAUDA, PALETA_MODELOS_SINTETICOS,
    PALETA_REGIMES_ROBUSTEZ,
)
from plotting_utils import (
    estilizar_eixo, plotar_barras_anotadas, plotar_heatmap_matriz,
    resolver_caminho_saida, salvar_figura,
)


def _rotular(ax, titulo: str, xlabel: str = "", ylabel: str = "", grid: str = "both") -> None:
    ax.set_title(titulo, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontweight="bold")
    estilizar_eixo(ax, grid)


def plotar_matrizes_fluxo(output_dir: str, mat_quimica: pd.DataFrame, mat_eletrica: pd.DataFrame, ordem_classes: List[str], nome_arquivo: str = "matrizes_fluxo.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    with salvar_figura(caminho, 1, 2, figsize=(16, 6.5)) as (fig, axes):
        for ax, (mat, tit, cmap) in zip(axes, [(mat_quimica, "Camada Química (Origem → Destino)", "YlOrRd"), (mat_eletrica, "Camada Elétrica / Gap Junctions", "PuBu")]):
            pct = mat.div(mat.sum(axis=1).replace(0, 1), axis=0) * 100.0
            plotar_heatmap_matriz(fig, ax, mat.values, ordem_classes, ordem_classes, cmap=cmap, fmt_celula=lambda i, j, v, p=pct.values: f"{int(v)}\n({p[i, j]:.1f}%)", cbar_label="Volume Sináptico", rotation=28)
            ax.set_xlabel("Destino", fontweight="bold"); ax.set_ylabel("Origem", fontweight="bold"); ax.set_title(tit, fontweight="bold")
    return caminho


def plotar_rich_club_ponderado(output_dir: str, df_rc: pd.DataFrame, k_otimo: int, membros_elite: List[str], camada: str = "Combinada", nome_arquivo: str = "rich_club_ponderado.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    k, pw, pn, rw = df_rc["k"].values, df_rc["Phi_w"].values, df_rc["Phi_w_Null"].values, df_rc["Rho_w"].values
    with salvar_figura(caminho, 1, 2, figsize=(16, 6.0)) as (_, (ax1, ax2)):
        ax1.plot(k, pw, color="#EA4335", lw=2.3, marker="o", ms=3.8, label=r"Real $\phi^{w}(k)$")
        ax1.plot(k, pn, color="#555", lw=1.8, ls="--", label=r"Nulo $\phi^{w}_{\mathrm{null}}(k)$")
        ax1.axvline(k_otimo, color="#4285F4", ls=":", lw=1.7, label=rf"$k > {k_otimo}$")
        ax1.legend(loc="upper left", fontsize=9)
        _rotular(ax1, f"Rich-Club Ponderado ({camada})", "Grau ($k$)", r"$\phi^{w}(k)$")

        ax2.plot(k, rw, color="#34A853", lw=2.3, marker="s", ms=3.8, label=r"$\rho^{w}(k)$")
        ax2.axhline(1.0, color="#222", ls="--", lw=1.4)
        ax2.axvline(k_otimo, color="#4285F4", ls=":", lw=1.7, label=f"Elite ($k>{k_otimo}$, $n={len(membros_elite)}$)")
        ax2.fill_between(k, 1.0, rw, where=(rw > 1.0), color="#34A853", alpha=0.18)
        ax2.legend(loc="lower left", fontsize=9)
        _rotular(ax2, f"Rich-Club Normalizado ({camada})", "Grau ($k$)", r"$\rho^{w}(k)$")
    return caminho


def plotar_comparacao_redes_sinteticas(output_dir: str, df_comparacao: pd.DataFrame, diagnostico: Dict[str, Any], nome_arquivo: str = "comparacao_redes_sinteticas.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    modelos = list(df_comparacao["Modelo"].values)
    curtos = {"C. elegans (Real)": "Real", "Erdős-Rényi (ER)": "ER", "Watts-Strogatz (WS)": "WS", "Barabási-Albert (BA)": "BA"}
    x, cores = np.arange(len(modelos)), [PALETA_MODELOS_SINTETICOS.get(m, "#666") for m in modelos]
    paineis = [
        ("Agrupamento_C_Media", "Agrupamento_C_Std", r"Agrupamento ($C$)", r"$C$", "{:.3f}", False),
        ("Caminho_Medio_L_Media", "Caminho_Medio_L_Std", r"Caminho Médio ($L$)", r"$L$", "{:.3f}", False),
        ("Eficiencia_Global_Media", "Eficiencia_Global_Std", r"Eficiência ($E_{\mathrm{glob}}$)", r"$E_{\mathrm{glob}}$", "{:.3f}", False),
        ("Assortatividade_r_Media", "Assortatividade_r_Std", r"Assortatividade ($r$)", r"$r$", "{:+.3f}", True),
        ("Heterogeneidade_Media", "Heterogeneidade_Std", r"Heterogeneidade ($\kappa$)", r"$\kappa$", "{:.2f}", False),
    ]
    with salvar_figura(caminho, 2, 3, figsize=(16, 9.0), tight_rect=(0, 0, 1, 0.95)) as (fig, axes):
        flat = axes.ravel()
        for ax, (cm, cs, tit, yl, fmt, zero) in zip(flat[:5], paineis):
            meds, stds = df_comparacao[cm].astype(float).values, df_comparacao[cs].astype(float).values
            plotar_barras_anotadas(ax, x, meds, yerr=stds, cores=cores, fmt=fmt, capsize=4)
            if zero:
                ax.axhline(0.0, color="#222", ls="--", lw=1.0)
            yt, yb = float(np.max(meds + stds)), float(np.min(meds - stds))
            amp = max(yt - yb, 0.05)
            ax.set_ylim((min(yb, 0) - 0.25 * amp, max(yt, 0) + 0.25 * amp) if zero else (0, yt + 0.22 * max(yt, amp)))
            ax.set_xticks(x); ax.set_xticklabels([curtos.get(m, m) for m in modelos], fontweight="bold")
            _rotular(ax, tit, ylabel=yl, grid="y")
        flat[5].axis("off")
        txt = "\n".join([
            rf"$N={diagnostico['N']}$, $M={diagnostico['M']}$, $\langle k \rangle={diagnostico['k_medio']:.2f}$",
            rf"$C/C_{{\mathrm{{ER}}}}={diagnostico['razao_C_ER']:.2f}\times$ | $L/L_{{\mathrm{{ER}}}}={diagnostico['razao_L_ER']:.2f}\times$",
            rf"$\sigma_{{\mathrm{{SW}}}}={diagnostico['sigma_small_world']:.2f}$ | $\omega_{{\mathrm{{SW}}}}={diagnostico['omega_small_world']:+.3f}$",
            rf"$\kappa_{{\mathrm{{real}}}}={diagnostico['heterogeneidade_real']:.2f}$ (ER={diagnostico['heterogeneidade_ER']:.2f}, BA={diagnostico['heterogeneidade_BA']:.2f})",
            f"Classe: {diagnostico['veredito_classificacao']}",
        ])
        flat[5].text(0.05, 0.85, txt, transform=flat[5].transAxes, fontsize=10.5, va="top", bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8f9fa", "edgecolor": "#c5ccd3"})
        fig.suptitle("Conectoma C. elegans vs. Modelos Sintéticos (ER, WS, BA)", fontsize=13.5, fontweight="bold")
    return caminho


def plotar_distribuicao_grau_comparativa(output_dir: str, graus_por_modelo: Dict[str, np.ndarray], diagnostico: Dict[str, Any], nome_arquivo: str = "distribuicao_grau_comparativa.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    mr = "C. elegans (Real)"
    gr = np.asarray(graus_por_modelo[mr], dtype=int)
    gmax = int(max(np.max(v) for v in graus_por_modelo.values() if len(v)))
    with salvar_figura(caminho, 1, 2, figsize=(16, 6.0)) as (_, (ax1, ax2)):
        ax1.hist(gr, bins=np.arange(0, gmax + 3, 2) - 0.5, density=True, color=PALETA_MODELOS_SINTETICOS[mr], alpha=0.35, edgecolor=PALETA_MODELOS_SINTETICOS[mr], label=mr)
        for nm, ls in (("Erdős-Rényi (ER)", "--"), ("Watts-Strogatz (WS)", "-."), ("Barabási-Albert (BA)", "-")):
            if nm in graus_por_modelo:
                kv, cnt = np.unique(np.asarray(graus_por_modelo[nm], dtype=int), return_counts=True)
                ax1.plot(kv, cnt / float(len(graus_por_modelo[nm])), color=PALETA_MODELOS_SINTETICOS[nm], lw=2.0, ls=ls, label=nm)
        ax1.axvline(float(diagnostico["k_medio"]), color="#222", ls=":", lw=1.6, label=rf"$\langle k \rangle={float(diagnostico['k_medio']):.2f}$")
        ax1.set_xlim(left=0); ax1.legend(loc="upper right", fontsize=9)
        _rotular(ax1, "Distribuição de Grau (PMF)", r"Grau ($k$)", r"$P(K=k)$")

        for nm, ls, mk in (("Erdős-Rényi (ER)", "--", None), ("Watts-Strogatz (WS)", "-.", None), ("Barabási-Albert (BA)", "-", None), (mr, "-", "o")):
            if nm in graus_por_modelo:
                gp = np.sort(np.asarray(graus_por_modelo[nm], dtype=int)); gp = gp[gp > 0]; ku = np.unique(gp)
                ax2.loglog(ku, (len(gp) - np.searchsorted(gp, ku, side="left")) / float(len(gp)), color=PALETA_MODELOS_SINTETICOS[nm], lw=2.1 if nm == mr else 1.8, ls=ls, marker=mk, ms=4, label=nm)
        km = int(diagnostico["k_min_cauda"])
        ax2.axvline(km, color="#444", ls=":", lw=1.6, label=rf"$k_{{\min}}={km}$")
        ax2.legend(loc="upper right", fontsize=9)
        _rotular(ax2, r"CCDF $P(K \geq k)$ (Log-Log)", r"Grau ($k$)", r"$P(K \geq k)$")
    return caminho


def plotar_evidencias_small_world(output_dir: str, ws_sweep_df: pd.DataFrame, resumo_sw: Dict[str, Any], nome_arquivo: str = "evidencias_small_world.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    dfp = ws_sweep_df[ws_sweep_df["p_rewiring"] > 0.0]
    pv, cn, cs, ln, ls_v = (dfp[c].astype(float).values for c in ("p_rewiring", "C_norm_media", "C_norm_std", "L_norm_media", "L_norm_std"))
    peq = float(resumo_sw["p_equiv_ws"])
    with salvar_figura(caminho, 1, 2, figsize=(16, 6.0)) as (_, (ax1, ax2)):
        ax1.axvspan(0.005, 0.25, color="#34A853", alpha=0.10, label="Regime Small-World")
        for yv, ys, cor, mk, lb in ((cn, cs, "#34A853", "o", r"$C(p)/C(0)$"), (ln, ls_v, "#4285F4", "s", r"$L(p)/L(0)$")):
            ax1.plot(pv, yv, color=cor, lw=2.2, marker=mk, ms=4, label=lb)
            ax1.fill_between(pv, np.clip(yv - ys, 0, 1.2), np.clip(yv + ys, 0, 1.2), color=cor, alpha=0.16)
        ax1.axvline(peq, color="#EA4335", ls=":", lw=1.8, label=rf"$p_{{\mathrm{{equiv}}}}\approx{peq:.3f}$")
        ax1.scatter([peq, peq], [resumo_sw["C_real_norm_latt"], resumo_sw["L_real_norm_latt"]], color=["#EA4335", "#FBBC05"], marker="*", s=180, edgecolor="#111", zorder=6)
        ax1.set_xscale("log"); ax1.set_ylim(-0.03, 1.12); ax1.legend(loc="lower left", fontsize=8.8)
        _rotular(ax1, "Transição de Fase de Watts–Strogatz", r"Religação ($p$)", "Razão pelo Lattice")

        sig, om, phi = float(resumo_sw["sigma"]), float(resumo_sw["omega"]), float(resumo_sw["phi_swp"])
        vals, y = [phi, max(0.0, 1.0 - abs(om)), sig], np.arange(3)
        ax2.barh(y, vals, height=0.52, color=["#4285F4", "#FBBC05", "#34A853"], edgecolor="#222")
        for i, lim in enumerate((0.60, 0.50, 1.00)):
            ax2.plot([lim, lim], [i - 0.32, i + 0.32], color="#EA4335", ls="--", lw=2.0, label="Limiar SW" if i == 0 else None)
        for i, (v, t) in enumerate(zip(vals, [rf"$\phi_{{\mathrm{{SWP}}}}={phi:.3f}$", rf"$\omega={om:+.3f}$", rf"$\sigma={sig:.2f}$"])):
            ax2.text(v + 0.12, i, t, va="center", fontweight="bold", fontsize=9.5)
        ax2.set_yticks(y); ax2.set_yticklabels([r"$\phi_{\mathrm{SWP}}$", r"$1-|\omega|$", r"$\sigma_{\mathrm{SW}}$"], fontweight="bold")
        ax2.set_xlim(0, max(max(vals) * 1.38, 6.0)); ax2.legend(loc="upper right", fontsize=9)
        _rotular(ax2, rf"Indicadores Small-World ($Z_C={float(resumo_sw.get('z_score_C_vs_ER', 0)):.1f}$)", "Magnitude", grid="x")
    return caminho


def plotar_evidencias_scale_free(output_dir: str, modelos_cauda_df: pd.DataFrame, resumo_sf: Dict[str, Any], nome_arquivo: str = "evidencias_scale_free.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    cur = resumo_sf["curvas_ccdf"]
    kv, c_emp = np.asarray(cur["k_vals"], dtype=float), np.asarray(cur["ccdf_empirica"], dtype=float)
    with salvar_figura(caminho, 1, 2, figsize=(16, 6.0)) as (_, (ax1, ax2)):
        ax1.loglog(kv, c_emp, color=PALETA_MODELOS_CAUDA["Empírico (C. elegans)"], marker="o", ms=5, ls="None", label="Empírico")
        for key, nm, ls in (("ccdf_pl", "Lei de Potência (PL)", "-"), ("ccdf_tpl", "PL Truncada (TPL)", "--"), ("ccdf_ln", "Log-Normal (LN)", "-."), ("ccdf_exp", "Exponencial (EXP)", ":")):
            ax1.loglog(kv, np.asarray(cur[key], dtype=float), color=PALETA_MODELOS_CAUDA[nm], lw=2.1, ls=ls, label=nm)
        cp = np.asarray(cur["ccdf_poisson"], dtype=float)
        if np.any(cp >= 1e-4):
            ax1.loglog(kv[cp >= 1e-4], cp[cp >= 1e-4], color=PALETA_MODELOS_CAUDA["Poisson / ER"], lw=1.7, ls="--", label="Poisson / ER")
        km = int(resumo_sf["k_min"])
        ax1.axvline(km, color="#444", ls=":", lw=1.6, label=rf"$k_{{\min}}^*={km}$ ($n={resumo_sf['n_tail']}$)")
        ax1.set_ylim(max(1e-3, float(np.min(c_emp)) * 0.45), 1.35); ax1.legend(loc="upper right", fontsize=8.5)
        _rotular(ax1, rf"Cauda ($\hat{{\gamma}}={resumo_sf['gamma_pl']:.2f}$, $p_{{\mathrm{{KS}}}}={resumo_sf['p_valor_ks_bootstrap']:.2f}$)", r"Grau ($k$)", r"$P(K \geq k \mid K \geq k_{\min}^*)$")

        dfo = modelos_cauda_df.sort_values("Delta_AIC", ascending=True).reset_index(drop=True)
        daic, y = dfo["Delta_AIC"].astype(float).values, np.arange(len(dfo))
        ax2.barh(y, daic, height=0.52, color=[PALETA_MODELOS_CAUDA.get(m, "#666") for m in dfo["Modelo"]], edgecolor="#222")
        mx = max(float(np.max(daic)), 4.0); ax2.set_xlim(0, mx * 1.65)
        for i, row in dfo.iterrows():
            w = float(row["Delta_AIC"])
            txt = rf"$\Delta\mathrm{{AIC}}={w:.2f}$" if row["Modelo"] == "Lei de Potência (PL)" else rf"$\Delta\mathrm{{AIC}}={w:.2f}$ | $R={float(row['Vuong_R_vs_PL']):+.2f}$ ($p={float(row['Vuong_p_vs_PL']):.3f}$)"
            ax2.text(w + 0.02 * mx, i, txt, va="center", fontsize=8.8, fontweight="bold")
        ax2.set_yticks(y); ax2.set_yticklabels(dfo["Modelo"], fontweight="bold"); ax2.invert_yaxis()
        _rotular(ax2, r"Seleção de Modelos ($\Delta \mathrm{AIC}$ e Vuong)", r"$\Delta \mathrm{AIC}$", grid="x")
    return caminho


def plotar_top_vertices_mais_importantes(output_dir: str, df_ranking_global: pd.DataFrame, top_n: int = 3, nome_arquivo: str = "top_vertices_mais_importantes.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    df_top, df_t10 = df_ranking_global.head(top_n), df_ranking_global.head(10).iloc[::-1].reset_index(drop=True)
    eixos = ["Pct_Grau_Ponderado", "Pct_Intermediacao", "Pct_Proximidade", "Pct_PageRank", "Score_Global"]
    with salvar_figura(caminho, 1, 3, figsize=(18.5, 5.8)) as (_, (ax1, ax2, ax3)):
        x, w = np.arange(len(eixos)), 0.25
        for i, (_, r) in enumerate(df_top.iterrows()):
            ax1.bar(x + (i - (len(df_top) - 1) / 2.0) * w, [float(r[c]) * 100 for c in eixos], width=w * 0.92, color=["#D32F2F", "#E65100", "#1976D2"][i % 3], edgecolor="#222", label=f"#{int(r['Posicao_Global'])} {r['Neuronio']}")
        ax1.set_xticks(x); ax1.set_xticklabels(["Força", "Betw.", "Close.", "PageRank", "Score"], fontweight="bold")
        ax1.set_ylim(0, 112); ax1.legend(loc="lower right", fontsize=8.8)
        _rotular(ax1, "Perfil dos Top 3 Vértices", ylabel="Percentil Global (%)", grid="y")

        plotar_barras_anotadas(ax2, np.arange(len(df_t10)), (df_t10["Score_Global"] * 100).values, cores=[PALETA_CORES_FUNCIONAIS.get(str(t), "#757575") for t in df_t10["Tipo"]], horizontal=True, fmt="{:.2f}%", largura=0.65)
        ax2.set_yticks(np.arange(len(df_t10))); ax2.set_yticklabels([f"#{int(r['Posicao_Global'])} {r['Neuronio']}" for _, r in df_t10.iterrows()], fontweight="bold")
        ax2.set_xlim(85, 105)
        _rotular(ax2, "Top 10 Neurônios Mais Centrais", "Score Global (%)", grid="x")

        for cls, cor in PALETA_CORES_FUNCIONAIS.items():
            sub = df_ranking_global[df_ranking_global["Tipo"] == cls]
            if not sub.empty:
                ax3.scatter(sub["Grau_Ponderado"], sub["Intermediacao"], s=np.clip(sub["PageRank"] * 12000 + 30, 25, 420), c=cor, alpha=0.65, label=cls)
        for _, r in df_top.iterrows():
            ax3.scatter([r["Grau_Ponderado"]], [r["Intermediacao"]], s=360, facecolors="none", edgecolors="#111", lw=2.0, zorder=6)
            ax3.annotate(str(r["Neuronio"]), (r["Grau_Ponderado"], r["Intermediacao"]), xytext=(6, 6), textcoords="offset points", fontweight="bold", fontsize=8.8)
        ax3.legend(loc="upper left", fontsize=8.2)
        _rotular(ax3, "Paisagem Topológica (Tamanho ∝ PageRank)", "Grau Ponderado", "Intermediação")
    return caminho


def plotar_navegacao_greedy(output_dir: str, df_navegacao: pd.DataFrame, df_nav_fluxo: pd.DataFrame, resumo_nav: Dict[str, Any], nome_arquivo: str = "navegacao_greedy.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    dfu = df_navegacao[df_navegacao["Visao"] == "Combinada (Não-Direcionada)"]
    ests = list(dfu["Estrategia"])
    curtos = [e.replace(" (", "\n(").replace(" Aleatória", "\nAleatória") for e in ests]
    cores, x, w = [PALETA_ESTRATEGIAS_NAVEGACAO.get(e, "#607D8B") for e in ests], np.arange(len(ests)), 0.36
    with salvar_figura(caminho, 1, 3, figsize=(18, 5.8)) as (fig, (ax1, ax2, ax3)):
        sr, er = dfu["Taxa_Sucesso_Pct"].to_numpy(dtype=float), dfu["Razao_Eficiencia"].to_numpy(dtype=float)
        b1 = ax1.bar(x - w / 2, sr, width=w, color=cores, edgecolor="#222", label="Real")
        ax1.bar(x + w / 2, dfu["Nulo_Sucesso_Media"].fillna(0), width=w, yerr=dfu["Nulo_Sucesso_Std"].fillna(0), color="#E0E0E0", edgecolor="#555", hatch="///", capsize=3, label="Nulo")
        for i, b in enumerate(b1):
            ax1.text(b.get_x() + b.get_width() / 2, min(b.get_height() + 2, 108), f"{sr[i]:.1f}%\n" + r"$E_R$=" + f"{er[i]:.2f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
        ax1.set_xticks(x); ax1.set_xticklabels(curtos, fontsize=8.5, fontweight="bold"); ax1.set_ylim(0, 118); ax1.legend(loc="lower right", fontsize=8.5)
        _rotular(ax1, r"Sucesso ($S_R$) e Eficiência ($E_R$)", ylabel=r"$S_R$ (%)", grid="y")

        w2 = 0.25
        p90 = dfu["Stretch_P90"].fillna(1).to_numpy(dtype=float)
        ax2.bar(x - w2, dfu["Stretch_Mediana"].fillna(1), width=w2, color="#90CAF9", edgecolor="#222", label=r"$\tau_{50}$")
        ax2.bar(x, dfu["Stretch_Medio"].fillna(1), width=w2, color=cores, edgecolor="#222", label=r"$\bar{\tau}$")
        ax2.bar(x + w2, p90, width=w2, color="#FFCC80", edgecolor="#222", label=r"$\tau_{90}$")
        ax2.axhline(1.0, color="#2E7D32", ls="--", lw=1.4)
        ax2.set_xticks(x); ax2.set_xticklabels(curtos, fontsize=8.5, fontweight="bold"); ax2.set_ylim(0, max(float(np.nanmax(p90)) * 1.25, 2.2)); ax2.legend(loc="upper left", fontsize=8.5)
        _rotular(ax2, r"Alongamento das Rotas ($\tau$)", ylabel=r"Stretch ($\tau$)", grid="y")

        melhor = resumo_nav.get("melhor_estrategia_undir", "Grau (Hubs)")
        sub = df_nav_fluxo[df_nav_fluxo["Estrategia"] == melhor]
        cls = [c for c in ORDEM_CLASSES_PADRAO if c in set(sub["Tipo_Origem"]) | set(sub["Tipo_Destino"])]
        mat_sr, mat_tau = np.full((len(cls), len(cls)), np.nan), np.full((len(cls), len(cls)), np.nan)
        for _, r in sub.iterrows():
            if r["Tipo_Origem"] in cls and r["Tipo_Destino"] in cls:
                mat_sr[cls.index(r["Tipo_Origem"]), cls.index(r["Tipo_Destino"])] = float(r["Taxa_Sucesso_Pct"])
                mat_tau[cls.index(r["Tipo_Origem"]), cls.index(r["Tipo_Destino"])] = float(r["Stretch_Medio"])
        rot_c = [c.replace("Interneurônio (Comando)", "Comando").replace("Interneurônio", "Inter.") for c in cls]
        plotar_heatmap_matriz(fig, ax3, mat_sr, rot_c, rot_c, cmap="YlGn", vmin=0, vmax=100, fmt_celula=lambda i, j, v: "—" if np.isnan(v) else f"{v:.1f}%\n({mat_tau[i, j]:.2f})", cbar_label=r"$S_R$ (%)", rotation=20)
        ax3.set_title(f"Fluxo Funcional — {melhor}", fontweight="bold")
    return caminho


def plotar_assortatividade(output_dir: str, df_assortatividade: pd.DataFrame, df_knn: pd.DataFrame, resumo_assort: Dict[str, Any], nome_arquivo: str = "assortatividade_conectoma.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    with salvar_figura(caminho, 1, 2, figsize=(16, 6.0)) as (_, (ax1, ax2)):
        if not df_knn.empty:
            kv, kr = df_knn["k"].to_numpy(dtype=float), df_knn["knn_real"].to_numpy(dtype=float)
            kn, ks = df_knn["knn_nulo_media"].to_numpy(dtype=float), df_knn["knn_nulo_std"].fillna(0).to_numpy(dtype=float)
            mk = np.isfinite(kn) & (kn > 0)
            if np.any(mk):
                ax1.fill_between(kv[mk], np.maximum(kn[mk] - ks[mk], 1e-2), kn[mk] + ks[mk], color="#BDBDBD", alpha=0.4, label="Nulo")
                ax1.plot(kv[mk], kn[mk], color="#616161", ls="--", lw=1.6, label=r"$k_{\mathrm{nn}}^{\mathrm{nulo}}$")
            ax1.scatter(kv, kr, color="#1565C0", s=34, alpha=0.85, edgecolor="white", lw=0.5, label="Empírico")
            bk, ik = float(resumo_assort.get("beta_knn", 0)), float(resumo_assort.get("intercept_knn", 0))
            if len(kv) >= 2:
                kg = np.geomspace(float(np.min(kv)), float(np.max(kv)), 80)
                ax1.plot(kg, np.exp(ik) * np.power(kg, bk), color="#D32F2F", lw=2.1, label=rf"$\beta_{{k_{{\mathrm{{nn}}}}}}={bk:.3f}$")
        ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.legend(loc="lower left", fontsize=8.8)
        _rotular(ax1, rf"Curva $k_{{\mathrm{{nn}}}}(k)$ ($r_k={float(resumo_assort.get('r_grau_combinada', 0)):+.3f}$, $Z={float(resumo_assort.get('z_score_r_grau', 0)):+.1f}$)", r"Grau ($k$)", r"$k_{\mathrm{nn}}(k)$")

        itens = [
            (r"Grau Comb. ($r_k$)", "r_grau_combinada"), (r"Grau Pond. ($r_k^w$)", "r_grau_combinada_ponderada"),
            (r"Química ($\mathrm{out}\to\mathrm{in}$)", "r_quimica_out_in"), (r"Elétrica ($r_k^{\mathrm{elet}}$)", "r_eletrica_grau"),
            ("Homofilia Funcional", "r_homofilia_tipo_funcional"), ("Homofilia Anatômica", "r_homofilia_regiao_soma"),
            ("Correlação Espacial", "r_espacial_soma_pos"),
        ]
        vals = [float(resumo_assort.get(k, 0)) for _, k in itens]
        plotar_barras_anotadas(ax2, np.arange(len(itens)), vals, cores=["#34A853" if v >= 0 else "#EA4335" for v in vals], horizontal=True, fmt="{:+.3f}")
        ax2.axvline(0.0, color="#222", lw=1.1)
        lim = max(0.35, min(1.05, max(abs(v) for v in vals) + 0.16)); ax2.set_xlim(-lim, lim)
        ax2.set_yticks(np.arange(len(itens))); ax2.set_yticklabels([r for r, _ in itens], fontweight="bold"); ax2.invert_yaxis()
        _rotular(ax2, "Perfil Multidimensional de Assortatividade", r"Assortatividade ($r$)", grid="x")
    return caminho


def plotar_preservacao_reciprocidade_scc(output_dir: str, df_preservacao_resumo: pd.DataFrame, mat_reciprocidade: pd.DataFrame, df_bowtie_classes: pd.DataFrame, resumo_json: Dict[str, Any], nome_arquivo: str = "preservacao_reciprocidade_scc.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    rec, scc = resumo_json.get("reciprocidade", {}), resumo_json.get("scc_bowtie", {})
    with salvar_figura(caminho, 1, 3, figsize=(18.5, 5.8)) as (fig, (ax1, ax2, ax3)):
        vr = [float(rec.get(k, 0)) * 100 for k in ("quimica_topologica", "quimica_ponderada", "combinada_topologica")]
        vn = [float(rec.get(k, 0)) * 100 for k in ("quimica_nulo_topologica_media", "quimica_nulo_ponderada_media", "combinada_nulo_topologica_media")]
        sn = [float(rec.get(k, 0)) * 100 for k in ("quimica_nulo_topologica_std", "quimica_nulo_ponderada_std", "combinada_nulo_topologica_std")]
        zs = [float(rec.get(k, 0)) for k in ("z_score_reciprocidade", "z_score_reciprocidade_ponderada", "z_score_combinada_topologica")]
        x, w = np.arange(3), 0.34
        b1 = ax1.bar(x - w / 2, vr, width=w, color="#EA4335", edgecolor="#222", label="Real")
        ax1.bar(x + w / 2, vn, width=w, yerr=sn, capsize=4, color="#9E9E9E", edgecolor="#222", label="Nulo")
        for b, v, z in zip(b1, vr, zs):
            ax1.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%\n(Z={z:+.1f})", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
        ax1.set_xticks(x); ax1.set_xticklabels([r"Química ($R$)", r"Química ($R_{w}$)", r"Combinada ($R$)"], fontweight="bold")
        ax1.set_ylim(0, min(115, max(vr + vn) * 1.3)); ax1.legend(loc="upper left", fontsize=8.5)
        _rotular(ax1, r"Reciprocidade ($R$ e $R_w$)", ylabel="Reciprocidade (%)", grid="y")

        mp = mat_reciprocidade.to_numpy(dtype=float) * 100.0
        rc = [c.replace("Interneurônio (Comando)", "Comando").replace("Interneurônio", "Inter.") for c in mat_reciprocidade.index]
        plotar_heatmap_matriz(fig, ax2, mp, rc, rc, cmap="YlOrRd", vmin=0, vmax=max(float(np.nanmax(mp)), 10), fmt_celula=lambda i, j, v: f"{v:.1f}%", cbar_label=r"$R_{AB}$ (%)", rotation=22)
        ax2.set_title(r"Reciprocidade Química ($R_{AB}$)", fontweight="bold")

        setores = ["IN (Entrada / Feeders)", "SCC (Núcleo Recorrente)", "OUT (Saída / Sinks)", "Outros (Isolados/Tubes)"]
        df_bt, xb, base = df_bowtie_classes.set_index("Setor_BowTie"), np.arange(4), np.zeros(4)
        for cls in [c for c in ORDEM_CLASSES_PADRAO if c in df_bt.columns]:
            alt = np.array([float(df_bt.loc[s, cls]) if s in df_bt.index else 0.0 for s in setores])
            ax3.bar(xb, alt, bottom=base, width=0.58, color=PALETA_CORES_FUNCIONAIS.get(cls, "#757575"), edgecolor="#222", label=cls.replace("Interneurônio (Comando)", "Comando"))
            base += alt
        for i, tot in enumerate(base):
            ax3.text(i, tot + max(base) * 0.02, f"{int(round(tot))}", ha="center", va="bottom", fontweight="bold", fontsize=9)
        ax3.set_xticks(xb); ax3.set_xticklabels(["IN", "SCC", "OUT", "Outros"], fontweight="bold")
        ax3.set_ylim(0, max(base) * 1.24 if max(base) > 0 else 10); ax3.legend(loc="upper right", fontsize=8)
        _rotular(ax3, rf"Bow-Tie (LSCC={scc.get('quimica_lscc_fracao_pct', 0):.1f}%)", ylabel="Neurônios", grid="y")
    return caminho


def plotar_preservacao_modularidade_ciclos(output_dir: str, df_modularidade: pd.DataFrame, df_ciclos_motifs: pd.DataFrame, df_ciclos_por_classe: pd.DataFrame, resumo_json: Dict[str, Any], nome_arquivo: str = "preservacao_modularidade_ciclos.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    mod = resumo_json.get("modularidade", {})
    with salvar_figura(caminho, 1, 3, figsize=(18.5, 5.8)) as (_, (ax1, ax2, ax3)):
        vq = [float(mod.get(k, 0)) for k in ("Q_quimica_louvain_ponderada", "Q_combinada_louvain_ponderada", "Q_funcional_quimica", "Q_anatomica_quimica", "Q_quimica_nulo_media")]
        plotar_barras_anotadas(ax1, np.arange(5), vq, yerr=[0, 0, 0, 0, float(mod.get("Q_quimica_nulo_std", 0))], cores=["#EA4335", "#4285F4", "#FBBC05", "#34A853", "#9E9E9E"], fmt="{:.3f}", capsize=4)
        ax1.axhline(0.0, color="#222", lw=0.9); ax1.set_xticks(np.arange(5)); ax1.set_xticklabels(["Louvain\nQuím.", "Louvain\nComb.", "Func.", "Anat.", "Nulo"], fontsize=8.5, fontweight="bold")
        ax1.set_ylim(min(0, min(vq) - 0.08), max(max(vq) * 1.32, 0.45))
        _rotular(ax1, rf"Modularidade ($Z_Q={float(mod.get('Q_quimica_z_score', 0)):+.1f}$)", ylabel=r"$Q$", grid="y")

        zm, ym = df_ciclos_motifs["Z_Score"].to_numpy(dtype=float), np.arange(len(df_ciclos_motifs))
        plotar_barras_anotadas(ax2, ym, zm, cores=["#EA4335" if z > 2 else ("#4285F4" if z < -2 else "#9E9E9E") for z in zm], horizontal=True, fmt="{:+.1f}")
        for x0, ls in ((0.0, "-"), (2.0, "--"), (-2.0, "--")):
            ax2.axvline(x0, color="#222" if x0 == 0 else "#666", ls=ls, lw=1.0)
        ax2.set_yticks(ym); ax2.set_yticklabels(df_ciclos_motifs["Estrutura"], fontsize=8.5, fontweight="bold"); ax2.invert_yaxis()
        _rotular(ax2, r"Ciclos ($C_2, C_3$) e Tríades", r"$Z$-Score vs. Nulo", grid="x")

        grps = [g.replace("Interneurônio (Comando)", "Comando").replace("Interneurônio", "Inter.").replace("Núcleo Rich-Club", "Rich-Club").replace("Periferia (Demais)", "Periferia") for g in df_ciclos_por_classe["Grupo"]]
        xc, w = np.arange(len(grps)), 0.36
        ax3.bar(xc - w / 2, np.maximum(df_ciclos_por_classe["C2_Media"].values, 0.05), width=w, color="#4285F4", edgecolor="#222", label=r"$C_2$")
        ax3.bar(xc + w / 2, np.maximum(df_ciclos_por_classe["C3_Media"].values, 0.05), width=w, color="#EA4335", edgecolor="#222", label=r"$C_3$")
        ax3.set_yscale("log"); ax3.set_xticks(xc); ax3.set_xticklabels(grps, rotation=24, ha="right", fontsize=8.5, fontweight="bold"); ax3.legend(loc="upper left", fontsize=8.5)
        _rotular(ax3, "Participação em Ciclos por Classe", ylabel="Média (log)", grid="y")
    return caminho


def plotar_difusao_estrutural(output_dir: str, df_estrutural: pd.DataFrame, df_distancias: pd.DataFrame, resumo_json: Dict[str, Any], nome_arquivo: str = "difusao_metricas_estruturais.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    eu = resumo_json.get("estrutural", {}).get("combinada_undir", {})
    w = 0.36
    with salvar_figura(caminho, 1, 3, figsize=(18.5, 5.8)) as (_, (ax1, ax2, ax3)):
        sub1 = df_estrutural[df_estrutural["Visao"].isin(["Combinada (Não-Direcionada)", "Combinada (Direcionada)"])]
        vr, ve, se = [], [], []
        for _, r in sub1.iterrows():
            vr.extend([float(r["L_Real"]), float(r["D_Real"])]); ve.extend([float(r["L_ER_Media"]), float(r["D_ER_Media"])]); se.extend([float(r["L_ER_Std"]), float(r["D_ER_Std"])])
        x1 = np.arange(len(vr))
        ax1.bar(x1 - w / 2, vr, width=w, color="#EA4335", edgecolor="#222", label="Real")
        ax1.bar(x1 + w / 2, ve, width=w, yerr=se, capsize=3, color="#9E9E9E", edgecolor="#222", label="ER")
        ax1.set_xticks(x1); ax1.set_xticklabels([r"$L$ Undir", r"$D$ Undir", r"$L$ Dir", r"$D$ Dir"], fontweight="bold"); ax1.legend(loc="upper left", fontsize=8.5)
        _rotular(ax1, r"Caminho Médio ($L$) e Diâmetro ($D$)", ylabel="Saltos", grid="y")

        x2 = np.arange(len(df_estrutural))
        ax2.bar(x2 - w / 2, df_estrutural["E_Glob_Real"], width=w, color="#4285F4", edgecolor="#222", label="Real")
        ax2.bar(x2 + w / 2, df_estrutural["E_Glob_ER_Media"], width=w, yerr=df_estrutural["E_Glob_ER_Std"], capsize=3, color="#9E9E9E", edgecolor="#222", label="ER")
        ax2.set_xticks(x2); ax2.set_xticklabels(["Comb.\nUndir", "Comb.\nDir", "Química\nDir"][:len(x2)], fontweight="bold"); ax2.legend(loc="upper right", fontsize=8.5)
        _rotular(ax2, rf"$E_{{\mathrm{{glob}}}}$ ($\lambda_c={float(eu.get('lambda_c_real') or 0):.4f}$)", ylabel=r"$E_{\mathrm{glob}}$", grid="y")

        sub_d = df_distancias[df_distancias["Visao"] == "Combinada (Não-Direcionada)"]
        x3 = np.arange(len(sub_d))
        ax3.bar(x3 - w / 2, sub_d["Pct_Real"], width=w, color="#EA4335", edgecolor="#222", label="Real")
        ax3.bar(x3 + w / 2, sub_d["Pct_ER_Media"], width=w, yerr=sub_d["Pct_ER_Std"], capsize=3, color="#9E9E9E", edgecolor="#222", label="ER")
        ax3.set_xticks(x3); ax3.set_xticklabels([f"d={d}" for d in sub_d["Distancia"]], fontweight="bold"); ax3.legend(loc="upper right", fontsize=8.5)
        _rotular(ax3, rf"Distâncias $P(d)$ ($d\leq 3$: {float(eu.get('pct_pares_ate_3_saltos_real') or 0):.1f}%)", ylabel=r"$P(d)$ (%)", grid="y")
    return caminho


def plotar_simulacao_difusao(output_dir: str, df_si_curvas: pd.DataFrame, df_si_resumo: pd.DataFrame, df_ic_curvas: pd.DataFrame, resumo_json: Dict[str, Any], nome_arquivo: str = "difusao_simulacao_propagacao.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    vp = "Combinada (Não-Direcionada)"
    b_si, b_ref = float(resumo_json.get("metadados", {}).get("beta_si_padrao", 0.08)), float(resumo_json.get("simulacao_ic", {}).get("beta_referencia", 0.08))
    cenarios = list(PALETA_CENARIOS_DIFUSAO.keys())
    with salvar_figura(caminho, 1, 3, figsize=(18.5, 5.8)) as (_, (ax1, ax2, ax3)):
        sub_si, sub_ic = df_si_curvas[df_si_curvas["Visao"] == vp], df_ic_curvas[df_ic_curvas["Visao"] == vp]
        for cen in cenarios:
            cor, ls = PALETA_CENARIOS_DIFUSAO[cen], ("--" if cen.startswith("Erdős") else "-")
            for ax, df_s, xcol, mcol, scol, mk in ((ax1, sub_si, "Passo_t", "Infectados_Pct_Media", "Infectados_Pct_Std", None), (ax2, sub_ic, "Beta", "Alcance_Final_Pct_Media", "Alcance_Final_Pct_Std", "o")):
                dc = df_s[df_s["Cenario"] == cen].sort_values(xcol)
                if not dc.empty:
                    m, s = dc[mcol].values, dc[scol].values
                    ax.plot(dc[xcol], m, color=cor, ls=ls, marker=mk, ms=4, lw=2.0, label=cen)
                    ax.fill_between(dc[xcol], np.clip(m - s, 0, 100), np.clip(m + s, 0, 100), color=cor, alpha=0.12)
        ax1.axhline(50, color="#333", ls=":", lw=1.2); ax1.set_ylim(0, 105); ax1.legend(loc="lower right", fontsize=7.6)
        _rotular(ax1, rf"Modelo SI ($\beta={b_si:.2f}$)", r"Passo ($t$)", r"$I(t)/N$ (%)")
        ax2.axhline(20, color="#666", ls=":", lw=1.1); ax2.axvline(b_ref, color="#333", ls="-.", lw=1.0); ax2.set_ylim(0, 105); ax2.legend(loc="lower right", fontsize=7.6)
        _rotular(ax2, "Cascata Independente (IC / SIR)", r"Transmissibilidade ($\beta$)", r"Alcance $R_\infty/N$ (%)")

        sr, ir = df_si_resumo[df_si_resumo["Visao"] == vp], sub_ic[np.isclose(sub_ic["Beta"].to_numpy(dtype=float), b_ref, atol=1e-9)]
        aucs = [float(sr[sr["Cenario"] == c]["AUC_SI_Media"].iloc[0]) * 100 if not sr[sr["Cenario"] == c].empty else 0.0 for c in cenarios]
        alcs = [float(ir[ir["Cenario"] == c]["Alcance_Final_Pct_Media"].iloc[0]) if not ir[ir["Cenario"] == c].empty else 0.0 for c in cenarios]
        y3, h = np.arange(len(cenarios)), 0.36
        ax3.barh(y3 - h / 2, aucs, height=h, color="#4285F4", edgecolor="#222", label=r"$\mathrm{AUC}_{\mathrm{SI}}$")
        ax3.barh(y3 + h / 2, alcs, height=h, color="#EA4335", edgecolor="#222", label=rf"IC ($\beta={b_ref:.2f}$)")
        ax3.set_yticks(y3); ax3.set_yticklabels([c.replace("C. elegans ", "C.el. ").replace("Erdős-Rényi ", "ER ") for c in cenarios], fontsize=8.3, fontweight="bold")
        ax3.invert_yaxis(); ax3.set_xlim(0, 115); ax3.legend(loc="lower right", fontsize=8.2)
        _rotular(ax3, "Resumo Dinâmico", "Desempenho (%)", grid="x")
    return caminho


def plotar_robustez_percolacao(output_dir: str, df_percolacao: pd.DataFrame, df_schneider: pd.DataFrame, resumo_json: Dict[str, Any], nome_arquivo: str = "robustez_percolacao.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    f_rc = float(resumo_json["metadados"]["n_rich_club"]) / max(int(resumo_json["metadados"]["N"]), 1)
    with salvar_figura(caminho, 1, 3, figsize=(19, 5.8)) as (_, (ax1, ax2, ax3)):
        for reg, cor in PALETA_REGIMES_ROBUSTEZ.items():
            sub = df_percolacao[df_percolacao["Regime"] == reg]
            if sub.empty:
                continue
            ls, x = ("--" if "Estático" in reg else "-"), sub["Fracao_Removida"].values
            sm, em = sub["GCC_Media"].values, sub["Eficiencia_Norm_Media"].values
            ax1.plot(x, sm, color=cor, ls=ls, lw=2.0, label=reg); ax2.plot(x, em, color=cor, ls=ls, lw=2.0)
            if reg == "Falha Aleatória":
                ax1.fill_between(x, np.clip(sm - sub["GCC_Std"].values, 0, 1), np.clip(sm + sub["GCC_Std"].values, 0, 1), color=cor, alpha=0.16)
                ax2.fill_between(x, np.clip(em - sub["Eficiencia_Norm_Std"].values, 0, 1), np.clip(em + sub["Eficiencia_Norm_Std"].values, 0, 1), color=cor, alpha=0.16)
        for ax, yl, tit in ((ax1, r"$S(q)$", "Colapso da GCC"), (ax2, r"$E_{\mathrm{glob}}(f)/E_0$", "Queda de Eficiência")):
            ax.axhline(0.5, color="#555", ls="--", lw=1.0); ax.axvline(f_rc, color="#FF6D00", ls=":", lw=1.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
            _rotular(ax, tit, r"Fração Removida ($f$)", yl)
        ax1.legend(loc="upper right", fontsize=7.5)

        dfo = df_schneider.sort_values("Schneider_R", ascending=True).reset_index(drop=True)
        y = np.arange(len(dfo))
        ax3.barh(y, dfo["Schneider_R"], xerr=dfo["Schneider_R_Std"], color=[PALETA_REGIMES_ROBUSTEZ.get(r, "#607D8B") for r in dfo["Regime"]], edgecolor="#222", capsize=3)
        ax3.axvline(0.5, color="#222", ls=":", lw=1.3, label=r"$R_{\max}=0{,}50$")
        for i, r in dfo.iterrows():
            ax3.text(float(r["Schneider_R"]) + 0.012, i, rf"$R={r['Schneider_R']:.3f}$ ($f_{{c,50}}={r['fc_50']:.2f}$)", va="center", fontsize=8.0, fontweight="bold")
        ax3.set_yticks(y); ax3.set_yticklabels(dfo["Regime"], fontsize=8.2, fontweight="bold"); ax3.set_xlim(0, 0.68); ax3.legend(loc="lower right", fontsize=8.0)
        _rotular(ax3, r"Ranking de Robustez ($R$)", r"Schneider ($R$)", grid="x")
    return caminho


def plotar_robustez_sinteticas_e_funcional(output_dir: str, df_sinteticas_curvas: pd.DataFrame, df_sinteticas_resumo: pd.DataFrame, df_funcional_dir: pd.DataFrame, resumo_json: Dict[str, Any], nome_arquivo: str = "robustez_sinteticas_funcional.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    with salvar_figura(caminho, 1, 3, figsize=(19, 5.8)) as (_, (ax1, ax2, ax3)):
        for mod, cor in PALETA_MODELOS_SINTETICOS.items():
            for reg, ls, mk, suf in (("Falha Aleatória", "--", None, "Aleat."), ("Ataque Adaptativo (Grau)", "-", "o", "Ataque")):
                sub = df_sinteticas_curvas[(df_sinteticas_curvas["Modelo"] == mod) & (df_sinteticas_curvas["Regime"] == reg)]
                if not sub.empty:
                    ax1.plot(sub["Fracao_Removida"], sub["GCC_Media"], color=cor, ls=ls, marker=mk, markevery=25, ms=3.8, lw=1.8, label=f"{mod} ({suf})")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.03); ax1.legend(loc="upper right", fontsize=6.9)
        _rotular(ax1, "Percolação: Real vs. ER, WS e BA", r"Fração Removida ($f$)", r"$S(q)$")

        x2, w = np.arange(len(df_sinteticas_resumo)), 0.36
        ax2.bar(x2 - w / 2, df_sinteticas_resumo["R_Aleatorio_Media"], w, yerr=df_sinteticas_resumo["R_Aleatorio_Std"], color="#34A853", edgecolor="#222", capsize=3, label="Aleatória")
        ax2.bar(x2 + w / 2, df_sinteticas_resumo["R_Ataque_Adapt_Media"], w, yerr=df_sinteticas_resumo["R_Ataque_Adapt_Std"], color="#EA4335", edgecolor="#222", capsize=3, label="Ataque")
        for i, r in df_sinteticas_resumo.iterrows():
            ax2.text(i, max(r["R_Aleatorio_Media"], r["R_Ataque_Adapt_Media"]) + 0.025, rf"$\rho_R={r['Razao_RYF']:.2f}\times$", ha="center", fontsize=8.4, fontweight="bold")
        ax2.set_xticks(x2); ax2.set_xticklabels([m.replace(" (", "\n(") for m in df_sinteticas_resumo["Modelo"]], fontsize=8.0, fontweight="bold"); ax2.set_ylim(0, 0.58); ax2.legend(loc="upper right", fontsize=8.0)
        _rotular(ax2, r"Assimetria $\rho_R = R_{\mathrm{aleat}}/R_{\mathrm{ataque}}$", ylabel=r"Schneider ($R$)", grid="y")

        for reg, c1, c2, lbl in (("Rich-Club -> Grau Adaptativo (Dir.)", "#EA4335", "#FF6D00", "Rich-Club"), ("Falha Aleatória (Dir.)", "#34A853", "#4285F4", "Aleatória")):
            sub = df_funcional_dir[df_funcional_dir["Regime"] == reg]
            if not sub.empty:
                ax3.plot(sub["Fracao_Removida"], sub["Frac_LSCC"] * 100.0, color=c1, lw=2.0, label=f"LSCC — {lbl}")
                ax3.plot(sub["Fracao_Removida"], sub["Alcance_S_para_M_Pct"], color=c2, lw=1.8, ls="--", label=rf"$S\to M$ — {lbl}")
        ab = resumo_json["ablacao_rich_club"]["combinada_dir"]
        ax3.set_xlim(0, 0.65); ax3.set_ylim(0, 105); ax3.legend(loc="lower left", fontsize=7.6)
        _rotular(ax3, rf"Direcionada ($\Delta E_{{S\to M}}=-{ab['eficiencia_sm_queda_rc_pct']:.1f}\%$, $Z={ab['z_score_queda_eficiencia_sm']:.1f}$)", r"Fração Removida ($f$)", "Integridade (%)")
    return caminho


def plotar_custo_cabeamento(output_dir: str, resultado_custo: Dict[str, Any], nome_arquivo: str = "custo_cabeamento_espacial.png") -> str:
    caminho = resolver_caminho_saida(output_dir, nome_arquivo)
    df_c = resultado_custo["df_custo_nulo"]
    with salvar_figura(caminho, 1, 2, figsize=(15.5, 5.8)) as (_, (ax1, ax2)):
        x, w = np.arange(len(df_c)), 0.36
        ax1.bar(x - w / 2, df_c["Custo_Real"], w, color="#4285F4", edgecolor="#222", label=r"$W_{\mathrm{real}}$")
        ax1.bar(x + w / 2, df_c["Custo_Nulo_Media"], w, yerr=df_c["Custo_Nulo_Std"], color="#9E9E9E", edgecolor="#222", capsize=4, label=r"$W_{\mathrm{null}}$")
        for i, r in df_c.iterrows():
            ax1.text(i, max(r["Custo_Real"], r["Custo_Nulo_Media"] + r["Custo_Nulo_Std"]) * 1.03, f"-{r['Economia_Cabeamento_Pct']:.1f}%\n(Z={r['Z_Score_Custo']:.1f})", ha="center", va="bottom", fontsize=8.4, fontweight="bold")
        ax1.set_xticks(x); ax1.set_xticklabels(df_c["Camada"], fontweight="bold"); ax1.set_ylim(0, df_c["Custo_Nulo_Media"].max() * 1.24); ax1.legend(loc="upper left", fontsize=8.5)
        _rotular(ax1, "Custo de Cabeamento vs. Modelo Nulo", ylabel=r"$W_{\mathrm{cost}}$", grid="y")

        bins = np.linspace(0.0, 1.0, 26)
        ax2.hist(resultado_custo["dist_comprimentos_real"], bins=bins, density=True, alpha=0.58, color="#4285F4", edgecolor="#1A73E8", label="Real")
        ax2.hist(resultado_custo["dist_comprimentos_nulo"], bins=bins, density=True, alpha=0.40, color="#9E9E9E", edgecolor="#616161", label="Nulo")
        for grp, cor in (("Periféricas", "#34A853"), ("Feeder / Ponte Rich-Club", "#FBBC05"), ("Internas ao Rich-Club", "#EA4335")):
            vals = resultado_custo["comprimentos_por_grupo_rc"].get(grp, [])
            if vals:
                ax2.axvline(float(np.mean(vals)), color=cor, ls="--", lw=1.8, label=f"{grp} ({float(np.mean(vals)):.3f})")
        ax2.set_xlim(0, 1.0); ax2.legend(loc="upper right", fontsize=8.0)
        _rotular(ax2, "Distribuição de Comprimentos Físicos", r"$|x_u - x_v|$", "Densidade")
    return caminho
