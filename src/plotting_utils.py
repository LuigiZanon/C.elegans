"""Utilitários compartilhados de renderização Matplotlib (backend Agg)."""

from contextlib import contextmanager
import os
from typing import Callable, Optional, Sequence, Tuple
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolver_caminho_saida(output_dir: str, nome_arquivo: str) -> str:
    """Garante a existência do diretório e resolve o caminho completo do arquivo de saída."""
    if os.path.dirname(nome_arquivo):
        os.makedirs(os.path.dirname(nome_arquivo), exist_ok=True)
        return nome_arquivo
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, nome_arquivo)


def estilizar_eixo(ax, grid_axis: str = "both") -> None:
    """Aplica estilo limpo padronizado: remove bordas superior/direita e ativa grid sutil."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.28, linestyle=":")


@contextmanager
def salvar_figura(
    caminho: str,
    nrows: int = 1,
    ncols: int = 1,
    figsize: Tuple[float, float] = (16.0, 6.0),
    dpi: int = 300,
    tight_rect: Optional[Sequence[float]] = None,
):
    """Cria (fig, axes), executa tight_layout + savefig e garante plt.close(fig) no finally."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    try:
        yield fig, axes
        if tight_rect is not None:
            plt.tight_layout(rect=list(tight_rect))
        else:
            plt.tight_layout()
        fig.savefig(caminho, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)


def plotar_barras_anotadas(
    ax,
    posicoes: Sequence[float],
    valores: Sequence[float],
    yerr: Optional[Sequence[float]] = None,
    cores: Optional[Sequence[str]] = None,
    largura: float = 0.58,
    fmt: str = "{:.2f}",
    horizontal: bool = False,
    fontsize: float = 8.5,
    **kwargs,
):
    """Plota barras verticais ou horizontais com rótulos numéricos padronizados."""
    kw = {"edgecolor": "#222222", "linewidth": 0.85, "alpha": 0.90}
    kw.update(kwargs)
    if horizontal:
        barras = ax.barh(posicoes, valores, height=largura, xerr=yerr, color=cores, **kw)
        span = max(abs(float(v)) for v in valores) if len(valores) else 1.0
        for b, v in zip(barras, valores):
            off = 0.02 * max(span, 1e-6) * (1 if v >= 0 else -1)
            ax.text(v + off, b.get_y() + b.get_height() / 2.0, fmt.format(v),
                    va="center", ha=("left" if v >= 0 else "right"), fontsize=fontsize, fontweight="bold")
        return barras
    barras = ax.bar(posicoes, valores, width=largura, yerr=yerr, color=cores, **kw)
    errs = yerr if yerr is not None else [0.0] * len(valores)
    span = max(abs(float(v)) + abs(float(e)) for v, e in zip(valores, errs)) if len(valores) else 1.0
    for b, v, e in zip(barras, valores, errs):
        cy = (v + e + 0.025 * span) if v >= 0 else (v - e - 0.025 * span)
        ax.text(b.get_x() + b.get_width() / 2.0, cy, fmt.format(v),
                ha="center", va=("bottom" if v >= 0 else "top"), fontsize=fontsize, fontweight="bold")
    return barras


def plotar_heatmap_matriz(
    fig,
    ax,
    matriz: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    cmap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt_celula: Optional[Callable[[int, int, float], str]] = None,
    cbar_label: str = "",
    rotation: int = 25,
) -> None:
    """Renderiza heatmap anotado com contraste automático de texto e colorbar."""
    im = ax.imshow(matriz, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=rotation, ha="right" if rotation else "center", fontsize=9, fontweight="bold")
    ax.set_yticklabels(ylabels, fontsize=9, fontweight="bold")

    v_lim = (vmax if vmax is not None else (float(np.nanmax(matriz)) if matriz.size else 1.0)) * 0.55
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            val = float(matriz[i, j])
            txt = fmt_celula(i, j, val) if fmt_celula else f"{val:.1f}"
            cor = "white" if (not np.isnan(val) and val > v_lim) else "#111111"
            ax.text(j, i, txt, ha="center", va="center", color=cor, fontsize=8.8, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9, fontweight="bold")
