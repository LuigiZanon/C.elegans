"""Fachada enxuta de visualização delegando plots estáticos para plots.py e HTML para dashboard_exporter.py."""

import os
from typing import Any
import networkx as nx
from config import OUTPUT_DIR
import dashboard_exporter
import plots


class ConnectomeVisualizer:
    """Mantém compatibilidade de interface delegando figuras a plots.py e HTML a dashboard_exporter.py."""

    def __init__(self, graph: nx.MultiDiGraph, output_dir: str = OUTPUT_DIR):
        self.graph = graph
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("plotar_") and hasattr(plots, name):
            fn = getattr(plots, name)
            return lambda *args, **kwargs: fn(self.output_dir, *args, **kwargs)
        raise AttributeError(f"'{type(self).__name__}' não possui o atributo '{name}'")

    def exportar_html_interativo(self, *args: Any, **kwargs: Any) -> str:
        return dashboard_exporter.exportar_html_interativo(self.graph, self.output_dir, *args, **kwargs)
