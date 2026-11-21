"""Utilitários compartilhados de projeção, modelos nulos, estatística e persistência."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import math
import os
import numpy as np
import pandas as pd
import networkx as nx


def projetar_grafo(
    graph: nx.MultiDiGraph,
    camada: str = "Combinada",
    direcionado: bool = True,
    adicionar_distancia: bool = False,
    preservar_atributos_nos: bool = True,
) -> Union[nx.DiGraph, nx.Graph]:
    """
    Projeta o MultiDiGraph em Graph ou DiGraph simples sem auto-laços.
    - camada in {'Quimica', 'Eletrica', 'Combinada'}
    - Na camada 'Eletrica' em modo direcionado, insere a aresta nos dois sentidos (u->v e v->u)
      usando max(peso) para pares duplicados.
    - Se adicionar_distancia=True, define dados['distance'] = 1.0 / dados['weight'].
    """
    if camada not in {"Quimica", "Eletrica", "Combinada"}:
        raise ValueError(f"Camada inválida: {camada}")

    g: Union[nx.DiGraph, nx.Graph] = nx.DiGraph() if direcionado else nx.Graph()
    if preservar_atributos_nos:
        for no, dados in graph.nodes(data=True):
            g.add_node(no, **dados)
    else:
        g.add_nodes_from(graph.nodes())

    pesos_eletricos: Dict[Tuple[Any, Any], float] = {}
    for u, v, chave, dados in graph.edges(keys=True, data=True):
        if u == v:
            continue
        tipo = dados.get("synapse_type", chave)
        peso = float(dados.get("weight", 1.0))
        if peso < 0:
            raise ValueError(f"Peso sináptico negativo em {u} -> {v}: {peso}")
        if peso == 0:
            continue

        if tipo == "Eletrica" and camada in {"Eletrica", "Combinada"}:
            par = (u, v) if str(u) <= str(v) else (v, u)
            pesos_eletricos[par] = max(pesos_eletricos.get(par, 0.0), peso)
        elif tipo == "Quimica" and camada in {"Quimica", "Combinada"}:
            if g.has_edge(u, v):
                g[u][v]["weight"] += peso
            else:
                g.add_edge(u, v, weight=peso)

    for (u, v), peso in pesos_eletricos.items():
        pares = ((u, v), (v, u)) if direcionado else ((u, v),)
        for a, b in pares:
            if g.has_edge(a, b):
                g[a][b]["weight"] += peso
            else:
                g.add_edge(a, b, weight=peso)

    if adicionar_distancia:
        for _, _, dados in g.edges(data=True):
            dados["distance"] = 1.0 / dados["weight"]
    return g


def extrair_componente_principal(
    g: Union[nx.Graph, nx.DiGraph],
) -> Union[nx.Graph, nx.DiGraph]:
    """Retorna a maior componente conexa (GCC) se não-direcionado, ou fortemente conexa (LSCC) se direcionado."""
    if g.number_of_nodes() == 0:
        return g
    if g.is_directed():
        comps = nx.strongly_connected_components(g)
    else:
        if nx.is_connected(g):
            return g
        comps = nx.connected_components(g)
    maior = max(comps, key=len)
    return g.subgraph(maior).copy()


def gerar_nulo_direcionado(
    arestas: Sequence[Tuple[str, str, float]],
    nos: Sequence[str],
    n_swaps: int,
    rng: np.random.Generator,
    embaralhar_pesos: bool = True,
) -> nx.DiGraph:
    """Executa Directed Double-Edge Swap preservando exatamente k_in e k_out de cada vértice."""
    g_null = nx.DiGraph()
    g_null.add_nodes_from(nos)
    m = len(arestas)
    if m == 0:
        return g_null
    if m == 1:
        u, v, w = arestas[0]
        g_null.add_edge(u, v, weight=float(w))
        return g_null

    origens = [u for u, _, _ in arestas]
    destinos = [v for _, v, _ in arestas]
    pesos = [float(w) for _, _, w in arestas]
    conjunto = set(zip(origens, destinos))

    for _ in range(n_swaps):
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        u1, v1 = origens[i], destinos[i]
        u2, v2 = origens[j], destinos[j]
        if u1 == v2 or u2 == v1 or u1 == u2 or v1 == v2:
            continue
        if (u1, v2) in conjunto or (u2, v1) in conjunto:
            continue
        conjunto.remove((u1, v1))
        conjunto.remove((u2, v2))
        conjunto.add((u1, v2))
        conjunto.add((u2, v1))
        destinos[i], destinos[j] = v2, v1

    pesos_finais = rng.permutation(pesos).tolist() if embaralhar_pesos else pesos
    g_null.add_weighted_edges_from(zip(origens, destinos, pesos_finais))
    return g_null


def calibrar_parametros_sinteticos(g_simples: nx.Graph) -> Dict[str, Union[int, float]]:
    """Retorna N, M, k_medio, densidade, k_ws (par >= 2) e m_ba (>= 1) a partir de g_simples."""
    n = g_simples.number_of_nodes()
    m = g_simples.number_of_edges()
    k_medio = (2.0 * m / n) if n > 0 else 0.0
    densidade = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
    k_ws = max(2, 2 * int(round(k_medio / 2.0)))
    if k_ws >= n and n >= 3:
        k_ws = n - 1 if (n - 1) % 2 == 0 else n - 2
    m_ba = max(1, int(round(k_medio / 2.0)))
    if m_ba >= n and n >= 2:
        m_ba = n - 1
    return {
        "N": int(n),
        "M": int(m),
        "k_medio": float(k_medio),
        "densidade": float(densidade),
        "k_ws": int(k_ws),
        "m_ba": int(m_ba),
    }


def gerar_rede_sintetica(
    modelo: str,
    n: int,
    m: int,
    k_ws: int = 2,
    m_ba: int = 1,
    ws_p: float = 0.10,
    seed: int = 42,
    direcionado: bool = False,
) -> Union[nx.Graph, nx.DiGraph]:
    """Ponto único para instanciar modelos 'ER', 'WS', 'BA' e 'LATTICE'."""
    chave = modelo.upper()
    if chave == "ER":
        return nx.gnm_random_graph(n=n, m=m, seed=seed, directed=direcionado)
    if chave == "WS":
        return nx.watts_strogatz_graph(n=n, k=k_ws, p=ws_p, seed=seed)
    if chave == "LATTICE":
        return nx.watts_strogatz_graph(n=n, k=k_ws, p=0.0, seed=seed)
    if chave == "BA":
        return nx.barabasi_albert_graph(n=n, m=m_ba, seed=seed)
    raise ValueError(f"Modelo sintético desconhecido: {modelo}")


def resumir_contra_nulo(
    valor_real: float, amostra_nula: Sequence[float], ddof: int = 1
) -> Dict[str, float]:
    """Retorna {'real', 'nulo_media', 'nulo_std', 'razao', 'z_score'} com proteção contra std == 0."""
    arr = np.asarray(amostra_nula, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"real": float(valor_real), "nulo_media": 0.0, "nulo_std": 0.0, "razao": 0.0, "z_score": 0.0}
    media = float(np.mean(arr))
    std = float(np.std(arr, ddof=ddof)) if arr.size > ddof else 0.0
    razao = float(valor_real / media) if abs(media) > 1e-12 else 0.0
    z_score = float((valor_real - media) / std) if std > 1e-12 else 0.0
    return {
        "real": float(valor_real),
        "nulo_media": media,
        "nulo_std": std,
        "razao": razao,
        "z_score": z_score,
    }


def sanitizar_json(obj: Any, nan_como_none: bool = False) -> Any:
    """Converte recursivamente tipos NumPy/NaN/Inf em tipos nativos seguros para json.dump."""
    if isinstance(obj, dict):
        return {str(k): sanitizar_json(v, nan_como_none) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitizar_json(v, nan_como_none) for v in obj]
    if isinstance(obj, np.ndarray):
        return [sanitizar_json(v, nan_como_none) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None if nan_como_none else 0.0
        return val
    return obj


def salvar_artefatos(
    output_dir: str,
    csvs: Optional[Dict[str, pd.DataFrame]] = None,
    jsons: Optional[Dict[str, Any]] = None,
) -> None:
    """Salva em lote DataFrames em CSV e dicionários sanitizados em JSON dentro de output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    for nome, df in (csvs or {}).items():
        df.to_csv(os.path.join(output_dir, nome), index=False)
    for nome, dados in (jsons or {}).items():
        caminho = os.path.join(output_dir, nome)
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(sanitizar_json(dados), f, indent=2, ensure_ascii=False, allow_nan=False)
