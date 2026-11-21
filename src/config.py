"""
Módulo de configurações, constantes biológicas e parâmetros visuais do conectoma C. elegans.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONNECT_DATA_PATH = os.path.join(PROJECT_ROOT, "NeuronConnect.xls")
TYPE_DATA_PATH_XLS = os.path.join(PROJECT_ROOT, "NeuronType.xls")
TYPE_DATA_PATH_XLSX = os.path.join(PROJECT_ROOT, "NeuronType.xlsx")

PREFIXOS_SENSORIAIS = (
    "ADF", "ADL", "AFD", "ALM", "ALN", "AQR", "ASE", "ASG", "ASH", "ASI",
    "ASJ", "ASK", "AVM", "AWA", "AWB", "AWC", "BAG", "CEP", "FLP", "IL1",
    "IL2", "OLL", "OLQ", "PHA", "PHB", "PHC", "PLM", "PLN", "PQR", "PVD",
    "PVM", "URB", "URX", "URY"
)

PREFIXOS_MOTORES = (
    "AS0", "AS1", "DA0", "DA1", "DB0", "DD0", "VA0", "VA1", "VB0", "VB1",
    "VC0", "VD0", "VD1", "RMD", "RME", "RMF", "RMH", "SIA", "SIB", "SMB",
    "SMD", "URA", "PDA", "PDB", "DVB", "HSN"
)


INTERNEURONIOS_COMANDO = {
    "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "PVCL", "PVCR"
}

ORDEM_CLASSES_PADRAO = [
    "Sensorial",
    "Interneurônio",
    "Interneurônio (Comando)",
    "Motor",
]

PALETA_CORES_FUNCIONAIS = {
    "Sensorial": "#4285F4",               # Azul
    "Interneurônio": "#FBBC05",           # Amarelo
    "Interneurônio (Comando)": "#EA4335", # Vermelho
    "Motor": "#34A853"                   # Verde
}

# Coordenadas (X, Y) dos centros de cada cluster funcional na tela (Modo 1: Clusters)
CENTROS_GRUPOS_FUNCIONAIS = {
    "Sensorial": (-950, -120),             # Esquerda (Entrada sensorial)
    "Interneurônio": (0, -620),            # Topo-Centro (Processamento)
    "Interneurônio (Comando)": (0, 80),    # Centro (Hubs de comando)
    "Motor": (950, -120),                  # Direita (Execução motora)
    "Desconhecido": (750, 600)
}

PYVIS_PHYSICS_OPTIONS = """
var options = {
  "nodes": {
    "font": {
      "size": 13,
      "face": "sans-serif"
    }
  },
  "edges": {
    "smooth": false,
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 0.45
      }
    }
  },
  "physics": {
    "enabled": true,
    "repulsion": {
      "centralGravity": 0.0,
      "springLength": 25,
      "springConstant": 0.045,
      "nodeDistance": 75,
      "damping": 0.85
    },
    "maxVelocity": 20,
    "minVelocity": 0.75,
    "solver": "repulsion",
    "stabilization": {
      "enabled": true,
      "iterations": 200,
      "updateInterval": 25,
      "fit": true
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 80,
    "hideEdgesOnDrag": true,
    "multiselect": true
  }
}
"""

SYNTHETIC_BENCHMARK_CONFIG = {
    "n_runs": 30,             # Quantidade de execuções Monte Carlo por modelo sintético
    "seed_base": 42,          # Semente determinística base (execução i usa seed_base + i)
    "ws_rewiring_p": 0.10,    # Probabilidade de religação canônica para Watts-Strogatz
}

PALETA_MODELOS_SINTETICOS = {
    "C. elegans (Real)": "#EA4335",   # Vermelho destaque
    "Erdős-Rényi (ER)": "#757575",    # Cinza neutro (aleatório)
    "Watts-Strogatz (WS)": "#34A853", # Verde (Small-World)
    "Barabási-Albert (BA)": "#4285F4" # Azul (Livre de Escala)
}

TOPOLOGY_CLASSIFICATION_CONFIG = {
    "n_runs_null": 30,          # Redes nulas ER para Z-scores e intervalos de confiança
    "n_runs_ws_sweep": 15,      # Repetições por ponto na curva de transição de Watts-Strogatz
    "n_bootstrap_ks": 200,      # Réplicas semi-paramétricas para p-valor KS (Clauset et al.)
    "min_tail_size": 30,        # Mínimo de neurônios exigidos na cauda k >= k_min
    "seed_base": 42,            # Semente determinística global
}

PALETA_MODELOS_CAUDA = {
    "Empírico (C. elegans)": "#111111",
    "Lei de Potência (PL)": "#EA4335",
    "PL Truncada (TPL)": "#34A853",
    "Log-Normal (LN)": "#4285F4",
    "Exponencial (EXP)": "#FBBC05",
    "Poisson / ER": "#9E9E9E",
}

NAVIGATION_ASSORTATIVITY_CONFIG = {
    "n_runs_null": 20,           # Repetições para modelos nulos (espacial e rewiring)
    "n_pares_nulo": 5000,        # Pares alcançáveis amostrados por rodada nula (alta velocidade)
    "max_hops_fator": 3,         # Limite de saltos = max(25, max_hops_fator * diametro)
    "alpha_hibrido": 0.55,       # Peso do termo espacial na estratégia híbrida (0..1)
    "seed_base": 42,             # Semente determinística global
}

PALETA_ESTRATEGIAS_NAVEGACAO = {
    "Espacial (soma_pos)": "#4285F4",       # Azul
    "Grau (Hubs)": "#EA4335",               # Vermelho
    "Híbrido (Espacial + Grau)": "#34A853", # Verde
    "Caminhada Aleatória": "#9E9E9E",       # Cinza
}

PRESERVATION_ANALYSIS_CONFIG = {
    "n_runs_null": 30,             # Quantidade de redes nulas direcionadas (Double-Edge Swap)
    "swaps_fator": 3,              # Trocas por rodada nula = max(1500, swaps_fator * M)
    "louvain_resolution": 1.0,     # Parâmetro de resolução para detecção de comunidades
    "seed_base": 42,               # Semente determinística global
}

PALETA_BOWTIE_PRESERVACAO = {
    "SCC (Núcleo Recorrente)": "#EA4335", # Vermelho (preservação / reverberação)
    "IN (Entrada / Feeders)": "#4285F4",  # Azul (injeta sinal no núcleo)
    "OUT (Saída / Sinks)": "#34A853",     # Verde (recebe do núcleo sem retornar)
    "Outros (Isolados/Tubes)": "#9E9E9E", # Cinza (periferia desconexa da SCC)
}

DIFFUSION_ANALYSIS_CONFIG = {
    "n_runs_er_estrutural": 20,      # Redes ER equivalentes para L, D e E_glob
    "n_runs_si": 150,                # Rodadas Monte Carlo por cenário no modelo SI
    "n_runs_ic": 150,                # Rodadas Monte Carlo por (cenário, beta) na Cascata Independente
    "beta_si_padrao": 0.08,          # Probabilidade de transmissão por passo no modelo SI
    "max_steps_si": 25,              # Horizonte temporal máximo do modelo SI
    "betas_ic": [0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, 0.35],  # Grade da curva de fase IC/SIR
    "beta_ic_referencia": 0.08,      # Probabilidade de destaque para comparação direta
    "seed_base": 42,                 # Semente determinística global
}

PALETA_CENARIOS_DIFUSAO = {
    "C. elegans (Hubs / Rich-Club)": "#EA4335", # Vermelho
    "C. elegans (Sensorial)": "#4285F4",        # Azul
    "C. elegans (Aleatório)": "#FBBC05",        # Amarelo / Âmbar
    "Erdős-Rényi (Top Hubs)": "#34A853",        # Verde
    "Erdős-Rényi (Aleatório)": "#757575",       # Cinza
}

ROBUSTNESS_ANALYSIS_CONFIG = {
    "n_runs_aleatorio": 30,        # Repetições Monte Carlo para falhas aleatórias no C. elegans
    "n_runs_sinteticas": 15,       # Instâncias por modelo sintético (ER, WS, BA)
    "ws_rewiring_p": 0.10,         # Probabilidade de religação para Watts-Strogatz
    "passo_eficiencia": 5,         # Avaliar E_glob a cada N passos (mantendo 0..10 e N) para alta velocidade
    "limiar_colapso_50": 0.50,     # Fração S(q) <= 0.50 para f_{c,50}
    "limiar_colapso_20": 0.20,     # Fração S(q) <= 0.20 para f_{c,20}
    "seed_base": 42,               # Semente determinística global
}

WIRING_NULL_CONFIG = {
    "n_runs_null": 50,             # Permutações aleatórias de soma_pos mantendo topologia e pesos fixos
    "seed_base": 42,
}

PALETA_REGIMES_ROBUSTEZ = {
    "Falha Aleatória": "#34A853",                    # Verde
    "Ataque Estático (Grau)": "#FBBC05",             # Âmbar
    "Ataque Adaptativo (Grau)": "#EA4335",           # Vermelho
    "Ataque Estático (Intermediação)": "#4285F4",    # Azul
    "Ataque Adaptativo (Intermediação)": "#9C27B0",  # Roxo
    "Rich-Club -> Grau Adaptativo": "#FF6D00",       # Laranja intenso
}




