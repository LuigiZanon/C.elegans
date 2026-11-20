"""
Módulo de configurações, constantes biológicas e parâmetros visuais do conectoma C. elegans.
"""

OUTPUT_DIR = "outputs"

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
