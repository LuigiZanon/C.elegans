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

PREFIXOS_FARINGEOS = (
    "I1", "I2", "I3", "I4", "I5", "I6", "M1", "M2", "M3", "M4", "M5", "MC", "MI", "NSM"
)

INTERNEURONIOS_COMANDO = {
    "AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "PVCL", "PVCR"
}

ORDEM_CLASSES_PADRAO = [
    "Sensorial",
    "Interneurônio",
    "Interneurônio (Comando)",
    "Motor",
    "Faríngeo"
]

PALETA_CORES_FUNCIONAIS = {
    "Sensorial": "#4285F4",               # Azul
    "Interneurônio": "#FBBC05",           # Amarelo
    "Interneurônio (Comando)": "#EA4335", # Vermelho
    "Motor": "#34A853",                   # Verde
    "Faríngeo": "#9C27B0"                 # Roxo
}

# Coordenadas (X, Y) dos centros de cada cluster funcional na tela
CENTROS_GRUPOS_FUNCIONAIS = {
    "Sensorial": (-950, -120),             # Esquerda (Entrada sensorial)
    "Interneurônio": (0, -620),            # Topo-Centro (Processamento)
    "Interneurônio (Comando)": (0, 80),    # Centro (Hubs de comando)
    "Motor": (950, -120),                  # Direita (Execução motora)
    "Faríngeo": (-750, 600),               # Inferior-Esquerda (Faringe)
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
