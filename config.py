"""
Módulo de configurações, constantes biológicas e parâmetros visuais do conectoma C. elegans.
"""

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
    "forceAtlas2Based": {
      "gravitationalConstant": -120,
      "centralGravity": 0.015,
      "springLength": 150,
      "springConstant": 0.02,
      "damping": 0.85,
      "avoidOverlap": 0.6
    },
    "maxVelocity": 25,
    "minVelocity": 1.2,
    "solver": "forceAtlas2Based",
    "stabilization": {
      "enabled": true,
      "iterations": 300,
      "updateInterval": 25,
      "fit": true
    }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "hideEdgesOnDrag": true,
    "multiselect": true
  }
}
"""
