import os
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 1. Leitura dos arquivos (ajuste o caminho se estiverem numa subpasta)
caminho_connect = "NeuronConnect.xls"
# Verifica automaticamente se a extensão salva é .xls ou .xlsx
caminho_type = "NeuronType.xls" if os.path.exists("NeuronType.xls") else "NeuronType.xlsx"

df_conn = pd.read_excel(caminho_connect)
df_type = pd.read_excel(caminho_type)

# Padroniza nomes das colunas (remove espaços extras)
df_conn.columns = [c.strip() for c in df_conn.columns]
df_type.columns = [c.strip() for c in df_type.columns]

# 2. Mapeamento funcional canônico do C. elegans (White et al. / WormAtlas)
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
INTERNEURONIOS_COMANDO = {"AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "PVCL", "PVCR"}

def classificar_funcao(nome_neuronio, tem_nmj=False):
    if nome_neuronio in INTERNEURONIOS_COMANDO:
        return "Interneurônio (Comando)"
    if nome_neuronio.startswith(PREFIXOS_FARINGEOS):
        return "Faríngeo"
    if nome_neuronio.startswith(PREFIXOS_SENSORIAIS):
        return "Sensorial"
    if nome_neuronio.startswith(PREFIXOS_MOTORES) or tem_nmj:
        return "Motor"
    return "Interneurônio"

# 3. Identifica quais neurônios possuem junções neuromusculares (NMJ) no NeuronConnect
df_conn['Neuron 1'] = df_conn['Neuron 1'].astype(str).str.strip()
df_conn['Neuron 2'] = df_conn['Neuron 2'].astype(str).str.strip()
df_conn['Type'] = df_conn['Type'].astype(str).str.strip()

nmj_por_neuronio = (
    df_conn[df_conn['Type'] == 'NMJ']
    .groupby('Neuron 1')['Nbr']
    .sum()
    .to_dict()
)

# 4. Instancia o MultiDiGraph e vincula os atributos do NeuronType.xls aos nós
G_multi = nx.MultiDiGraph()

# Colunas do NeuronType.xls (trata variações de maiúsculas/minúsculas)
col_soma_reg = 'Soma region' if 'Soma region' in df_type.columns else 'Soma Region'

for _, row in df_type.iterrows():
    neuronio = str(row['Neuron']).strip()
    peso_nmj = int(nmj_por_neuronio.get(neuronio, 0))
    
    G_multi.add_node(
        neuronio,
        tipo_funcional=classificar_funcao(neuronio, tem_nmj=(peso_nmj > 0)),
        soma_pos=float(row.get('Soma Position', 0.0)),
        regiao_soma=str(row.get(col_soma_reg, 'N/A')).strip(),  # H (Cabeça), M (Meio), T (Cauda)
        span=str(row.get('Span', 'N/A')).strip(),              # S (Curto) ou L (Longo)
        ganglio_ay=str(row.get('AY NeuronType', 'N/A')).strip(),
        peso_nmj=peso_nmj
    )

# 5. Adiciona as arestas filtrando apenas Envios Químicos ('S', 'Sp') e Junções Elétricas ('EJ')
df_arestas = df_conn[df_conn['Type'].isin(['S', 'Sp', 'EJ'])]

for _, row in df_arestas.iterrows():
    u = row['Neuron 1']
    v = row['Neuron 2']
    peso = int(row['Nbr'])
    tipo_cod = row['Type']
    
    # Ignora auto-laços ou conexões com alvos não-neuronais
    if u == v or v == 'NMJ':
        continue
        
    # Garante que se houver algum neurônio no Connect ausente no NeuronType, ele receba classificação
    for no in (u, v):
        if no not in G_multi:
            peso_m = int(nmj_por_neuronio.get(no, 0))
            G_multi.add_node(
                no,
                tipo_funcional=classificar_funcao(no, tem_nmj=(peso_m > 0)),
                soma_pos=None, regiao_soma='N/A', span='N/A', ganglio_ay='N/A', peso_nmj=peso_m
            )

    # Define a camada do multigrafo: 'Quimica' (soma S + Sp) ou 'Eletrica' (EJ)
    camada = "Eletrica" if tipo_cod == "EJ" else "Quimica"
    
    if G_multi.has_edge(u, v, key=camada):
        G_multi[u][v][camada]['weight'] += peso
    else:
        G_multi.add_edge(u, v, key=camada, weight=peso, synapse_type=camada)

# 6. Resumo da estruturação
print(f"Total de Neurônios (Vértices): {G_multi.number_of_nodes()}")
print(f"Total de Conexões (Multi-arestas): {G_multi.number_of_edges()}")

# Contagem por tipo funcional
contagem_tipos = pd.Series(nx.get_node_attributes(G_multi, 'tipo_funcional')).value_counts()
print("\n--- Distribuição dos Tipos de Neurônios ---")
print(contagem_tipos)

# Exemplo de inspeção de um nó específico (AVAL)
print("\n--- Atributos vinculados ao neurônio AVAL ---")
print(G_multi.nodes['AVAL'])

# 1. Extrai todas as conexões do MultiDiGraph com os tipos de origem e destino
registros = []
for u, v, camada, dados in G_multi.edges(keys=True, data=True):
    tipo_u = G_multi.nodes[u].get('tipo_funcional', 'Desconhecido')
    tipo_v = G_multi.nodes[v].get('tipo_funcional', 'Desconhecido')
    peso = dados.get('weight', 1)
    
    registros.append({
        'Origem': tipo_u,
        'Destino': tipo_v,
        'Camada': camada,      # 'Quimica' ou 'Eletrica'
        'Peso_Sinaptico': peso,
        'Arestas': 1
    })

df_fluxo = pd.DataFrame(registros)

# Se preferir agrupar "Interneurônio (Comando)" dentro de "Interneurônio" para ter apenas 3 classes
# somáticas (Sensorial, Interneurônio, Motor), descomente as duas linhas abaixo:
# df_fluxo['Origem'] = df_fluxo['Origem'].replace({'Interneurônio (Comando)': 'Interneurônio'})
# df_fluxo['Destino'] = df_fluxo['Destino'].replace({'Interneurônio (Comando)': 'Interneurônio'})

# Define a ordem lógica do processamento neural nos eixos da matriz
ordem_classes = [
    c for c in ["Sensorial", "Interneurônio", "Interneurônio (Comando)", "Motor", "Faríngeo"]
    if c in df_fluxo['Origem'].unique() or c in df_fluxo['Destino'].unique()
]

# 2. Função para gerar a matriz de fluxo (por Peso Sináptico ou Número de Arestas)
def criar_matriz_fluxo(df, camada, metrica='Peso_Sinaptico'):
    sub = df[df['Camada'] == camada]
    matriz = pd.pivot_table(
        sub,
        values=metrica,
        index='Origem',
        columns='Destino',
        aggfunc='sum',
        fill_value=0
    )
    # Reordena linhas e colunas seguindo a hierarquia biológica
    return matriz.reindex(index=ordem_classes, columns=ordem_classes, fill_value=0)

mat_quimica = criar_matriz_fluxo(df_fluxo, 'Quimica', metrica='Peso_Sinaptico')
mat_eletrica = criar_matriz_fluxo(df_fluxo, 'Eletrica', metrica='Peso_Sinaptico')

# Exibe as tabelas numéricas no terminal/notebook
print("=== MATRIZ DE FLUXO: SINAPSES QUÍMICAS (Peso Total: Origem -> Destino) ===")
print(mat_quimica, "\n")

print("=== MATRIZ DE ACOPLAMENTO: JUNÇÕES ELÉTRICAS (Peso Total: Origem <-> Destino) ===")
print(mat_eletrica, "\n")

# 3. Plotagem comparativa (Heatmaps com valor absoluto + porcentagem por linha)
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

matrizes = [
    (mat_quimica, "Camada Química (Fluxo Direcionado: Origem → Destino)", "YlOrRd"),
    (mat_eletrica, "Camada Elétrica / Gap Junctions (Acoplamento Simétrico)", "PuBu")
]

for ax, (mat, titulo, cmap) in zip(axes, matrizes):
    # Calcula a porcentagem por linha (para onde vai o sinal que sai de cada classe)
    soma_linhas = mat.sum(axis=1).replace(0, 1)
    mat_pct = mat.div(soma_linhas, axis=0) * 100

    im = ax.imshow(mat.values, cmap=cmap, aspect='auto')
    
    # Configuração dos eixos
    ax.set_xticks(np.arange(len(ordem_classes)))
    ax.set_yticks(np.arange(len(ordem_classes)))
    ax.set_xticklabels(ordem_classes, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(ordem_classes, fontsize=10)
    ax.set_xlabel("Neurônio de Destino (Pós-sináptico)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Neurônio de Origem (Pré-sináptico)", fontsize=11, fontweight='bold')
    ax.set_title(titulo, fontsize=12, fontweight='bold', pad=12)

    # Anota cada célula com o Peso Total e a % da linha
    limiar_cor = mat.values.max() / 2.0
    for i in range(len(ordem_classes)):
        for j in range(len(ordem_classes)):
            valor = mat.values[i, j]
            pct = mat_pct.values[i, j]
            cor_texto = "white" if valor > limiar_cor else "black"
            ax.text(
                j, i, f"{valor}\n({pct:.1f}%)",
                ha="center", va="center", color=cor_texto, fontsize=9, fontweight='bold'
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Volume Sináptico Acumulado")

plt.tight_layout()
plt.show()