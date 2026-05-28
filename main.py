import numpy as np 
# =========================================
# Entrada dos dados
# [RPM, Temperatura e Vibração]
# Valores normalizados entre 0.0 e 1.0
# =========================================
entrada = np.array([
    [0.30, 0.25, 0.20], # Normal
    [0.40, 0.35, 0.30], # Normal    
    [0.65, 0.60, 0.55], # Alerta 
    [0.75, 0.70, 0.80], # Alerta 
    [0.95, 0.90, 0.95], # Crítico
])

# ============================================
# SAÍDA ESPERADA
# 0.0 → 0.5 = NORMAL
# 0.6 → 0.8 = ALERTA
# 0.9 → 1.0 = CRÍTICO
# ============================================
saida_esperada = np.array([
    [0.20], # Normal
    [0.30], # Normal
    [0.65], # Alerta
    [0.75], # Alerta
    [0.95], # Crítico
])

# ============================================
# Pesos Aleatórios
# ============================================
# Setar um peso aleatório fixo.
np.random.seed(1)
pesos = np.random.random((3, 1))

# ============================================
# Função Sigmoid
# Converte os valores para uma faixa entre 0 e 1.
# Função derivada calcula o ajuste final.
# ============================================
def sigmoid(valor):
    return 1 / (1 + np.exp(-valor))

def derivada(valor):
    return valor * (1 - valor)

# ============================================
# Fase de treinamento
# ============================================
print("\n" + "=" * 50)
print("INICIANDO TREINAMENTO DA REDE NEURAL")
print("=" * 50)

for epoca in range(10000):

    # Processamento
    calc = sigmoid(np.dot(entrada, pesos))

    # Erro
    erro = saida_esperada - calc

    # Ajuste
    ajuste = erro * derivada(calc)

    # Atualização dos pesos
    pesos += np.dot(entrada.T, ajuste)

    # Exibir progresso do treinamento
    if epoca % 1000 == 0:

        erro_medio = np.mean(np.abs(erro))
        percentual_acerto = (1 - erro_medio) * 100

        print("\n" + "-" * 50)
        print(f"Treinamento atual : {epoca}")
        print(f"Erro médio        : {erro_medio:.6f}")
        print(f"Precisão estimada : {percentual_acerto:.2f}%")
        print("-" * 50)

# ============================================
# Resultado final da rede
# ============================================
print("\n" + "=" * 50)
print("RESULTADO FINAL DAS AMOSTRAS")
print("=" * 50)

for indice, valor in enumerate(calc):

    if valor[0] < 0.5:
        status = "NORMAL"

    elif valor[0] < 0.8:
        status = "ALERTA"

    else:
        status = "CRÍTICO"

    print(
        f"Amostra {indice + 1:>2} | "
        f"Probabilidade: {valor[0]:.6f} | "
        f"Status: {status}"
    )

print("=" * 50)

# ============================================
# Teste de novo rolamento
# ============================================
print("\nNOVO TESTE DE ROLAMENTO")
print("-" * 50)

teste = np.array([[0.55, 0.45, 0.45]])

resultado = sigmoid(np.dot(teste, pesos))

valor = resultado[0][0]

print(f"Leitura sensores : {teste[0]}")
print(f"Resultado da IA  : {valor:.6f}")

# Diagnóstico final
if valor < 0.5:
    status = "NORMAL"

elif valor < 0.8:
    status = "ALERTA"

else:
    status = "CRÍTICO"

print(f"Status final: {status}")
print("-" * 50)



