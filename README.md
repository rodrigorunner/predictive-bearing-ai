# Predictive Maintenance AI - Neural Network for Bearing Analysis

Projeto desenvolvido em Python utilizando NumPy para simular uma análise preditiva aplicada à manutenção industrial.

A proposta do projeto é utilizar uma rede neural simples para analisar condições operacionais de um rolamento através de três grandezas principais:

* RPM
* Temperatura
* Vibração

Os dados são normalizados entre 0.0 e 1.0 para facilitar o treinamento da rede neural.

## Objetivo

Simular o comportamento de um sistema inteligente capaz de identificar níveis de severidade operacional com base nos sinais coletados dos sensores instalados em equipamentos rotativos.

A rede neural classifica os dados em três estados:

* NORMAL
* ALERTA
* CRÍTICO

## Tecnologias utilizadas

* Python
* NumPy
* Redes Neurais
* Machine Learning básico
* Função Sigmoid

## Funcionamento

A rede neural recebe dados simulados de sensores industriais e realiza o treinamento ajustando automaticamente os pesos através do cálculo do erro.

As variáveis analisadas representam condições comuns monitoradas em manutenção preditiva industrial.

## Conceitos aplicados

* Normalização de dados
* Treinamento supervisionado
* Ajuste de pesos
* Função de ativação sigmoid
* Classificação de severidade
* Simulação de manutenção preditiva

## Aplicação industrial

O projeto foi inspirado em processos reais de análise preditiva utilizados na indústria para monitoramento de:

* Rolamentos
* Motores elétricos
* Bombas
* Mancais
* Equipamentos rotativos

## Próximos passos

* Adicionar mais sensores
* Trabalhar com datasets maiores
* Implementar gráficos
* Criar dashboard de monitoramento
* Utilizar TensorFlow ou PyTorch
* Aplicar detecção de anomalias
* Simular dados reais industriais

## Autor

Rodrigo Regis

