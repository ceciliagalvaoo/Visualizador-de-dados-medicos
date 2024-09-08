import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 1. Carregar os dados
df = pd.read_csv('medical_examination.csv')
# 2. Criar a coluna overweight
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)
# 3. Normalizar os dados
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else 0)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else 0)
# 4. Função para desenhar o gráfico categórico
def draw_cat_plot():
    # 5. Criar DataFrame categórico
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    # 6. Agrupar os dados e ajustar o formato
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    # 7. Criar o gráfico categórico
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar").fig
    # 8. Salvar e retornar o gráfico
    fig.savefig('catplot.png')
    return fig
# 10. Função para desenhar o heatmap
def draw_heat_map():
    # 11. Limpar os dados
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]
    # 12. Calcular a matriz de correlação
    corr = df_heat.corr()
    # 13. Gerar máscara para o triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # 14. Configurar o gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    # 15. Desenhar o heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=0.5, ax=ax, cmap='coolwarm')
    # 16. Salvar e retornar o gráfico
    fig.savefig('heatmap.png')
    return fig