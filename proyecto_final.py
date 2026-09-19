import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo gráfico
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------
# ETAPA 1: GENERACIÓN Y LIMPIEZA DE DATOS (MÓDULO 1)
# -------------------------------------------------------------
np.random.seed(42)
n_samples = 200

tipos_equipo = ['Laptop', 'Desktop', 'Servidor', 'Impresora', 'Smartphone']
tipos_falla = ['Software', 'Hardware Pantalla', 'Placa Madre', 'Mantenimiento General']
niveles_tecnico = ['Junior', 'Semi-Senior', 'Senior']

# Generación del dataset inicial
df = pd.DataFrame({
    'ID_Equipo': [f'EQ-{1000+i}' for i in range(n_samples)],
    'Tipo_Equipo': np.random.choice(tipos_equipo, n_samples, p=[0.35, 0.25, 0.10, 0.15, 0.15]),
    'Tipo_Falla': np.random.choice(tipos_falla, n_samples, p=[0.40, 0.25, 0.15, 0.20]),
    'Nivel_Experiencia_Tecnico': np.random.choice(niveles_tecnico, n_samples, p=[0.40, 0.40, 0.20]),
    'Piezas_Reemplazadas': np.random.poisson(lam=1.2, size=n_samples),
    'Garantia_Activa': np.random.choice(['Sí', 'No'], n_samples, p=[0.3, 0.7])
})

# Creación lógica del tiempo de reparación (Variable Objetivo)
base_time = np.where(df['Tipo_Falla'] == 'Placa Madre', 18, 
             np.where(df['Tipo_Falla'] == 'Hardware Pantalla', 8,
             np.where(df['Tipo_Falla'] == 'Software', 4, 3)))

tec_factor = np.where(df['Nivel_Experiencia_Tecnico'] == 'Junior', 1.5,
              np.where(df['Nivel_Experiencia_Tecnico'] == 'Semi-Senior', 1.0, 0.7))

df['Tiempo_Reparacion_Horas'] = np.round(np.maximum(1, base_time * tec_factor + df['Piezas_Reemplazadas'] * 2 + np.random.normal(0, 2, n_samples)), 1)
df['Costo_Repuesto_USD'] = np.where(df['Piezas_Reemplazadas'] == 0, 0, df['Piezas_Reemplazadas'] * np.random.uniform(20, 80, n_samples))

# Introduciendo fallas a propósito para la demostración de limpieza
mask_nulls = np.random.choice([True, False], n_samples, p=[0.08, 0.92])
df.loc[mask_nulls, 'Costo_Repuesto_USD'] = np.nan
df.loc[df['Tipo_Equipo'] == 'Laptop', 'Tipo_Equipo'] = np.random.choice(['Laptop', 'laptop', 'Lap-top'], sum(df['Tipo_Equipo'] == 'Laptop'))

# PROCESO DE LIMPIEZA
df_clean = df.drop_duplicates().copy()
df_clean['Tipo_Equipo'] = df_clean['Tipo_Equipo'].replace({'laptop': 'Laptop', 'Lap-top': 'Laptop'})
df_clean['Costo_Repuesto_USD'] = df_clean['Costo_Repuesto_USD'].fillna(
    df_clean.groupby('Piezas_Reemplazadas')['Costo_Repuesto_USD'].transform('median')
).fillna(0)

# Guardar los archivos de datos
df_clean.to_csv('dataset_servicio_tecnico_limpio.csv', index=False)

# -------------------------------------------------------------
# ETAPA 2: ANÁLISIS ESTADÍSTICO DESCRIPTIVO (MÓDULO 2)
# -------------------------------------------------------------
media = df_clean['Tiempo_Reparacion_Horas'].mean()
mediana = df_clean['Tiempo_Reparacion_Horas'].median()
desviacion = df_clean['Tiempo_Reparacion_Horas'].std()
iqr = df_clean['Tiempo_Reparacion_Horas'].quantile(0.75) - df_clean['Tiempo_Reparacion_Horas'].quantile(0.25)

print("=== RESUMEN ESTADÍSTICO ===")
print(f"Media (Promedio): {media:.2f} horas")
print(f"Mediana (Centro): {mediana:.2f} horas")
print(f"Desviación Estándar (Dispersión): {desviacion:.2f} horas")
print(f"Rango Intercuartílico (IQR): {iqr:.2f} horas")

# -------------------------------------------------------------
# ETAPA 3: VISUALIZACIÓN DE RESULTADOS (MÓDULO 3)
# -------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Histograma
sns.histplot(df_clean['Tiempo_Reparacion_Horas'], kde=True, ax=axes[0], color='#2b5c8f')
axes[0].axvline(media, color='red', linestyle='--', label=f'Media: {media:.1f}h')
axes[0].axvline(mediana, color='green', linestyle='-', label=f'Mediana: {mediana:.1f}h')
axes[0].set_title('1. Distribución del Tiempo de Reparación')
axes[0].legend()

# Gráfico 2: Boxplot Comparativo
sns.boxplot(data=df_clean, x='Tipo_Falla', y='Tiempo_Reparacion_Horas', hue='Nivel_Experiencia_Tecnico', ax=axes[1], palette='Set2')
axes[1].set_title('2. Tiempo por Falla y Experiencia del Técnico')
axes[1].tick_params(axis='x', rotation=15)

# Gráfico 3: Dispersión
sns.scatterplot(data=df_clean, x='Piezas_Reemplazadas', y='Tiempo_Reparacion_Horas', hue='Tipo_Falla', ax=axes[2], s=70)
axes[2].set_title('3. Impacto de Piezas Reemplazadas en el Tiempo')

plt.tight_layout()
plt.savefig('analisis_servicio_tecnico.png', dpi=300)
plt.show()
