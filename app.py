import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Dashboard Financiero & EDA",
    page_icon="📊",
    layout="wide"
)

# --- 1. GENERACIÓN DE DATOS SINTÉTICOS ---
@st.cache_data # Mantiene los datos en caché para que no cambien en cada clic
def generar_datos_financieros(dias=365):
    np.random.seed(42) # Semilla para reproducibilidad
    fechas = [datetime.now() - timedelta(days=x) for x in range(dias)]
    fechas.reverse()
    
    # Activos financieros sintéticos
    activos = ['Acción_A', 'Acción_B', 'Acción_C', 'Crypto_X']
    precios_iniciales = [100, 250, 50, 1000]
    volatilidades = [0.015, 0.02, 0.025, 0.05] # Volatilidad diaria
    
    data = []
    for fecha in fechas:
        fila = {'Fecha': fecha.strftime('%Y-%m-%d')}
        for activo, precio_ini, vol in zip(activos, precios_iniciales, volatilidades):
            # Movimiento Browniano Geométrico simple para simular precios de acciones
            retorno = np.random.normal(0.0005, vol) # media retorno diario ligero positivo
            precio_actual = precio_ini * (1 + retorno)
            fila[activo] = round(precio_actual, 2)
        data.append(fila)
        # Actualizar precio inicial para el siguiente día
        precios_iniciales = [fila[a] for a in activos]
        
    df = pd.DataFrame(data)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    # Añadir variables Cualitativas Sintéticas
    sectores = {
        'Acción_A': 'Tecnología',
        'Acción_B': 'Finanzas',
        'Acción_C': 'Energía',
        'Crypto_X': 'Blockchain'
    }
    
    # Pivotar a formato largo para facilitar análisis cuali-cuanti
    df_long = df.melt(id_vars=['Fecha'], var_name='Activo', value_name='Precio')
    df_long['Sector'] = df_long['Activo'].map(sectores)
    
    # Calcular retornos diarios como variable cuantitativa adicional
    df_long['Retorno_Diario'] = df_long.groupby('Activo')['Precio'].pct_change()
    # Categorizar el rendimiento diario (Cualitativa)
    df_long['Rendimiento_Cat'] = pd.cut(df_long['Retorno_Diario'], 
                                       bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                                       labels=['Caída Fuerte', 'Baja', 'Estable', 'Alza', 'Subida Fuerte'])
    
    return df_long.dropna()

# Cargar datos
df_finanzas = generar_datos_financieros()

# --- 2. INTERFAZ DE USUARIO & INTERACCIÓN ---
st.title("📊 Dashboard de Análisis Financiero (EDA)")
st.markdown("Esta plataforma genera datos financieros sintéticos y permite explorar variables cuantitativas y cualitativas de manera interactiva.")

# Sidebar (Barra lateral para filtros)
st.sidebar.header("Filtros Interactivos")

# Filtro 1: Selección de Activos
activos_disponibles = df_finanzas['Activo'].unique()
activos_seleccionados = st.sidebar.multiselect(
    "Selecciona los Activos a analizar:",
    options=activos_disponibles,
    default=activos_disponibles
)

# Filtro 2: Rango de Fechas
fecha_min = df_finanzas['Fecha'].min().to_pydatetime()
fecha_max = df_finanzas['Fecha'].max().to_pydatetime()
rango_fechas = st.sidebar.date_input(
    "Selecciona el rango de fechas:",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max
)

# Aplicar filtros a los datos
if len(rango_fechas) == 2 and len(activos_seleccionados) > 0:
    inicio, fin = rango_fechas
    df_filtrado = df_finanzas[
        (df_finanzas['Activo'].isin(activos_seleccionados)) & 
        (df_finanzas['Fecha'] >= pd.to_datetime(inicio)) & 
        (df_finanzas['Fecha'] <= pd.to_datetime(fin))
    ]
else:
    df_filtrado = df_finanzas.copy()
    st.sidebar.warning("Por favor, selecciona al menos un activo y un rango de fechas válido.")

# --- 3. SECCIÓN DE MÉTRICAS CLAVE (KPIs) ---
st.subheader("📈 Resumen del Mercado")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Registros Analizados", value=len(df_filtrado))
with col2:
    retorno_promedio = df_filtrado['Retorno_Diario'].mean() * 100
    st.metric(label="Retorno Diario Promedio", value=f"{retorno_promedio:.3f}%")
with col3:
    activo_mas_volatil = df_filtrado.groupby('Activo')['Retorno_Diario'].std().idxmax()
    st.metric(label="Activo Más Volátil", value=activo_mas_volatil)

# --- 4. ANÁLISIS EXPLORATORIO DE DATOS (EDA) ---
tab1, tab2, tab3 = st.tabs(["📊 Análisis Cuantitativo", "🎭 Análisis Cualitativo", "📁 Vista de Datos"])

# TAB 1: Análisis Cuantitativo (Precios y Tendencias)
with tab1:
    st.subheader("Evolución Temporal de Precios (Cuantitativa vs Cuantitativa)")
    fig_lineas = px.line(df_filtrado, x='Fecha', y='Precio', color='Activo', title="Evolución del Precio en el Tiempo")
    st.plotly_chart(fig_lineas, use_container_width=True)
    
    st.subheader("Distribución de Retornos Diarios (Volatilidad)")
    fig_hist = px.histogram(df_filtrado, x='Retorno_Diario', color='Activo', marginal="box", 
                            title="Histograma de Retornos Diarios", barmode="overlay")
    st.plotly_chart(fig_hist, use_container_width=True)

# TAB 2: Análisis Cualitativo y Cuali-Cuanti
with tab2:
    col_cuali1, col_cuali2 = st.columns(2)
    
    with col_cuali1:
        st.subheader("Distribución por Sectores (Cualitativa)")
        # Contar cuántas muestras pertenecen a cada sector en los datos filtrados
        fig_pastel = px.pie(df_filtrado.drop_duplicates(subset=['Activo']), names='Sector', title='Composición del Portafolio por Sector')
        st.plotly_chart(fig_pastel, use_container_width=True)
        
    with col_cuali2:
        st.subheader("Comportamiento Diario por Categoría (Cuali-Cuanti)")
        # Frecuencia de tipos de rendimiento por activo
        fig_barras = px.histogram(df_filtrado, x="Activo", color="Rendimiento_Cat", 
                                  title="Frecuencia de Tipos de Rendimiento", barmode="group")
        st.sidebar.markdown("---")
        st.plotly_chart(fig_barras, use_container_width=True)
        
    st.subheader("Precio Promedio por Sector y Activo (Cuali-Cuanti Plot)")
    fig_box = px.box(df_filtrado, x="Sector", y="Precio", color="Activo", title="Diagrama de Caja de Precios por Sector")
    st.plotly_chart(fig_box, use_container_width=True)

# TAB 3: Tabla Interactiva y Descarga
with tab3:
    st.subheader("Datos Filtrados")
    st.dataframe(df_filtrado, use_container_width=True)
    
    # Botón para descargar los datos que el usuario filtró
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar datos filtrados como CSV",
        data=csv,
        file_name='datos_financieros_filtrados.csv',
        mime='text/csv',
    )
