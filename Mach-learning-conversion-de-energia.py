import numpy as np
import pandas as pd
import time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN TÉCNICA CH4 ---
n_moles = 5000
R = 8.314
a_ch4 = 0.2283         
b_ch4 = 4.27e-5        
masa_molar_ch4 = 0.01604 
cp_ch4 = 35.05  # J/(mol·K) a 300K
cv_ch4 = 26.73  # J/(mol·K)
gamma = 1.31    # Cp/Cv

# --- PREPARACIÓN DEL MODELO IA ---
def generar_datos_scada(muestras=7000):
    v = np.random.uniform(0.3, 2.5, muestras)
    t = np.random.uniform(280, 650, muestras)
    p = (n_moles * R * t) / (v - n_moles * b_ch4) - (a_ch4 * n_moles**2 / v**2)
    return pd.DataFrame({'Volumen': v, 'Temp': t, 'Presion': p})

df = generar_datos_scada()
scaler = StandardScaler()
X = scaler.fit_transform(df[['Volumen', 'Presion']])
y = df['Temp']
ia_engine = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000).fit(X, y)

def scada_monitor_avanzado(v_sensor, p_consigna, flujo_m3_s):
    start_time = time.time()
    
    # 1. Inferencia de IA
    entrada_scaled = scaler.transform([[v_sensor, p_consigna]])
    t_pred = ia_engine.predict(entrada_scaled)[0]
    inference_time = (time.time() - start_time) * 1000 # ms
    
    # 2. Cálculos Termodinámicos de Proceso
    z_factor = (p_consigna * v_sensor) / (n_moles * R * t_pred)
    densidad = (n_moles * masa_molar_ch4) / v_sensor
    caudal_masico = densidad * flujo_m3_s # kg/s
    
    # Cálculo de Entalpía (H) y Entropía (S) aproximadas
    # H = U + PV (Asumiendo Gas Ideal para el incremento térmico)
    entalpia_especifica = (cp_ch4 * t_pred) / masa_molar_ch4 # J/kg
    
    # 3. Conversión de Energía (Turbina de Expansión)
    p_descarga = 101325 # Presión atmosférica
    t_iso = t_pred * (p_descarga / p_consigna)**((gamma-1)/gamma)
    trabajo_especifico = 0.82 * cp_ch4 * (t_pred - t_iso) # 82% eficiencia
    potencia_mw = (trabajo_especifico * n_moles) / 1e6 # Megavatios
    
    # --- CONSOLA DE OPERADOR SCADA ---
    print(f"\n{'#'*60}")
    print(f"{'HMI - CENTRAL TERMOELÉCTRICA DE METANO':^60}")
    print(f"{'#'*60}")
    
    print(f" [ESTADO DEL PROCESO]")
    print(f"  > Presión de Red:      {p_consigna/1e6:10.3f} MPa      |  Flujo Másico: {caudal_masico:8.2f} kg/s")
    print(f"  > Temperatura (IA):    {t_pred:10.2f} K        |  Densidad:     {densidad:8.2f} kg/m3")
    print(f"  > Volumen Cámara:      {v_sensor:10.3f} m3       |  Factor Z:     {z_factor:8.4f}")
    
    print(f"-"*60)
    print(f" [MÉTRICAS DE ENERGÍA Y ENTALPIA]")
    print(f"  > Entalpía Específ.:   {entalpia_especifica/1e3:10.2f} kJ/kg")
    print(f"  > Potencia Bruta:      {potencia_mw:10.3f} MW")
    print(f"  > Eficiencia Térmica:  {82.0:10.1f} %")
    
    print(f"-"*60)
    print(f" [DIAGNÓSTICO DE CONTROL - HEARTBEAT]")
    status_ia = "ÓPTIMO" if inference_time < 50 else "LATENCIA ALTA"
    print(f"  > Latencia IA:         {inference_time:10.4f} ms       |  Status CPU:   {status_ia}")
    
    # Alarmas Lógicas
    alerta = "OK"
    if p_consigna > 22e6: alerta = "ALERTA: SOBREPRESIÓN"
    if t_pred > 600: alerta = "CRÍTICO: ESTRÉS TÉRMICO"
    
    print(f"  > ALERTAS SISTEMA:     {alerta}")
    print(f"{'#'*60}\n")

# Simulando lectura de sensores: Volumen, Presión, Caudal de entrada m3/s
scada_monitor_avanzado(v_sensor=1.2, p_consigna=18500000, flujo_m3_s=0.05)