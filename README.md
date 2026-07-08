# Proyecto Final TISC — Sistema de Análisis y Captura de ECG

Sistema de captura, visualización y análisis de señal ECG en tiempo real con Arduino, incluyendo un predictor basado en Machine Learning.

---

## Requisitos previos

- **Python 3.10 o superior** → https://www.python.org/downloads/
- **Git** (opcional, para clonar el repositorio) → https://git-scm.com/downloads

---

## Instalación en un PC nuevo

### 1. Obtener los archivos del proyecto

Copia la carpeta del proyecto al nuevo PC (USB, OneDrive, etc.) o clónala con Git:

```bash
git clone <URL_DEL_REPOSITORIO>
cd Proyecto_finalTISC
```

### 2. (Recomendado) Crear un entorno virtual

```bash
python -m venv venv
```

Activar el entorno virtual:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

> Deberías ver `(venv)` al inicio de la línea del terminal.

### 3. Instalar todas las dependencias

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn imbalanced-learn wfdb pywavelets pyserial pyqtgraph PyQt5
```

Un solo comando. Espera a que termine (~2-5 minutos según la conexión).

---

## Cómo usar cada script

### `dcp.py` — Visualización ECG en tiempo real

Muestra la señal ECG en vivo desde el Arduino con filtrado causal y detección de frecuencia cardíaca.

**Con Arduino conectado (reemplaza `COM3` con tu puerto):**
```bash
python dcp.py --port COM3
```

**Modo simulación con CSV (sin hardware):**
```bash
python dcp.py --simulate ecg_live_capture_20260706_181344.csv
```

> En Linux/macOS el puerto suele ser `/dev/ttyUSB0` o `/dev/ttyACM0`.

---

### `predictor.py` — Entrenamiento del modelo ML

Carga datos del CSV, extrae características y entrena un clasificador Random Forest.

```bash
python predictor.py
```

Genera el modelo entrenado y actualiza `metadata_modelo.json`.

---

### `analizar_csv_ecg.py` — Análisis estadístico del CSV

Analiza y genera gráficas de la señal ECG guardada en CSV.

```bash
python analizar_csv_ecg.py
```

Las gráficas se guardan como archivos de imagen en la misma carpeta.

---

## Estructura de archivos

```
Proyecto_finalTISC/
├── dcp.py                          # Visualizador ECG en tiempo real
├── predictor.py                    # Entrenamiento del modelo ML
├── analizar_csv_ecg.py             # Análisis offline de CSV
├── serial.py                       # Módulo de comunicación serial (interno)
├── metadata_modelo.json            # Metadata del modelo entrenado
├── ecg_live_capture_*.csv          # Capturas de ECG guardadas
└── README.md                       # Este archivo
```

---

## Solución de problemas frecuentes

| Problema | Solución |
|---|---|
| `ModuleNotFoundError: No module named 'X'` | Ejecuta de nuevo el `pip install` del paso 3 |
| Puerto serial no encontrado | Instala el driver CH340/CP210x de tu Arduino y verifica el puerto en el Administrador de dispositivos |
| `pip` no se reconoce | Asegúrate de marcar "Add Python to PATH" al instalar Python |
| Gráficas no se abren | El script usa backend `Agg` (guarda imágenes en disco en vez de mostrarlas) |
| Error con `PyQt5` en Linux | Ejecuta `sudo apt install python3-pyqt5` |

---

## Dependencias resumidas

| Paquete | Uso |
|---|---|
| `numpy` | Cálculos numéricos |
| `pandas` | Lectura de CSV |
| `scipy` | Filtros de señal |
| `matplotlib` / `seaborn` | Gráficas |
| `scikit-learn` | Modelo Random Forest |
| `imbalanced-learn` | SMOTE para balanceo de clases |
| `wfdb` | Base de datos MIT-BIH |
| `pywavelets` | Análisis wavelet |
| `pyserial` | Comunicación con Arduino |
| `pyqtgraph` / `PyQt5` | Visualización en tiempo real |
