"""
Muestra en tiempo real la señal ECG del Arduino (o de un CSV ya capturado),
aplicando un filtrado causal (apto para streaming) para que la traza se
vea limpia, similar a la de un electrocardiografo real, y calcula la
frecuencia cardiaca latido a latido detectando picos R.

Requiere:
    pip install pyserial pyqtgraph pyqt5 numpy scipy --break-system-packages

Uso:
    # Con el Arduino conectado, tal como en tu script de captura:
    python3 ecg_realtime_view.py --port /dev/ttyUSB0

    # Modo simulacion, reproduce un CSV ya capturado (util para ensayar
    # la demo o como respaldo si el dia de la exposicion falla el hardware):
    python3 ecg_realtime_view.py --simulate ecg_raw_capture.csv
"""
import sys
import time
import argparse
import threading
import collections
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, find_peaks, iirnotch, lfilter, lfilter_zi

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

FS_NOMINAL = 360.2305
LEADS_OFF_SENTINEL = 65535
BAUD = 115200

# Ventana visible en pantalla (estilo "barrido" de monitor de hospital)
WINDOW_SECONDS = 6
# Cuanta historia reciente usamos para el umbral adaptativo de deteccion de picos R
PEAK_WINDOW_SECONDS = 2.5
# Periodo refractario tras un pico (evita contar dos veces el mismo QRS)
REFRACTORY_SECONDS = 0.3


class CausalECGFilter:
    """
    Filtro pasa-banda 0.5-40 Hz, causal, con estado (zi) persistente entre
    bloques de datos. Esto es lo que permite filtrar en tiempo real sin que
    la traza "salte" cada vez que llega un nuevo paquete de muestras
    (a diferencia de filtfilt, que necesita la señal completa).
    """

    def __init__(self, fs, low=0.5, high=40.0, order=2, notch_hz=60.0, notch_q=30.0):
        nyq = fs / 2
        self.sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        self.zi = sosfilt_zi(self.sos)
        self.use_notch = notch_hz > 0 and notch_hz < nyq
        if self.use_notch:
            b_notch, a_notch = iirnotch(w0=notch_hz, Q=notch_q, fs=fs)
            self.b_notch = b_notch
            self.a_notch = a_notch
            self.zi_notch = lfilter_zi(self.b_notch, self.a_notch)

    def process(self, chunk):
        chunk = np.asarray(chunk, dtype=float)
        out, self.zi = sosfilt(self.sos, chunk, zi=self.zi)
        if self.use_notch:
            out, self.zi_notch = lfilter(self.b_notch, self.a_notch, out, zi=self.zi_notch)
        return out


class LiveCSVRecorder:
    """Guarda la captura mientras se visualiza, sin frenar la interfaz."""

    def __init__(self, output_path):
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["indice", "tiempo_s", "adc_raw", "ecg_filtrado", "bpm_estimado"])

    def write_rows(self, rows):
        if rows:
            self._writer.writerows(rows)

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()


class RPeakDetector:
    """
    Deteccion de picos R en vivo:
    - umbral adaptativo = media + k * desviacion estandar de una ventana reciente
    - periodo refractario para no contar el mismo latido dos veces
    """

    def __init__(self, fs, k=1.4):
        self.fs = fs
        self.k = k
        self.refractory_samples = int(REFRACTORY_SECONDS * fs)
        self.samples_since_peak = self.refractory_samples
        self.rr_intervals = collections.deque(maxlen=8)
        self.last_peak_time = None

    def update(self, filtered_chunk, t_start):
        """Devuelve lista de (indice_en_chunk, tiempo) de picos detectados."""
        peaks_found = []
        thresh = filtered_chunk.mean() + self.k * filtered_chunk.std()
        idx, _ = find_peaks(filtered_chunk, height=thresh, distance=max(1, self.refractory_samples))
        for i in idx:
            t_peak = t_start + i / self.fs
            if self.last_peak_time is None or (t_peak - self.last_peak_time) > REFRACTORY_SECONDS:
                if self.last_peak_time is not None:
                    self.rr_intervals.append(t_peak - self.last_peak_time)
                self.last_peak_time = t_peak
                peaks_found.append((i, t_peak))
        return peaks_found

    @property
    def bpm(self):
        if len(self.rr_intervals) < 2:
            return None
        rr = np.median(self.rr_intervals)
        if rr <= 0:
            return None
        return 60.0 / rr


class SerialReader(threading.Thread):
    """Lee lineas del Arduino en un hilo aparte para no bloquear la interfaz grafica."""

    def __init__(self, port, baud, out_queue):
        super().__init__(daemon=True)
        import serial  # import local para que el modo --simulate no requiera pyserial
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # esperar bootloader del Arduino Uno
        self.ser.reset_input_buffer()
        self.out_queue = out_queue
        self._stop = threading.Event()

    def run(self):
        buffer = bytearray()
        while not self._stop.is_set():
            chunk = self.ser.read(4096)
            if not chunk:
                continue
            buffer.extend(chunk)
            while True:
                nl = buffer.find(b"\n")
                if nl < 0:
                    break
                line = buffer[:nl].strip()
                del buffer[: nl + 1]
                if not line:
                    continue
                try:
                    raw = int(line)
                except ValueError:
                    continue
                self.out_queue.append(None if raw == LEADS_OFF_SENTINEL else raw)

    def stop(self):
        self._stop.set()
        self.ser.close()


class CSVSimulator(threading.Thread):
    """Reproduce un CSV ya capturado a la velocidad de muestreo original, como si viniera del Arduino."""

    def __init__(self, path, fs, out_queue):
        super().__init__(daemon=True)
        import csv

        self.samples = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("adc_raw", "")
                self.samples.append(None if v == "" else int(v))
        self.fs = fs
        self.out_queue = out_queue
        self._stop = threading.Event()

    def run(self):
        period = 1.0 / self.fs
        for v in self.samples:
            if self._stop.is_set():
                break
            self.out_queue.append(v)
            time.sleep(period)

    def stop(self):
        self._stop.set()


class ECGMonitorWindow(QtWidgets.QMainWindow):
    def __init__(self, fs, sample_queue, line_frequency, record_path):
        super().__init__()
        self.fs = fs
        self.sample_queue = sample_queue
        self.setWindowTitle("Monitor ECG en tiempo real")
        self.resize(1000, 500)

        pg.setConfigOption("background", "k")
        pg.setConfigOption("foreground", "w")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "tiempo", "s")
        self.plot_widget.setLabel("left", "amplitud (u.a.)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color=(0, 255, 90), width=2))
        self.peak_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush(255, 60, 60), size=10)
        self.plot_widget.addItem(self.peak_scatter)
        layout.addWidget(self.plot_widget, stretch=4)

        self.bpm_label = QtWidgets.QLabel("-- bpm")
        self.bpm_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.bpm_label.setStyleSheet(
            "color: #2ecc71; background-color: black; font-size: 64px; font-weight: bold;"
        )
        layout.addWidget(self.bpm_label, stretch=1)

        n_window = int(WINDOW_SECONDS * fs)
        self.buffer = collections.deque([0.0] * n_window, maxlen=n_window)
        self.time_buffer = collections.deque(
            [i / fs for i in range(-n_window, 0)], maxlen=n_window
        )
        self.sample_index = 0
        self.filter = CausalECGFilter(fs, notch_hz=line_frequency)
        self.detector = RPeakDetector(fs)
        self.peak_points = []  # [(t, valor), ...] visibles en la ventana actual
        self.recorder = LiveCSVRecorder(record_path) if record_path else None

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(40)  # ~25 fps, suficiente para que se vea fluido

    def update_plot(self):
        # sacar todo lo que haya llegado desde el ultimo refresco
        new_raw = []
        while self.sample_queue:
            new_raw.append(self.sample_queue.popleft())

        if not new_raw:
            return

        # tratar electrodo desconectado como mantener el ultimo valor (evita picos falsos)
        clean_raw = []
        last_val = self.buffer[-1] if self.buffer else 0.0
        for v in new_raw:
            if v is None:
                clean_raw.append(last_val)
            else:
                clean_raw.append(float(v))
                last_val = float(v)

        t_start = self.sample_index / self.fs
        filtered = self.filter.process(clean_raw)

        peaks = self.detector.update(filtered, t_start)
        for i, t_peak in peaks:
            self.peak_points.append((t_peak, filtered[i]))

        bpm = self.detector.bpm
        rows_to_write = []

        for i, val in enumerate(filtered):
            self.buffer.append(val)
            t_sample = t_start + i / self.fs
            self.time_buffer.append(t_sample)

            if self.recorder is not None:
                raw_v = new_raw[i]
                rows_to_write.append([
                    self.sample_index + i,
                    f"{t_sample:.6f}",
                    "" if raw_v is None else raw_v,
                    f"{val:.6f}",
                    "" if bpm is None else f"{bpm:.2f}",
                ])

        self.sample_index += len(clean_raw)

        if self.recorder is not None:
            self.recorder.write_rows(rows_to_write)

        t_now = self.time_buffer[-1]
        self.peak_points = [(t, v) for (t, v) in self.peak_points if t_now - t <= WINDOW_SECONDS]

        self.curve.setData(list(self.time_buffer), list(self.buffer))
        if self.peak_points:
            xs, ys = zip(*self.peak_points)
            self.peak_scatter.setData(list(xs), list(ys))
        else:
            self.peak_scatter.setData([], [])

        self.plot_widget.setXRange(t_now - WINDOW_SECONDS, t_now, padding=0)

        if bpm is not None:
            self.bpm_label.setText(f"{bpm:.0f}\nbpm")
        else:
            self.bpm_label.setText("--\nbpm")

    def closeEvent(self, event):
        if self.recorder is not None:
            self.recorder.close()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Monitor ECG en tiempo real")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Puerto serie del Arduino")
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--fs", type=float, default=FS_NOMINAL, help="Frecuencia de muestreo nominal")
    parser.add_argument(
        "--line-frequency",
        type=float,
        default=60.0,
        help="Frecuencia de la red para notch (0 desactiva, 50 o 60 recomendado)",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Ruta del CSV para guardar en vivo (por defecto crea uno con timestamp)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Desactiva guardado en CSV durante la visualizacion",
    )
    parser.add_argument(
        "--simulate",
        default=None,
        help="Ruta a un CSV ya capturado (columna adc_raw) para reproducir en vez de leer el Arduino",
    )
    args = parser.parse_args()

    sample_queue = collections.deque(maxlen=max(1000, int(args.fs * 12)))

    if args.no_record:
        record_path = None
    elif args.record:
        record_path = args.record
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        record_path = str(Path(__file__).with_name(f"ecg_live_capture_{ts}.csv"))

    if args.simulate:
        source = CSVSimulator(args.simulate, args.fs, sample_queue)
    else:
        source = SerialReader(args.port, args.baud, sample_queue)
    source.start()

    app = QtWidgets.QApplication(sys.argv)
    window = ECGMonitorWindow(args.fs, sample_queue, args.line_frequency, record_path)
    if record_path:
        print(f"Grabando captura en: {Path(record_path).expanduser().resolve()}")
    window.show()

    try:
        exit_code = app.exec()
    finally:
        source.stop()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()