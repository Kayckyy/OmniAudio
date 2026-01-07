import sys
import os
import json
import threading
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np

# Import existing modules
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, 'src'))

from utils.audio_io import load_audio
from core.omni_format import OMNIFormat
from dsp.hrtf_engine import HRTFEngine
from dsp.processor import OMNIProcessor
from utils.separator import AudioSeparator

class VideoProcessorWorker(QRunnable):
    def __init__(self, video_path, output_format='binaural'):
        super().__init__()
        self.signals = WorkerSignals()
        self.video_path = video_path
        self.output_format = output_format
        self.hrtf = HRTFEngine(os.path.join(base_path, "data", "hrtf"))
        self.processor = OMNIProcessor(self.hrtf)

    @pyqtSlot()
    def run(self):
        try:
            self.signals.status.emit("Extraindo áudio do vídeo...")
            work_file = self.extract_audio_from_video(self.video_path)
            
            self.signals.status.emit("Analisando áudio...")
            import soundfile as sf
            info = sf.info(work_file)
            proj_dir = os.path.join("temp", os.path.splitext(os.path.basename(self.video_path))[0])
            if not os.path.exists(proj_dir): os.makedirs(proj_dir)
            
            self.signals.progress.emit(20)
            
            is_surround = info.channels > 2
            if is_surround:
                self.signals.status.emit("Separando canais surround...")
                stems = self.split_multichannel(work_file, proj_dir)
            else:
                self.signals.status.emit("Separando stems com IA...")
                try:
                    stems = AudioSeparator.separate(work_file, "temp")[1]
                except:
                    self.signals.status.emit("Usando separação simples...")
                    stems = self.simple_stem_separation(work_file, "temp")[1]
            
            self.signals.progress.emit(60)
            
            if stems:
                self.signals.status.emit("Criando projeto OMNI...")
                omni_path = os.path.join(proj_dir, os.path.basename(proj_dir)+".omni")
                OMNIFormat.create_multi_stem_omni(stems, info.duration, info.samplerate, omni_path, is_surround=is_surround)
                
                self.signals.progress.emit(80)
                self.signals.status.emit("Processando áudio 3D...")
                self.process_omni_file(omni_path, self.output_format)
                
                self.signals.progress.emit(100)
                self.signals.finished.emit(f"Vídeo processado: {self.video_path}")
                
        except Exception as e:
            self.signals.error.emit(str(e))

    def extract_audio_from_video(self, video_path):
        VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm')
        if not video_path.lower().endswith(VIDEO_EXTENSIONS): 
            return video_path
        
        output_wav = os.path.splitext(video_path)[0] + "_extracted.wav"
        try:
            import subprocess
            probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channels', '-of', 'csv=p=0', video_path], capture_output=True, text=True)
            channels = int(probe.stdout.strip()) if probe.stdout.strip() else 2
            cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', output_wav, '-y']
            if channels <= 2: 
                cmd.insert(-2, '-ac'); cmd.insert(-2, '2')
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return output_wav
        except Exception as e:
            raise Exception(f"Erro ao extrair áudio do vídeo: {str(e)}")

    def split_multichannel(self, wav_path, output_dir):
        import soundfile as sf
        import subprocess
        try:
            info = sf.info(wav_path)
            if info.channels <= 2: 
                return None
            
            layout = ["FL", "FR", "FC", "LFE", "SL", "SR", "BL", "BR"][:info.channels]
            stems = []
            for i, name in enumerate(layout):
                ch_file = os.path.join(output_dir, f"{name}.wav")
                cmd = ['ffmpeg', '-i', wav_path, '-af', f'pan=mono|c0=c{i}', ch_file, '-y']
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                stems.append({'name': name, 'file': ch_file})
            return stems
        except Exception as e:
            raise Exception(f"Erro ao separar canais surround: {str(e)}")

    def simple_stem_separation(self, audio_path, output_root):
        """Fallback simple separation when demucs is not available"""
        try:
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            project_dir = os.path.join("projects", filename)
            os.makedirs(project_dir, exist_ok=True)
            
            # Create a simple stereo stem
            import soundfile as sf
            import shutil
            data, sr = sf.read(audio_path)
            
            # If stereo, keep as is. If mono, duplicate to stereo
            if len(data.shape) == 1:
                data = np.column_stack([data, data])
            
            output_file = os.path.join(project_dir, "stereo.wav")
            sf.write(output_file, data, sr)
            
            stems_configs = [{'name': 'Stereo', 'file': output_file}]
            return project_dir, stems_configs
        except Exception as e:
            raise Exception(f"Erro na separação simples: {str(e)}")

    def process_omni_file(self, omni_path, output_format):
        try:
            self.hrtf.load()
            
            with open(omni_path, 'r') as f: 
                data = json.load(f)
            
            sr, master_len = data['sample_rate'], int(data['duration'] * data['sample_rate'])
            master_mix = np.zeros((master_len, 2))

            for obj in data['objects']:
                audio, _ = load_audio(obj['file'])
                if len(audio) > master_len: 
                    audio = audio[:master_len]
                
                if output_format == 'cinema_binaural':
                    if len(audio.shape) == 1: 
                        audio = np.stack([audio, audio], axis=1)
                    l, r = self.processor.process_stereo_cinema(audio, sr)
                    res = np.stack([l, r], axis=1)
                elif output_format in ['static_binaural', 'static_speaker']:
                    if len(audio.shape) == 1: 
                        audio = np.stack([audio, audio], axis=1)
                    l, r = self.processor.process_stereo_fixed(audio, sr)
                    res = np.stack([l, r], axis=1)
                else:
                    if len(audio.shape) > 1: 
                        audio = np.mean(audio, axis=1)
                    l, r = self.processor.process_object_binaural(audio, sr, obj)
                    res = np.stack([l, r], axis=1)
                
                l_mix = min(len(res), master_len)
                master_mix[:l_mix] += res[:l_mix]

            if output_format in ['speaker_3d', 'static_speaker']: 
                master_mix = self.processor.apply_xtc(master_mix, sr)
            
            peak = np.max(np.abs(master_mix))
            if peak > 0: 
                master_mix /= (peak + 1e-6)
            
            out_name = f"{os.path.splitext(omni_path)[0]}_{output_format}.wav"
            import soundfile as sf
            sf.write(out_name, master_mix, sr)
            
            # Merge with video
            self.merge_audio_to_video(self.video_path, out_name)
        except Exception as e:
            raise Exception(f"Erro no processamento 3D: {str(e)}")

    def merge_audio_to_video(self, video_path, audio_3d_path):
        VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm')
        if not video_path.lower().endswith(VIDEO_EXTENSIONS): 
            return
            
        output_video = os.path.splitext(video_path)[0] + "_Omni_3D.mp4"
        import subprocess
        cmd = ['ffmpeg', '-i', video_path, '-i', audio_3d_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', output_video, '-y']
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if os.path.exists(audio_3d_path): 
                os.remove(audio_3d_path)
            ext_wav = video_path.replace(os.path.splitext(video_path)[1], "_extracted.wav")
            if os.path.exists(ext_wav): 
                os.remove(ext_wav)
        except:
            pass

class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

class VideoDropZone(QLabel):
    fileDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #e74c3c;
                border-radius: 15px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                color: #495057;
                font-size: 16px;
                font-weight: bold;
                padding: 40px;
                min-height: 200px;
            }
            QLabel:hover {
                border-color: #c0392b;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #dee2e6);
            }
        """)
        self.setText("🎬 Arraste e solte arquivos de vídeo aqui\n\nou clique para selecionar")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            try:
                self.setStyleSheet(self.styleSheet().replace("#e74c3c", "#c0392b"))
            except:
                pass

    def dragLeaveEvent(self, event):
        try:
            self.setStyleSheet(self.styleSheet().replace("#c0392b", "#e74c3c"))
        except:
            pass

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.fileDropped.emit(files[0])
        try:
            self.setStyleSheet(self.styleSheet().replace("#c0392b", "#e74c3c"))
        except:
            pass

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione um arquivo de vídeo",
            "",
            "Arquivos de Vídeo (*.mp4 *.mkv *.mov *.avi *.flv *.wmv *.webm)"
        )
        if file_path:
            self.fileDropped.emit(file_path)

class VideoProcessorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        self.current_file = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("🎬 Omni Video Processor - 3D Audio for Videos")
        self.setGeometry(150, 150, 900, 700)
        
        # Dark theme
        self.setStyleSheet("""
            QWidget {
                background: #2c3e50;
                color: #ecf0f1;
            }
            QMainWindow {
                background: #2c3e50;
            }
            QLabel {
                color: #ecf0f1;
                font-family: 'Segoe UI', Arial;
                background: transparent;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e74c3c, stop:1 #c0392b);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ec7063, stop:1 #e74c3c);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #c0392b, stop:1 #a93226);
            }
            QPushButton:disabled {
                background: #7f8c8d;
            }
            QComboBox {
                background: #34495e;
                border: 2px solid #e74c3c;
                border-radius: 6px;
                padding: 8px;
                color: #ecf0f1;
                font-size: 14px;
            }
            QComboBox QAbstractItemView {
                background: #34495e;
                color: #ecf0f1;
                selection-background-color: #e74c3c;
            }
            QProgressBar {
                border: 2px solid #e74c3c;
                border-radius: 8px;
                text-align: center;
                color: white;
                font-weight: bold;
                background: #34495e;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e74c3c, stop:1 #c0392b);
                border-radius: 6px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title = QLabel("🎬 Omni Video Processor")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #ecf0f1; margin: 20px 0;")
        layout.addWidget(title)

        # Drop zone
        self.drop_zone = VideoDropZone()
        self.drop_zone.fileDropped.connect(self.on_file_dropped)
        layout.addWidget(self.drop_zone)

        # Controls
        controls_layout = QHBoxLayout()
        
        # Format selector
        format_label = QLabel("Formato de Saída:")
        format_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        controls_layout.addWidget(format_label)
        
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "Binaural 3D",
            "Cinema Binaural", 
            "Estéreo Fixo",
            "Speaker 3D",
            "Estéreo Estático"
        ])
        self.format_combo.setStyleSheet("min-width: 200px;")
        controls_layout.addWidget(self.format_combo)
        
        controls_layout.addStretch()
        
        # Process button
        self.process_btn = QPushButton("🚀 Processar Vídeo")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("font-size: 16px; padding: 15px 30px; min-width: 200px;")
        controls_layout.addWidget(self.process_btn)
        
        layout.addLayout(controls_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Pronto para processar vídeos")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; color: #bdc3c7; margin: 10px 0;")
        layout.addWidget(self.status_label)

        # Current file
        self.file_label = QLabel("")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("font-size: 12px; color: #95a5a6; margin: 5px 0;")
        layout.addWidget(self.file_label)

        layout.addStretch()

    def on_file_dropped(self, file_path):
        self.current_file = file_path
        self.file_label.setText(f"📁 {os.path.basename(file_path)}")
        self.process_btn.setEnabled(True)
        self.status_label.setText("Vídeo carregado. Selecione o formato e clique em Processar.")

    def start_processing(self):
        if not self.current_file:
            return

        format_map = {
            "Binaural 3D": "binaural",
            "Cinema Binaural": "cinema_binaural",
            "Estéreo Fixo": "static_binaural", 
            "Speaker 3D": "speaker_3d",
            "Estéreo Estático": "static_speaker"
        }
        
        output_format = format_map[self.format_combo.currentText()]
        
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        worker = VideoProcessorWorker(self.current_file, output_format)
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.status.connect(self.status_label.setText)
        worker.signals.finished.connect(self.on_processing_finished)
        worker.signals.error.connect(self.on_processing_error)
        
        self.threadpool.start(worker)

    def on_processing_finished(self, message):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_label.setText("✅ " + message)
        QMessageBox.information(self, "Sucesso!", message)

    def on_processing_error(self, error):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.status_label.setText("❌ Erro no processamento")
        QMessageBox.critical(self, "Erro", f"Ocorreu um erro:\n{error}")

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Omni Video Processor")
    app.setStyle('Fusion')
    
    window = VideoProcessorWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
