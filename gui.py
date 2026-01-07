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

class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    log_message = pyqtSignal(str, str)  # message, level

    def __init__(self):
        super().__init__()

class AudioProcessorWorker(QRunnable):
    def __init__(self, mode, input_file, output_format='binaural', seed=None, keyframes=None, output_folder=None):
        super().__init__()
        self.signals = WorkerSignals()
        self.mode = mode
        self.input_file = input_file
        self.output_format = output_format
        self.seed = seed
        self.keyframes = keyframes or []
        self.output_folder = output_folder
        self.hrtf = HRTFEngine(os.path.join(base_path, "data", "hrtf"))
        self.processor = OMNIProcessor(self.hrtf)

    @pyqtSlot()
    def run(self):
        try:
            if self.mode == 'process_omni':
                # Processar arquivo OMNI existente
                self.signals.status.emit("Carregando projeto OMNI existente...")
                self.signals.log_message.emit("Carregando configurações do arquivo OMNI...", "INFO")
                
                with open(self.input_file, 'r', encoding='utf-8') as f:
                    omni_data = json.load(f)
                
                self.signals.progress.emit(20)
                
                # Processar diretamente o OMNI
                self.process_omni_file(self.input_file, self.output_format)
                
                self.signals.progress.emit(100)
                self.signals.log_message.emit("Processamento OMNI concluído com sucesso!", "SUCCESS")
                self.signals.finished.emit(f"Projeto OMNI processado: {self.input_file}")
                
            elif self.mode == 'auto':
                # Processamento normal de arquivo novo
                self.signals.status.emit("Extraindo áudio do vídeo...")
                self.signals.log_message.emit("Iniciando processamento automático...", "INFO")
                work_file = self.extract_audio_from_video(self.input_file)
                
                self.signals.status.emit("Analisando áudio...")
                self.signals.log_message.emit("Analisando propriedades do arquivo de áudio...", "INFO")
                import soundfile as sf
                info = sf.info(work_file)
                
                # Criar pasta de projeto personalizada
                if self.output_folder:
                    proj_dir = os.path.join(self.output_folder, os.path.splitext(os.path.basename(self.input_file))[0])
                else:
                    proj_dir = os.path.join("temp", os.path.splitext(os.path.basename(self.input_file))[0])
                
                if not os.path.exists(proj_dir): 
                    os.makedirs(proj_dir)
                    self.signals.log_message.emit(f"Pasta de projeto criada: {proj_dir}", "INFO")
                
                self.signals.progress.emit(20)
                
                is_surround = info.channels > 2
                if is_surround:
                    self.signals.status.emit("Separando canais surround...")
                    self.signals.log_message.emit(f"Detectado áudio surround com {info.channels} canais", "INFO")
                    stems = self.split_multichannel(work_file, proj_dir)
                    if stems:
                        self.signals.log_message.emit(f"Separados {len(stems)} canais surround", "SUCCESS")
                else:
                    try:
                        self.signals.status.emit("Separando stems com IA...")
                        self.signals.log_message.emit("Tentando separação com Demucs (IA)...", "INFO")
                        stems = AudioSeparator.separate(work_file, "temp")[1]
                        self.signals.log_message.emit(f"Separados {len(stems)} stems com IA", "SUCCESS")
                    except:
                        self.signals.status.emit("Usando separação simples (Demucs não disponível)...")
                        self.signals.log_message.emit("Demucs não disponível, usando separação simples", "WARNING")
                        stems = self.simple_stem_separation(work_file, "temp")[1]
                        self.signals.log_message.emit("Separação simples concluída", "SUCCESS")
                
                self.signals.progress.emit(60)
                
                if stems:
                    self.signals.status.emit("Criando projeto OMNI...")
                    self.signals.log_message.emit("Criando arquivo de projeto OMNI...", "INFO")
                    omni_path = os.path.join(proj_dir, os.path.basename(proj_dir)+".omni")
                    generated_seed = OMNIFormat.create_multi_stem_omni(
                        stems, info.duration, info.samplerate, omni_path, 
                        is_surround=is_surround, seed=self.seed, keyframes=self.keyframes
                    )
                    
                    if self.seed:
                        self.signals.log_message.emit(f"Usando seed fornecida: {self.seed}", "INFO")
                    else:
                        self.signals.log_message.emit(f"Seed gerada automaticamente: {generated_seed}", "INFO")
                    
                    if self.keyframes:
                        self.signals.log_message.emit(f"Aplicados {len(self.keyframes)} keyframes", "INFO")
                    
                    self.signals.progress.emit(80)
                    self.signals.status.emit("Processando áudio 3D...")
                    self.signals.log_message.emit("Iniciando processamento 3D com HRTF...", "INFO")
                    self.process_omni_file(omni_path, self.output_format)
                    
                    self.signals.progress.emit(100)
                    self.signals.log_message.emit("Processamento concluído com sucesso!", "SUCCESS")
                    self.signals.finished.emit(f"Projeto concluído: {omni_path}")
                    
        except Exception as e:
            self.signals.log_message.emit(f"Erro durante processamento: {str(e)}", "ERROR")
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
            
            # Try to merge with video
            self.merge_audio_to_video(self.input_file, out_name)
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

class DropZone(QLabel):
    fileDropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #8e44ad;
                border-radius: 15px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
                color: #495057;
                font-size: 16px;
                font-weight: bold;
                padding: 40px;
                min-height: 200px;
                margin: 5px;
            }
            QLabel:hover {
                border-color: #9b59b6;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f8f9fa, stop:1 #dee2e6);
            }
        """)
        self.setText("🎵 Arraste e solte arquivos de áudio ou vídeo aqui\n\nou clique para selecionar")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            try:
                self.setStyleSheet(self.styleSheet().replace("#8e44ad", "#9b59b6"))
            except:
                pass

    def dragLeaveEvent(self, event):
        try:
            self.setStyleSheet(self.styleSheet().replace("#9b59b6", "#8e44ad"))
        except:
            pass

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.fileDropped.emit(files[0])
        try:
            self.setStyleSheet(self.styleSheet().replace("#9b59b6", "#8e44ad"))
        except:
            pass

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecione um arquivo de áudio ou vídeo",
            "",
            "Arquivos de Áudio/Vídeo (*.mp3 *.wav *.flac *.mp4 *.mkv *.mov *.avi)"
        )
        if file_path:
            self.fileDropped.emit(file_path)

class OmniAudioProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        self.current_file = None
        self.is_dark_theme = True
        
        # Carregar pasta de saída salva ou usar padrão
        self.output_folder = self.load_output_folder()
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        self.initUI()
    
    def load_output_folder(self):
        """Carregar pasta de saída salva ou usar padrão"""
        config_file = os.path.join(os.path.expanduser("~"), ".omni_audio_processor_config.json")
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('output_folder', os.path.join(os.path.expanduser("~"), "OmniAudioProcessor_Output"))
        except:
            pass
        
        # Retornar pasta padrão
        return os.path.join(os.path.expanduser("~"), "OmniAudioProcessor_Output")
    
    def save_output_folder(self, folder):
        """Salvar pasta de saída no arquivo de configuração"""
        config_file = os.path.join(os.path.expanduser("~"), ".omni_audio_processor_config.json")
        try:
            config = {'output_folder': folder}
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Erro ao salvar configuração: {e}")

    def get_dark_theme(self):
        return """
            QWidget {
                background: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #34495e;
            }
            QMainWindow {
                background: #2c3e50;
                border: 2px solid #8e44ad;
            }
            QLabel {
                color: #ecf0f1;
                font-family: 'Segoe UI', Arial;
                background: transparent;
                border: 1px solid transparent;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #9b59b6, stop:1 #8e44ad);
                border: 2px solid #8e44ad;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a569bd, stop:1 #9b59b6);
                border: 2px solid #9b59b6;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #8e44ad, stop:1 #7d3c98);
                border: 2px solid #7d3c98;
            }
            QPushButton:disabled {
                background: #7f8c8d;
                border: 2px solid #7f8c8d;
            }
            QComboBox {
                background: #34495e;
                border: 2px solid #8e44ad;
                border-radius: 6px;
                padding: 8px;
                color: #ecf0f1;
                font-size: 14px;
            }
            QComboBox QAbstractItemView {
                background: #34495e;
                color: #ecf0f1;
                selection-background-color: #8e44ad;
                border: 1px solid #8e44ad;
            }
            QProgressBar {
                border: 2px solid #8e44ad;
                border-radius: 8px;
                text-align: center;
                color: white;
                font-weight: bold;
                background: #34495e;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #9b59b6, stop:1 #8e44ad);
                border-radius: 6px;
            }
            QGroupBox {
                border: 2px solid #8e44ad;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #ecf0f1;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #8e44ad;
            }
            QTabBar::tab:hover {
                background: #9b59b6;
            }
            QLineEdit {
                background: #34495e;
                border: 2px solid #8e44ad;
                border-radius: 6px;
                padding: 8px;
                color: #ecf0f1;
                font-size: 14px;
            }
            QSpinBox {
                background: #34495e;
                border: 2px solid #8e44ad;
                border-radius: 6px;
                padding: 8px;
                color: #ecf0f1;
                font-size: 14px;
            }
            QDoubleSpinBox {
                background: #34495e;
                border: 2px solid #8e44ad;
                border-radius: 6px;
                padding: 8px;
                color: #ecf0f1;
                font-size: 14px;
            }
        """

    def get_light_theme(self):
        return """
            QWidget {
                background: #f8f9fa;
                color: #212529;
            }
            QMainWindow {
                background: #f8f9fa;
            }
            QLabel {
                color: #212529;
                font-family: 'Segoe UI', Arial;
                background: transparent;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6f42c1, stop:1 #563d7c);
                border: none;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #7952b3, stop:1 #6f42c1);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #563d7c, stop:1 #495057);
            }
            QPushButton:disabled {
                background: #adb5bd;
                color: #6c757d;
            }
            QComboBox {
                background: #ffffff;
                border: 2px solid #6f42c1;
                border-radius: 6px;
                padding: 8px;
                color: #212529;
                font-size: 14px;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #212529;
                selection-background-color: #6f42c1;
                border: 1px solid #6f42c1;
            }
            QProgressBar {
                border: 2px solid #6f42c1;
                border-radius: 8px;
                text-align: center;
                color: white;
                font-weight: bold;
                background: #e9ecef;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6f42c1, stop:1 #563d7c);
                border-radius: 6px;
            }
            QTabWidget::pane {
                border: 1px solid #6f42c1;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #f8f9fa;
                color: #212529;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #6f42c1;
                color: white;
            }
            QTabBar::tab:hover {
                background: #e9ecef;
                color: #212529;
            }
            QLineEdit {
                background: #ffffff;
                border: 2px solid #6f42c1;
                border-radius: 6px;
                padding: 8px;
                color: #212529;
                font-size: 14px;
            }
            QSpinBox {
                background: #ffffff;
                border: 2px solid #6f42c1;
                border-radius: 6px;
                padding: 8px;
                color: #212529;
                font-size: 14px;
            }
            QDoubleSpinBox {
                background: #ffffff;
                border: 2px solid #6f42c1;
                border-radius: 6px;
                padding: 8px;
                color: #212529;
                font-size: 14px;
            }
            QGroupBox {
                border: 2px solid #6f42c1;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #212529;
                font-weight: bold;
                background: #ffffff;
            }
            QGroupBox::title {
                color: #212529;
                font-weight: bold;
            }
            QListWidget {
                background: #ffffff;
                border: 1px solid #6f42c1;
                border-radius: 6px;
                color: #212529;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e9ecef;
            }
            QListWidget::item:selected {
                background: #6f42c1;
                color: white;
            }
        """

    def toggle_theme(self):
        self.is_dark_theme = not self.is_dark_theme
        if self.is_dark_theme:
            self.setStyleSheet(self.get_dark_theme())
            self.theme_btn.setText("☀️ Tema Claro")
        else:
            self.setStyleSheet(self.get_light_theme())
            self.theme_btn.setText("🌙 Tema Escuro")

    def toggle_fullscreen(self):
        """Toggle fullscreen mode - simplified version"""
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_btn.setText("⛶ Tela Cheia")
        else:
            self.showFullScreen()
            self.fullscreen_btn.setText("⛶ Janela Normal")
    
    def adjust_layout_for_fullscreen(self):
        """Adjust layout proportions for fullscreen mode"""
        try:
            # Adjust margins for better fullscreen spacing
            if hasattr(self, 'centralWidget') and self.centralWidget().layout():
                self.centralWidget().layout().setContentsMargins(*self.fullscreen_margins)
            
            # Make drop zone smaller in fullscreen
            if hasattr(self, 'drop_zone'):
                self.drop_zone.setStyleSheet("""
                    QLabel {
                        border: 3px dashed #8e44ad;
                        border-radius: 15px;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #f8f9fa, stop:1 #e9ecef);
                        color: #495057;
                        font-size: 16px;
                        font-weight: bold;
                        padding: 30px;
                        min-height: 150px;
                        max-height: 200px;
                        margin: 5px;
                    }
                    QLabel:hover {
                        border-color: #9b59b6;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #f8f9fa, stop:1 #dee2e6);
                    }
                """)
            
            # Increase log area height in fullscreen
            if hasattr(self, 'log_area'):
                self.log_area.setMinimumHeight(250)
                self.log_area.setMaximumHeight(400)
            
            # Adjust log group height for fullscreen
            if hasattr(self, 'log_group'):
                self.log_group.setMinimumHeight(450)
            
            # Add spacing to keep controls separated and visible
            if hasattr(self, 'centralWidget') and self.centralWidget().layout():
                main_layout = self.centralWidget().layout()
                main_layout.setSpacing(15)
                
            # Ensure tab widget doesn't expand too much in fullscreen
            if hasattr(self, 'tab_widget'):
                self.tab_widget.setMaximumHeight(700)
        except Exception as e:
            print(f"Erro ao ajustar layout para tela cheia: {e}")
    
    def adjust_layout_for_windowed(self):
        """Restore normal layout proportions"""
        try:
            # Restore original margins
            if hasattr(self, 'centralWidget') and self.centralWidget().layout():
                self.centralWidget().layout().setContentsMargins(*self.original_margins)
            
            # Restore original drop zone size
            if hasattr(self, 'drop_zone'):
                self.drop_zone.setStyleSheet("""
                    QLabel {
                        border: 3px dashed #8e44ad;
                        border-radius: 15px;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #f8f9fa, stop:1 #e9ecef);
                        color: #495057;
                        font-size: 16px;
                        font-weight: bold;
                        padding: 40px;
                        min-height: 200px;
                        margin: 5px;
                    }
                    QLabel:hover {
                        border-color: #9b59b6;
                        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                            stop:0 #f8f9fa, stop:1 #dee2e6);
                    }
                """)
            
            # Restore log area height
            if hasattr(self, 'log_area'):
                self.log_area.setMinimumHeight(200)
                self.log_area.setMaximumHeight(300)
            
            # Restore log group height
            if hasattr(self, 'log_group'):
                self.log_group.setMinimumHeight(350)
            
            # Restore original spacing
            if hasattr(self, 'centralWidget') and self.centralWidget().layout():
                main_layout = self.centralWidget().layout()
                main_layout.setSpacing(12)
                
            # Remove tab widget height restriction
            if hasattr(self, 'tab_widget'):
                self.tab_widget.setMaximumHeight(16777215)
        except Exception as e:
            print(f"Erro ao restaurar layout normal: {e}")

    def initUI(self):
        # Detectar resolução da tela e ajustar tamanho
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # Ajustar tamanho baseado na resolução
        if screen_width < 1280 or screen_height < 720:
            # Resolução baixa
            window_width, window_height = 900, 950  # Aumentado novamente
        elif screen_width < 1920 or screen_height < 1080:
            # Resolução média (HD)
            window_width, window_height = 1100, 1100  # Aumentado novamente
        else:
            # Resolução alta (Full HD+)
            window_width, window_height = 1200, 1200  # Aumentado novamente
        
        # Centralizar janela na tela
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.setWindowTitle("🎧 Omni Audio Processor - 3D Audio Enhancement")
        self.setGeometry(x, y, window_width, window_height)
        
        # Apply initial theme
        self.setStyleSheet(self.get_dark_theme())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(12)  # Espaçamento adaptativo
        layout.setContentsMargins(20, 20, 20, 20)  # Margem adaptativa
        
        # Store original margins for fullscreen adjustment
        self.original_margins = (20, 20, 20, 20)
        self.fullscreen_margins = (40, 20, 40, 20)

        # Header with theme toggle
        header_layout = QHBoxLayout()
        
        # Title
        title = QLabel("🎧 Omni Audio Processor")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #ecf0f1; margin: 20px 0;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Theme toggle button
        self.theme_btn = QPushButton("🌙 Tema Escuro")
        self.theme_btn.clicked.connect(self.toggle_theme)
        self.theme_btn.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a5f7a;
            }
            QPushButton:pressed {
                background-color: #2c3e50;
            }
        """)
        header_layout.addWidget(self.theme_btn)
        
        # Fullscreen toggle button
        self.fullscreen_btn = QPushButton("⛶ Tela Cheia")
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        
        # Add F11 shortcut for fullscreen toggle
        fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F11), self)
        fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        self.fullscreen_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
        """)
        header_layout.addWidget(self.fullscreen_btn)
        
        layout.addLayout(header_layout)

        # Tab widget for different sections
        self.tab_widget = QTabWidget()
        
        # Main processing tab
        main_tab = QWidget()
        main_layout = QVBoxLayout(main_tab)
        main_layout.setSpacing(10)  # Reduzido espaçamento interno
        
        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.fileDropped.connect(self.on_file_dropped)
        main_layout.addWidget(self.drop_zone)

        # Controls - Reorganized into multiple rows
        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setSpacing(10)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # First row - Format selector
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(10)
        
        format_label = QLabel("Formato de Saída:")
        format_label.setStyleSheet("font-size: 14px; font-weight: bold; min-width: 120px;")
        row1_layout.addWidget(format_label)
        
        self.format_combo = QComboBox()
        # Adicionando presets com nomes originais e descrições detalhadas
        presets = [
            ("binaural", "Binaural 360°", "Processamento binaural espacial completo com movimento 3D dos objetos"),
            ("cinema_binaural", "Cinema Binaural", "Simulação de home theater com reflexões de sala para experiência cinematográfica"),
            ("static_binaural", "Estéreo 3D", "Posicionamento estéreo fixo em 90° para separação clara dos canais"),
            ("speaker_3d", "Speaker 360°", "Processamento estéreo com cross-talk cancellation para fones de ouvido"),
            ("static_speaker", "Speaker3D (static)", "Posicionamento estéreo estático sem movimento espacial")
        ]
        
        for internal_name, display_name, description in presets:
            self.format_combo.addItem(display_name, description)
        
        self.format_combo.setStyleSheet("""
            QComboBox {
                min-width: 280px;
                max-width: 350px;
                font-size: 14px;
                padding: 6px;
                background: #34495e;
                border: 2px solid #8e44ad;
                border-radius: 6px;
                color: #ecf0f1;
            }
            QComboBox QAbstractItemView {
                background: #34495e;
                color: #ecf0f1;
                selection-background-color: #8e44ad;
                border: 1px solid #8e44ad;
            }
        """)
        self.format_combo.currentIndexChanged.connect(self.update_preset_description)
        row1_layout.addWidget(self.format_combo)
        
        row1_layout.addStretch()
        controls_layout.addLayout(row1_layout)
        
        # Second row - OMNI file and seed
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(10)
        
        # OMNI file section
        omni_label = QLabel("Arquivo OMNI:")
        omni_label.setStyleSheet("font-size: 14px; font-weight: bold; min-width: 120px;")
        row2_layout.addWidget(omni_label)
        
        self.omni_input = QLineEdit()
        self.omni_input.setPlaceholderText("Arraste um arquivo .omni ou deixe em branco")
        self.omni_input.setStyleSheet("""
            QLineEdit {
                min-width: 250px;
                max-width: 300px;
                font-size: 14px;
                padding: 8px;
                background: #34495e;
                border: 2px solid #8e44ad;
                border-radius: 6px;
                color: #ecf0f1;
            }
        """)
        row2_layout.addWidget(self.omni_input)
        
        # Browse OMNI button
        browse_omni_btn = QPushButton("📂")
        browse_omni_btn.clicked.connect(self.browse_omni_file)
        browse_omni_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px 12px;
                min-width: 45px;
                background: #34495e;
                border: 2px solid #8e44ad;
                border-radius: 6px;
                color: #ecf0f1;
            }
            QPushButton:hover {
                background: #4a5f7a;
                border-color: #9b59b6;
            }
        """)
        row2_layout.addWidget(browse_omni_btn)
        
        row2_layout.addSpacing(20)
        
        # Seed section
        seed_label = QLabel("Seed:")
        seed_label.setStyleSheet("font-size: 14px; font-weight: bold; min-width: 80px;")
        row2_layout.addWidget(seed_label)
        
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Opcional")
        self.seed_input.setStyleSheet("""
            QLineEdit {
                min-width: 120px;
                max-width: 180px;
                font-size: 14px;
                padding: 8px;
                background: #34495e;
                border: 2px solid #8e44ad;
                border-radius: 6px;
                color: #ecf0f1;
            }
        """)
        row2_layout.addWidget(self.seed_input)
        
        row2_layout.addStretch()
        controls_layout.addLayout(row2_layout)
        
        # Third row - Process button
        row3_layout = QHBoxLayout()
        row3_layout.setSpacing(10)
        
        # Process button
        self.process_btn = QPushButton("🚀 Processar Áudio")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 15px 30px;
                min-width: 200px;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #27ae60, stop:1 #229954);
                border: 2px solid #27ae60;
                border-radius: 8px;
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2ecc71, stop:1 #27ae60);
                border-color: #2ecc71;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #229954, stop:1 #1e8449);
            }
            QPushButton:disabled {
                background: #7f8c8d;
                border-color: #7f8c8d;
                color: #bdc3c7;
            }
        """)
        row3_layout.addWidget(self.process_btn)
        
        row3_layout.addStretch()
        controls_layout.addLayout(row3_layout)
        
        main_layout.addWidget(controls_container)

        # Description label
        self.preset_description = QLabel("")
        self.preset_description.setWordWrap(True)
        self.preset_description.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #bdc3c7;
                margin: 10px 0;
                padding: 12px;
                background: rgba(255,255,255,0.08);
                border-radius: 6px;
                border: 1px solid rgba(255,255,255,0.15);
            }
        """)
        self.preset_description.setMaximumHeight(60)
        self.preset_description.setMinimumHeight(40)
        main_layout.addWidget(self.preset_description)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #8e44ad;
                border-radius: 8px;
                text-align: center;
                color: white;
                font-weight: bold;
                background: #34495e;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #9b59b6, stop:1 #8e44ad);
                border-radius: 6px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Pronto para processar arquivos")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #bdc3c7;
                margin: 15px 0;
                padding: 8px;
                background: rgba(255,255,255,0.05);
                border-radius: 4px;
            }
        """)
        main_layout.addWidget(self.status_label)

        # Current file
        self.file_label = QLabel("")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #95a5a6;
                margin: 8px 0;
                padding: 6px;
                background: rgba(255,255,255,0.03);
                border-radius: 4px;
            }
        """)
        main_layout.addWidget(self.file_label)

        # Log area
        self.log_group = QGroupBox("Log Detalhado")
        self.log_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #8e44ad;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 20px;
                font-size: 14px;
                font-weight: bold;
                color: #ecf0f1;
                min-height: 350px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        log_layout = QVBoxLayout(self.log_group)
        log_layout.setSpacing(15)
        log_layout.setContentsMargins(15, 25, 15, 15)
        
        self.log_area = QTextEdit()
        self.log_area.setMinimumHeight(200)  # Aumentado ainda mais
        self.log_area.setMaximumHeight(300)  # Permitir muito mais espaço
        self.log_area.setReadOnly(True)
        
        # Configurar scroll bar
        self.log_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.log_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Melhorar aparencia da scroll bar
        self.log_area.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                background: #1a1a1a;
                color: #00ff00;
                border: 1px solid #444;
                padding: 10px;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background: #2c3e50;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #8e44ad;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9b59b6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #2c3e50;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background: #8e44ad;
                min-width: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #9b59b6;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        log_layout.addWidget(self.log_area)
        
        # Button container with proper alignment
        button_container = QWidget()
        button_container.setMinimumHeight(50)  # Garantir espaço para o botão
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 10, 0, 5)  # Mais espaço acima do botão
        
        clear_log_btn = QPushButton("🗑️ Limpar Log")
        clear_log_btn.clicked.connect(self.clear_log)
        clear_log_btn.setStyleSheet("""
            QPushButton {
                font-size: 13px;
                padding: 15px 30px;
                background: #34495e;
                border: 1px solid #8e44ad;
                border-radius: 4px;
                color: #ecf0f1;
                min-width: 160px;
                min-height: 45px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #4a5f7a;
                border-color: #9b59b6;
            }
            QPushButton:pressed {
                background: #2c3e50;
            }
        """)
        button_layout.addWidget(clear_log_btn)
        button_layout.addStretch()
        
        log_layout.addWidget(button_container)
        
        main_layout.addWidget(self.log_group)
        
        main_layout.addStretch()
        self.tab_widget.addTab(main_tab, "🎵 Processamento")

        # Advanced settings tab
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        advanced_layout.setSpacing(10)  # Reduzido espaçamento
        
        # Output folder section
        output_group = QGroupBox("Pasta de Saída")
        output_group.setStyleSheet("QGroupBox { border: 2px solid #8e44ad; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        output_layout = QHBoxLayout(output_group)
        output_layout.setSpacing(10)
        
        self.output_label = QLabel(f"📁 {self.output_folder}")
        self.output_label.setStyleSheet("font-size: 12px; color: #95a5a6;")
        output_layout.addWidget(self.output_label)
        
        change_folder_btn = QPushButton("📂 Alterar Pasta")
        change_folder_btn.clicked.connect(self.change_output_folder)
        change_folder_btn.setStyleSheet("font-size: 12px; padding: 8px;")
        output_layout.addWidget(change_folder_btn)
        
        advanced_layout.addWidget(output_group)
        
        # Keyframes section
        keyframes_group = QGroupBox("Keyframes (Avançado)")
        keyframes_group.setStyleSheet("QGroupBox { border: 2px solid #8e44ad; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        keyframes_layout = QVBoxLayout(keyframes_group)
        keyframes_layout.setSpacing(8)  # Reduzido espaçamento
        
        keyframes_info = QLabel("Configure pontos específicos no tempo para animação de objetos 3D")
        keyframes_info.setWordWrap(True)
        keyframes_layout.addWidget(keyframes_info)
        
        # Add keyframe button
        add_keyframe_btn = QPushButton("➕ Adicionar Keyframe")
        add_keyframe_btn.clicked.connect(self.add_keyframe_dialog)
        add_keyframe_btn.setStyleSheet("font-size: 12px; padding: 8px 15px; min-width: 150px;")
        keyframes_layout.addWidget(add_keyframe_btn)
        
        # Keyframes list
        self.keyframes_list = QListWidget()
        self.keyframes_list.setMaximumHeight(100)  # Reduzido altura
        keyframes_layout.addWidget(self.keyframes_list)
        
        advanced_layout.addWidget(keyframes_group)
        
        # Export/Import section
        io_group = QGroupBox("Importar/Exportar Configurações")
        io_group.setStyleSheet("QGroupBox { border: 2px solid #8e44ad; border-radius: 8px; margin-top: 10px; padding-top: 10px; }")
        io_layout = QHBoxLayout(io_group)
        io_layout.setSpacing(10)
        
        export_btn = QPushButton("📤 Exportar Config")
        export_btn.clicked.connect(self.export_config)
        export_btn.setStyleSheet("font-size: 12px; padding: 8px 15px; min-width: 120px;")
        io_layout.addWidget(export_btn)
        
        import_btn = QPushButton("📥 Importar Config")
        import_btn.clicked.connect(self.import_config)
        import_btn.setStyleSheet("font-size: 12px; padding: 8px 15px; min-width: 120px;")
        io_layout.addWidget(import_btn)
        
        advanced_layout.addWidget(io_group)
        
        # Video processor button
        video_btn = QPushButton("🎬 Abrir Processador de Vídeo")
        video_btn.clicked.connect(self.open_video_processor)
        video_btn.setStyleSheet("font-size: 14px; padding: 12px; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #e74c3c, stop:1 #c0392b);")
        advanced_layout.addWidget(video_btn)
        
        advanced_layout.addStretch()
        self.tab_widget.addTab(advanced_tab, "⚙️ Configurações Avançadas")
        
        layout.addWidget(self.tab_widget)

        # Initialize preset description
        self.update_preset_description(0)

    def log_message(self, message, level="INFO"):
        """Adiciona mensagem ao log com timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        color_map = {
            "INFO": "#00ff00",
            "WARNING": "#ffff00", 
            "ERROR": "#ff0000",
            "SUCCESS": "#00ffff"
        }
        
        color = color_map.get(level, "#00ff00")
        formatted_message = f'<span style="color: #888888;">[{timestamp}]</span> <span style="color: {color};">[{level}]</span> {message}'
        
        self.log_area.append(formatted_message)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def clear_log(self):
        """Limpa a área de log"""
        self.log_area.clear()

    def update_preset_description(self, index):
        """Atualiza a descrição do preset selecionado"""
        if index >= 0:
            description = self.format_combo.itemData(index)
            self.preset_description.setText(description)

    def change_output_folder(self):
        """Abre diálogo para selecionar pasta de saída"""
        folder = QFileDialog.getExistingDirectory(
            self, "Selecione a Pasta de Saída", self.output_folder
        )
        if folder:
            self.output_folder = folder
            self.output_label.setText(f"📁 {folder}")
            # Salvar preferência
            self.save_output_folder(folder)
            self.log_message(f"Pasta de saída alterada e salva: {folder}", "INFO")

    def browse_omni_file(self):
        """Abre diálogo para selecionar arquivo OMNI existente"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Selecione um arquivo OMNI", "", "OMNI Files (*.omni)"
        )
        if file_path:
            self.omni_input.setText(file_path)
            self.load_omni_info(file_path)

    def load_omni_info(self, omni_path):
        """Carrega informações do arquivo OMNI existente"""
        try:
            with open(omni_path, 'r', encoding='utf-8') as f:
                omni_data = json.load(f)
            
            # Extrair seed do arquivo OMNI
            if 'seed' in omni_data:
                self.seed_input.setText(str(omni_data['seed']))
                self.log_message(f"Seed carregada do OMNI: {omni_data['seed']}", "INFO")
            
            # Extrair keyframes se existirem
            if 'keyframes' in omni_data and omni_data['keyframes']:
                self.keyframes_list.clear()
                for kf in omni_data['keyframes']:
                    item_text = f"{kf['time']}s - {kf.get('object_name', 'N/A')} (A:{kf.get('azimuth', 0)}° E:{kf.get('elevation', 0)}°)"
                    item = QListWidgetItem(item_text)
                    item.setData(Qt.ItemDataRole.UserRole, kf)
                    self.keyframes_list.addItem(item)
                
                self.log_message(f"Carregados {len(omni_data['keyframes'])} keyframes do OMNI", "INFO")
            
            self.log_message(f"Arquivo OMNI carregado: {os.path.basename(omni_path)}", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"Erro ao carregar OMNI: {str(e)}", "ERROR")
            QMessageBox.warning(self, "Erro", f"Não foi possível carregar o arquivo OMNI:\n{str(e)}")

    def on_file_dropped(self, file_path):
        # Verificar se é arquivo OMNI
        if file_path.lower().endswith('.omni'):
            self.omni_input.setText(file_path)
            self.load_omni_info(file_path)
            self.file_label.setText(f"📁 {os.path.basename(file_path)} (OMNI)")
            self.process_btn.setEnabled(True)
            self.log_message(f"Arquivo OMNI carregado: {os.path.basename(file_path)}", "INFO")
        else:
            self.current_file = file_path
            self.file_label.setText(f"📁 {os.path.basename(file_path)}")
            self.process_btn.setEnabled(True)
            self.log_message(f"Arquivo carregado: {os.path.basename(file_path)}", "INFO")

    def start_processing(self):
        # Verificar se está usando arquivo OMNI existente
        omni_file = self.omni_input.text().strip()
        
        if omni_file and os.path.exists(omni_file):
            # Processar usando OMNI existente
            self.process_existing_omni(omni_file)
        elif not self.current_file:
            QMessageBox.warning(self, "Aviso", "Selecione um arquivo de áudio/vídeo ou um arquivo OMNI existente.")
            return
        else:
            # Processar novo arquivo
            self.process_new_file()

    def process_existing_omni(self, omni_path):
        """Processa usando arquivo OMNI existente"""
        try:
            with open(omni_path, 'r', encoding='utf-8') as f:
                omni_data = json.load(f)
            
            # Mapeamento usando nomes de exibição para nomes internos
            format_map = {
                "Binaural 360°": "binaural",
                "Cinema Binaural": "cinema_binaural",
                "Estéreo 3D": "static_binaural", 
                "Speaker 360°": "speaker_3d",
                "Speaker 3D (static)": "static_speaker"
            }
            
            output_format = format_map[self.format_combo.currentText()]
            seed_text = self.seed_input.text().strip()
            seed = seed_text if seed_text else omni_data.get('seed')
            
            # Get keyframes from list (se houver alterações) ou do OMNI
            keyframes = []
            if self.keyframes_list.count() > 0:
                for i in range(self.keyframes_list.count()):
                    item = self.keyframes_list.item(i)
                    keyframes.append(item.data(Qt.ItemDataRole.UserRole))
            else:
                keyframes = omni_data.get('keyframes', [])
            
            self.process_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.log_message(f"Processando OMNI existente: {os.path.basename(omni_path)}", "INFO")
            self.log_message(f"Formato: {self.format_combo.currentText()}", "INFO")
            self.log_message(f"Seed: {seed}", "INFO")
            self.log_message(f"Keyframes: {len(keyframes)}", "INFO")
            
            worker = AudioProcessorWorker('process_omni', omni_path, output_format, seed, keyframes, self.output_folder)
            worker.signals.progress.connect(self.progress_bar.setValue)
            worker.signals.status.connect(self.status_label.setText)
            worker.signals.finished.connect(self.on_processing_finished)
            worker.signals.error.connect(self.on_processing_error)
            worker.signals.log_message.connect(self.log_message)
            
            self.threadpool.start(worker)
            
        except Exception as e:
            self.log_message(f"Erro ao processar OMNI: {str(e)}", "ERROR")
            QMessageBox.critical(self, "Erro", f"Não foi possível processar o arquivo OMNI:\n{str(e)}")

    def process_new_file(self):
        """Processa novo arquivo de áudio/vídeo"""
        # Mapeamento usando nomes de exibição para nomes internos
        format_map = {
            "Binaural 360°": "binaural",
            "Cinema Binaural": "cinema_binaural",
            "Estéreo 3D": "static_binaural", 
            "Speaker 360°": "speaker_3d",
            "Speaker3D (static)": "static_speaker"
        }
        
        output_format = format_map[self.format_combo.currentText()]
        seed_text = self.seed_input.text().strip()
        seed = seed_text if seed_text else None
        
        # Get keyframes from list
        keyframes = []
        for i in range(self.keyframes_list.count()):
            item = self.keyframes_list.item(i)
            keyframes.append(item.data(Qt.ItemDataRole.UserRole))
        
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.log_message(f"Iniciando processamento com formato: {self.format_combo.currentText()}", "INFO")
        self.log_message(f"Pasta de saída: {self.output_folder}", "INFO")
        
        worker = AudioProcessorWorker('auto', self.current_file, output_format, seed, keyframes, self.output_folder)
        worker.signals.progress.connect(self.progress_bar.setValue)
        worker.signals.status.connect(self.status_label.setText)
        worker.signals.finished.connect(self.on_processing_finished)
        worker.signals.error.connect(self.on_processing_error)
        worker.signals.log_message.connect(self.log_message)
        
        self.threadpool.start(worker)

    def add_keyframe_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Adicionar Keyframe")
        dialog.setFixedSize(400, 300)
        layout = QVBoxLayout(dialog)
        
        # Time input
        time_layout = QHBoxLayout()
        time_layout.addWidget(QLabel("Tempo (segundos):"))
        time_input = QDoubleSpinBox()
        time_input.setRange(0, 9999)
        time_input.setDecimals(2)
        time_layout.addWidget(time_input)
        layout.addLayout(time_layout)
        
        # Object name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Nome do Objeto:"))
        name_input = QLineEdit()
        name_input.setPlaceholderText("Ex: vocals, drums, bass...")
        name_layout.addWidget(name_input)
        layout.addLayout(name_layout)
        
        # Azimuth input
        azi_layout = QHBoxLayout()
        azi_layout.addWidget(QLabel("Azimute (0-360°):"))
        azi_input = QDoubleSpinBox()
        azi_input.setRange(0, 360)
        azi_input.setDecimals(1)
        azi_layout.addWidget(azi_input)
        layout.addLayout(azi_layout)
        
        # Elevation input
        ele_layout = QHBoxLayout()
        ele_layout.addWidget(QLabel("Elevação (-90 a 90°):"))
        ele_input = QDoubleSpinBox()
        ele_input.setRange(-90, 90)
        ele_input.setDecimals(1)
        ele_layout.addWidget(ele_input)
        layout.addLayout(ele_layout)
        
        # Buttons
        buttons = QHBoxLayout()
        ok_btn = QPushButton("Adicionar")
        cancel_btn = QPushButton("Cancelar")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addLayout(buttons)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            keyframe = {
                "time": time_input.value(),
                "object_name": name_input.text(),
                "azimuth": azi_input.value(),
                "elevation": ele_input.value()
            }
            
            item_text = f"{keyframe['time']}s - {keyframe['object_name']} (A:{keyframe['azimuth']}° E:{keyframe['elevation']}°)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, keyframe)
            self.keyframes_list.addItem(item)

    def export_config(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Configurações", "", "JSON Files (*.json)"
        )
        if file_path:
            config = {
                "seed": self.seed_input.text(),
                "format": self.format_combo.currentText(),
                "keyframes": []
            }
            
            for i in range(self.keyframes_list.count()):
                item = self.keyframes_list.item(i)
                config["keyframes"].append(item.data(Qt.ItemDataRole.UserRole))
            
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            QMessageBox.information(self, "Sucesso", "Configurações exportadas com sucesso!")

    def import_config(self):
        file_path,  = QFileDialog.getOpenFileName(
            self, "Importar Configurações", "", "JSON Files (*.json)"
        )
        if file_path:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            self.seed_input.setText(config.get("seed", ""))
            
            # Set format if exists
            format_index = self.format_combo.findText(config.get("format", ""))
            if format_index >= 0:
                self.format_combo.setCurrentIndex(format_index)
            
            # Clear and add keyframes
            self.keyframes_list.clear()
            for kf in config.get("keyframes", []):
                item_text = f"{kf['time']}s - {kf['object_name']} (A:{kf['azimuth']}° E:{kf['elevation']}°)"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, kf)
                self.keyframes_list.addItem(item)
            
            QMessageBox.information(self, "Sucesso", "Configurações importadas com sucesso!")

    def open_video_processor(self):
        try:
            from video_processor import VideoProcessorWindow
            self.video_window = VideoProcessorWindow()
            self.video_window.show()
        except ImportError:
            QMessageBox.warning(self, "Erro", "Módulo de processamento de vídeo não encontrado.")

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
    app.setApplicationName("Omni Audio Processor")
    
    # Set app icon and style
    app.setStyle('Fusion')
    
    # Set application icon
    icon_path = os.path.join(base_path, "icons", "app_icon.ico")
    icon_path_png = os.path.join(base_path, "icons", "app_icon.png")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    elif os.path.exists(icon_path_png):
        app.setWindowIcon(QIcon(icon_path_png))
    
    window = OmniAudioProcessor()
    
    # Set window icon
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))
    elif os.path.exists(icon_path_png):
        window.setWindowIcon(QIcon(icon_path_png))
    
    # Ensure normal window with title bar
    window.setWindowFlags(Qt.WindowType.Window)
    window.showNormal()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
