import os
import subprocess
import tempfile
import shutil

class VideoProcessor:
    
    @staticmethod
    def check_ffmpeg():
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         check=True)
            return True
        except:
            return False
    
    @staticmethod
    def extract_audio(video_path, output_audio_path, preserve_quality=True):
        print(f"\n[Vídeo] Extraindo áudio de: {video_path}")
        
        if not VideoProcessor.check_ffmpeg():
            print("❌ ffmpeg não encontrado. Instale: pkg install ffmpeg")
            return False, None, None
        
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate,channels,codec_name',
            '-of', 'json',
            video_path
        ]
        
        try:
            result = subprocess.run(probe_cmd, 
                                  capture_output=True, 
                                  text=True,
                                  check=True)
            
            import json
            info = json.loads(result.stdout)
            
            if not info.get('streams'):
                print("❌ Vídeo não tem áudio")
                return False, None, None
            
            stream = info['streams'][0]
            sr = int(stream.get('sample_rate', 48000))
            channels = int(stream.get('channels', 2))
            codec = stream.get('codec_name', 'unknown')
            
            print(f"  Codec: {codec}, SR: {sr}Hz, Canais: {channels}")
            
        except Exception as e:
            print(f"  ⚠️  Erro ao detectar info: {e}")
            sr, channels = 48000, 2
        
        if preserve_quality:
            extract_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'flac',
                '-y',
                output_audio_path
            ]
        else:
            extract_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-y',
                output_audio_path
            ]
        
        try:
            subprocess.run(extract_cmd, 
                         check=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
            
            print(f"  ✓ Áudio extraído: {output_audio_path}")
            return True, sr, channels
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao extrair áudio: {e}")
            return False, None, None
    
    @staticmethod
    def remux_video(original_video, processed_audio, output_video, copy_video_codec=True):
        print(f"\n[Vídeo] Remuxando com áudio processado...")
        
        if copy_video_codec:
            remux_cmd = [
                'ffmpeg',
                '-i', original_video,
                '-i', processed_audio,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '320k',
                '-shortest',
                '-y',
                output_video
            ]
        else:
            remux_cmd = [
                'ffmpeg',
                '-i', original_video,
                '-i', processed_audio,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'aac',
                '-b:a', '320k',
                '-shortest',
                '-y',
                output_video
            ]
        
        try:
            print("  Processando (pode demorar)...", end=" ", flush=True)
            
            subprocess.run(remux_cmd,
                         check=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
            
            print("✓")
            print(f"  ✓ Vídeo final: {output_video}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Erro ao remuxar: {e}")
            return False
    
    @staticmethod
    def process_video_full(video_path, omni_path, output_video, 
                          processor, omni_data, 
                          mode='binaural',
                          keep_intermediate=False):
        """
        Pipeline completo com seleção de modo
        
        Args:
            mode: 'binaural', 'static_binaural', 'speaker_3d', 'surround'
        """
        temp_dir = tempfile.mkdtemp(prefix='omni_video_')
        
        try:
            extracted_audio = os.path.join(temp_dir, 'extracted_audio.flac')
            processed_audio = os.path.join(temp_dir, 'processed_audio.wav')
            
            success, sr, channels = VideoProcessor.extract_audio(
                video_path, 
                extracted_audio
            )
            
            if not success:
                return False
            
            print(f"\n[OMNI] Processando áudio (modo: {mode})...")
            
            if mode == 'static_binaural':
                # Modo static: usar áudio extraído direto
                processor.process_static_binaural_streaming(
                    audio_path=extracted_audio,
                    sr=omni_data['sample_rate'],
                    output_path=processed_audio
                )
            
            elif mode == 'speaker_3d':
                # Modo speaker 3D
                layout = omni_data.get('speaker_layout', '5.1')
                processor.process_speaker_3d_streaming(
                    audio_path=extracted_audio,
                    sr=omni_data['sample_rate'],
                    output_path=processed_audio,
                    speaker_layout=layout
                )
            
            else:
                # Modo binaural/surround com stems
                processed_stems = []
                
                for obj in omni_data['objects']:
                    stem_path = obj['file']
                    
                    if not os.path.exists(stem_path):
                        stem_path = extracted_audio
                    
                    temp_out = os.path.join(temp_dir, f"proc_{obj['name']}.wav")
                    
                    if mode == 'surround':
                        processor.process_surround_streaming(
                            audio_path=stem_path,
                            sr=omni_data['sample_rate'],
                            obj_config=obj,
                            output_path=temp_out,
                            format=omni_data.get('surround_format', '5.1')
                        )
                    else:
                        processor.process_object_binaural_streaming(
                            audio_path=stem_path,
                            sr=omni_data['sample_rate'],
                            obj_config=obj,
                            output_path=temp_out
                        )
                    
                    processed_stems.append(temp_out)
                    
  processor.mix_stems_streaming(
                    stem_paths=processed_stems,
                    output_path=processed_audio,
                    sr=omni_data['sample_rate']
                )
            
            # Remuxar vídeo
            success = VideoProcessor.remux_video(
                original_video=video_path,
                processed_audio=processed_audio,
                output_video=output_video,
                copy_video_codec=True
            )
            
            if keep_intermediate:
                keep_dir = f"{output_video}_intermediate"
                os.makedirs(keep_dir, exist_ok=True)
                shutil.copy(extracted_audio, keep_dir)
                shutil.copy(processed_audio, keep_dir)
                print(f"\n  ℹ️  Intermediários salvos em: {keep_dir}")
            
            return success
            
        finally:
            if not keep_intermediate:
                print(f"\n[Limpeza] Removendo temporários...")
                shutil.rmtree(temp_dir)