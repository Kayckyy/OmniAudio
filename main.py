import sys
import os
import json
import soundfile as sf
import numpy as np
import subprocess

# Configuração de caminhos
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_path, 'src'))

from utils.audio_io import load_audio
from core.omni_format import OMNIFormat
from dsp.hrtf_engine import HRTFEngine
from dsp.processor import OMNIProcessor
from utils.separator import AudioSeparator

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm')

def extract_audio_from_video(video_path):
    if not video_path.lower().endswith(VIDEO_EXTENSIONS): return video_path
    output_wav = os.path.splitext(video_path)[0] + "_extracted.wav"
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=channels', '-of', 'csv=p=0', video_path], capture_output=True, text=True)
    channels = int(probe.stdout.strip()) if probe.stdout.strip() else 2
    cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', output_wav, '-y']
    if channels <= 2: cmd.insert(-2, '-ac'); cmd.insert(-2, '2')
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_wav

def split_multichannel(wav_path, output_dir):
    info = sf.info(wav_path)
    if info.channels <= 2: return None
    layout = ["FL", "FR", "FC", "LFE", "SL", "SR", "BL", "BR"][:info.channels]
    stems = []
    for i, name in enumerate(layout):
        ch_file = os.path.join(output_dir, f"{name}.wav")
        cmd = ['ffmpeg', '-i', wav_path, '-af', f'pan=mono|c0=c{i}', ch_file, '-y']
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        stems.append({'name': name, 'file': ch_file})
    return stems

def merge_audio_to_video(video_path, audio_3d_path):
    output_video = os.path.splitext(video_path)[0] + "_Omni_3D.mp4"
    cmd = ['ffmpeg', '-i', video_path, '-i', audio_3d_path, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', output_video, '-y']
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"✅ Vídeo final: {output_video}")
        if os.path.exists(audio_3d_path): os.remove(audio_3d_path)
        ext_wav = video_path.replace(os.path.splitext(video_path)[1], "_extracted.wav")
        if os.path.exists(ext_wav): os.remove(ext_wav)
    except: print("❌ Erro no Muxing.")

def main():
    args = sys.argv
    if len(args) < 2: return print("Uso: python main.py [auto|create_multi|process_multi] [arquivo]")
    
    mode = args[1]
    hrtf = HRTFEngine(os.path.join(base_path, "data", "hrtf"))
    if mode == 'process_multi': hrtf.load()
    processor = OMNIProcessor(hrtf)

    if mode in ['auto', 'create_multi']:
        input_file = " ".join(args[2:]).strip('"').strip("'")
        work_file = extract_audio_from_video(input_file)
        info = sf.info(work_file)
        proj_dir = os.path.join("temp", os.path.splitext(os.path.basename(input_file))[0])
        if not os.path.exists(proj_dir): os.makedirs(proj_dir)
        
        is_surround = info.channels > 2
        stems = split_multichannel(work_file, proj_dir) if is_surround else (AudioSeparator.separate(work_file, "temp")[1] if mode == 'auto' else [{'name': 'Stereo', 'file': work_file}])
        
        if stems:
            omni_path = os.path.join(proj_dir, os.path.basename(proj_dir)+".omni")
            OMNIFormat.create_multi_stem_omni(stems, info.duration, info.samplerate, omni_path, is_surround=is_surround)
            print(f"✅ Projeto: {omni_path}")

    elif mode == 'process_multi':
        omni_path = args[2]
        if not os.path.exists(omni_path):
            base = os.path.splitext(os.path.basename(omni_path))[0]
            omni_path = os.path.join("temp", base, base + ".omni")
        
        output_format = 'binaural'
        if '--format' in args: output_format = args[args.index('--format')+1].lower().replace('-', '_')

        with open(omni_path, 'r') as f: data = json.load(f)
        sr, master_len = data['sample_rate'], int(data['duration'] * data['sample_rate'])
        master_mix = np.zeros((master_len, 2))

        for obj in data['objects']:
            audio, _ = load_audio(obj['file'])
            if len(audio) > master_len: audio = audio[:master_len]
            
            if output_format == 'cinema_binaural':
                if len(audio.shape) == 1: audio = np.stack([audio, audio], axis=1)
                l, r = processor.process_stereo_cinema(audio, sr)
                res = np.stack([l, r], axis=1)
            elif output_format in ['static_binaural', 'static_speaker']:
                if len(audio.shape) == 1: audio = np.stack([audio, audio], axis=1)
                l, r = processor.process_stereo_fixed(audio, sr)
                res = np.stack([l, r], axis=1)
            else:
                if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
                l, r = processor.process_object_binaural(audio, sr, obj)
                res = np.stack([l, r], axis=1)
            
            l_mix = min(len(res), master_len)
            master_mix[:l_mix] += res[:l_mix]

        if output_format in ['speaker_3d', 'static_speaker']: master_mix = processor.apply_xtc(master_mix, sr)
        peak = np.max(np.abs(master_mix))
        if peak > 0: master_mix /= (peak + 1e-6)
        
        out_name = f"{os.path.splitext(omni_path)[0]}_{output_format}.wav"
        sf.write(out_name, master_mix, sr)

        # Muxing
        video_to_merge = None
        base_name = os.path.splitext(os.path.basename(omni_path))[0].replace('_extracted', '')
        for sd in [os.path.dirname(omni_path), os.path.dirname(os.path.dirname(omni_path)), "."]:
            for ext in VIDEO_EXTENSIONS:
                potential = os.path.join(sd, base_name + ext)
                if os.path.exists(potential): video_to_merge = potential; break
            if video_to_merge: break
        if video_to_merge: merge_audio_to_video(video_to_merge, out_name)

if __name__ == "__main__": 
  main()