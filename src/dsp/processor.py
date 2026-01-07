import numpy as np
from scipy import signal
from core.math_utils import calculate_5_1_gains, calculate_stereo_gains, cartesian_to_spherical

class OMNIProcessor:
    def __init__(self, hrtf_engine):
        self.hrtf = hrtf_engine

    def _apply_hpf(self, audio, sr, cutoff=3000):
        sos = signal.butter(2, cutoff, 'hp', fs=sr, output='sos')
        return signal.sosfilt(sos, audio)

    def _calculate_procedural_pos(self, physics, time, role):
        speed = physics.get('speed', 1.0)
        direction = physics.get('direction', 1)
        start_phase = physics.get('start_phase', 0)
        angle = np.radians(start_phase) + (direction * speed * 0.5 * time)
        x, y = np.cos(angle), np.sin(angle)
        z = 0.7 + (np.sin(time * 0.5) * 0.2) if role == 'ethereal' else np.sin(time * 0.3) * 0.3
        return cartesian_to_spherical(x, y, z)

    def process_object_binaural(self, mono_audio, sr, obj_config):
        chunk_size, hop = 4096, 2048
        role = obj_config.get('role', 'spatial')
        if role == 'ethereal': mono_audio = self._apply_hpf(mono_audio, sr)
        out_l, out_r = np.zeros(len(mono_audio) + chunk_size), np.zeros(len(mono_audio) + chunk_size)
        window = np.hanning(chunk_size)

        for pos in range(0, len(mono_audio) - chunk_size, hop):
            chunk = mono_audio[pos:pos+chunk_size] * window
            t = (pos + chunk_size/2) / sr
            if role == 'anchor': azi, ele = 0.0, 0.0
            elif role == 'lfe_focused': azi, ele = 0.0, -10.0
            else: azi, ele = self._calculate_procedural_pos(obj_config.get('physics', {}), t, role)
            ir_l, ir_r = self.hrtf.get_ir(azi, ele)
            out_l[pos:pos+chunk_size] += signal.fftconvolve(chunk, ir_l, mode='same')
            out_r[pos:pos+chunk_size] += signal.fftconvolve(chunk, ir_r, mode='same')
        return out_l[:len(mono_audio)], out_r[:len(mono_audio)]

    # --- MODO: ESTÉREO WIDE (90°) ---
    def process_stereo_fixed(self, stereo_audio, sr):
        chunk_size, hop = 4096, 2048
        l_in, r_in = stereo_audio[:, 0], stereo_audio[:, 1]
        out_l, out_r = np.zeros(len(l_in) + chunk_size), np.zeros(len(l_in) + chunk_size)
        window = np.hanning(chunk_size)
        ir_l_left, ir_r_left = self.hrtf.get_ir(90, 0)
        ir_l_right, ir_r_right = self.hrtf.get_ir(270, 0)
        shadow = 0.6
        ir_r_left *= shadow
        ir_l_right *= shadow
        for pos in range(0, len(l_in) - chunk_size, hop):
            cl, cr = l_in[pos:pos+chunk_size] * window, r_in[pos:pos+chunk_size] * window
            out_l[pos:pos+chunk_size] += signal.fftconvolve(cl, ir_l_left, mode='same') + signal.fftconvolve(cr, ir_l_right, mode='same')
            out_r[pos:pos+chunk_size] += signal.fftconvolve(cl, ir_r_left, mode='same') + signal.fftconvolve(cr, ir_r_right, mode='same')
        return out_l[:len(l_in)], out_r[:len(l_in)]

    # --- MODO: CINEMA BINAURAL (30° + Sala) ---
    def process_stereo_cinema(self, stereo_audio, sr):
        chunk_size, hop = 4096, 2048
        l_in, r_in = stereo_audio[:, 0], stereo_audio[:, 1]
        out_l, out_r = np.zeros(len(l_in) + chunk_size), np.zeros(len(l_in) + chunk_size)
        window = np.hanning(chunk_size)
        
        # Ângulos de Home Theater (30° à frente)
        ir_l_left, ir_r_left = self.hrtf.get_ir(30, 0) 
        ir_l_right, ir_r_right = self.hrtf.get_ir(330, 0)

        for pos in range(0, len(l_in) - chunk_size, hop):
            cl, cr = l_in[pos:pos+chunk_size] * window, r_in[pos:pos+chunk_size] * window
            out_l[pos:pos+chunk_size] += signal.fftconvolve(cl, ir_l_left, mode='same') + signal.fftconvolve(cr, ir_l_right, mode='same')
            out_r[pos:pos+chunk_size] += signal.fftconvolve(cl, ir_r_left, mode='same') + signal.fftconvolve(cr, ir_r_right, mode='same')

        # Simulação de Sala (Early Reflections)
        delay_samples = int(0.022 * sr) # 22ms
        sos = signal.butter(2, 4000, 'lp', fs=sr, output='sos')
        ref_l = signal.sosfilt(sos, np.pad(out_r[:len(l_in)], (delay_samples, 0))[:len(l_in)]) * 0.25
        ref_r = signal.sosfilt(sos, np.pad(out_l[:len(l_in)], (delay_samples, 0))[:len(l_in)]) * 0.25
        return out_l[:len(l_in)] + ref_l, out_r[:len(l_in)] + ref_r

    def apply_xtc(self, stereo_audio, sr):
        delay = int((0.15 * np.sin(np.radians(20)) / 343.0) * sr)
        sos = signal.butter(2, 2200, 'lp', fs=sr, output='sos')
        l_in, r_in = stereo_audio[:, 0], stereo_audio[:, 1]
        l_c = signal.sosfilt(sos, np.pad(l_in, (delay, 0))[:len(l_in)])
        r_c = signal.sosfilt(sos, np.pad(r_in, (delay, 0))[:len(r_in)])
        return np.stack([(l_in - r_c * 0.55) * 1.1, (r_in - l_c * 0.55) * 1.1], axis=1)