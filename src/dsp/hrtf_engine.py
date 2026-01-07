import os
import glob
import numpy as np
import soundfile as sf
from scipy.spatial import KDTree
class HRTFEngine:
    def __init__(self, hrtf_path):
        self.hrtf_path = hrtf_path
        self.hrirs = {}
        self.coords = []
        self.tree = None

    def load(self):
        path = os.path.join(self.hrtf_path, "D1_HRIR_WAV", "*.wav")
        files = glob.glob(path)
        if not files: return False
        
        print(f"Carregando {len(files)} IRs...")
        for f in files:
            try:
                # Parser robusto para achar azi/ele no nome do arquivo
                name = os.path.basename(f)
                # Exemplo esperado: ...azi_270,0_ele_0,0.wav
                azi_str = name.split('azi_')[1].split('_')[0].replace(',', '.')
                ele_str = name.split('ele_')[1].replace('.wav', '').replace(',', '.')
                
                azi, ele = float(azi_str), float(ele_str)
                if azi < 0: azi += 360
                
                data, _ = sf.read(f)
                self.hrirs[(ele, azi)] = data
                self.coords.append([ele, azi])
            except: continue
            
        if not self.coords: return False
        self.coords = np.array(self.coords)
        self.tree = KDTree(self.coords)
        return True

    def get_ir(self, azi, ele):
        dists, idxs = self.tree.query([ele, azi], k=3)
        
        if dists[0] < 0.1:
            ir = self.hrirs[tuple(self.coords[idxs[0]])]
            return ir[:, 0], ir[:, 1]
        
        weights = 1.0 / (dists + 1e-6)
        weights /= np.sum(weights)
        
        out_l, out_r = np.zeros(256), np.zeros(256)
        for i, idx in enumerate(idxs):
            ir = self.hrirs[tuple(self.coords[idx])]
            l = min(len(ir), 256)
            out_l[:l] += ir[:l, 0] * weights[i]
            out_r[:l] += ir[:l, 1] * weights[i]
        return out_l, out_r