import json
import random
import hashlib
from datetime import datetime

class OMNIFormat:
    # Mapeamento padrão de ângulos para Cinema Virtual
    SURROUND_MAP = {
        "FL":  {"azi": 30,  "role": "spatial"},      # Front Left
        "FR":  {"azi": 330, "role": "spatial"},     # Front Right
        "FC":  {"azi": 0,   "role": "anchor"},      # Front Center
        "LFE": {"azi": 0,   "role": "lfe_focused"}, # Subwoofer
        "SL":  {"azi": 110, "role": "spatial"},     # Surround Left
        "SR":  {"azi": 250, "role": "spatial"},     # Surround Right
        "BL":  {"azi": 150, "role": "spatial"},     # Back Left (7.1)
        "BR":  {"azi": 210, "role": "spatial"}      # Back Right (7.1)
    }

    @staticmethod
    def _detect_role(name):
        name = name.lower()
        if any(x in name for x in ['vocal', 'voice', 'voz', 'lead', 'speech', 'dialog', 'fc']):
            return 'anchor'
        elif any(x in name for x in ['bass', 'baixo', 'kick', 'sub', '808', 'lfe']):
            return 'lfe_focused'
        elif any(x in name for x in ['air', 'pad', 'ambience', 'noise', 'fx', 'atm']):
            return 'ethereal'
        return 'spatial'

    @staticmethod
    def _generate_seed(input_string=None):
        """Gera seed determinística a partir de uma string ou aleatória"""
        if input_string:
            return hashlib.md5(input_string.encode()).hexdigest()[:16]
        else:
            return hashlib.md5(str(random.random()).encode()).hexdigest()[:16]

    @staticmethod
    def create_multi_stem_omni(stems_list, duration, sr, path, is_surround=False, seed=None, keyframes=None):
        # Gera seed se não fornecida
        if seed is None:
            seed = OMNIFormat._generate_seed()
        
        # Define keyframes padrão se não fornecidos
        if keyframes is None:
            keyframes = []
        
        data = {
            "version": "5.1 (Enhanced with Keyframes)",
            "created_at": datetime.now().isoformat(),
            "seed": seed,
            "duration": duration,
            "sample_rate": sr,
            "keyframes": keyframes,
            "objects": []
        }
        
        # Configura random com seed para reproduzibilidade
        random.seed(seed)
        
        for s in stems_list:
            name_key = s['name'].upper()
            
            # Se for surround, usa posições fixas
            if is_surround and name_key in OMNIFormat.SURROUND_MAP:
                config = OMNIFormat.SURROUND_MAP[name_key]
                role = config['role']
                physics = {
                    "speed": 0.0,           # Estático
                    "direction": 1,
                    "start_phase": config['azi'],
                    "randomness": 0.0        # Sem variação
                }
            else:
                # Lógica original para música/stems
                role = OMNIFormat._detect_role(s['name'])
                speed = random.uniform(0.3, 1.2)
                direction = random.choice([1, -1])
                start_phase = random.uniform(0, 360)
                randomness = 0.2
                
                if role == 'anchor':
                    speed = 0.05; randomness = 0.02; start_phase = 0
                elif role == 'ethereal':
                    speed = 1.2; randomness = 0.5
                
                physics = {
                    "speed": speed,
                    "direction": direction,
                    "start_phase": start_phase,
                    "randomness": randomness
                }

            # Adiciona keyframes específicos do objeto se existirem
            obj_keyframes = [kf for kf in keyframes if kf.get('object_name') == s['name']]
            
            data["objects"].append({
                "name": s['name'],
                "file": s['file'],
                "role": role,
                "physics": physics,
                "keyframes": obj_keyframes
            })
        
        # Reseta random para não afetar outras partes do sistema
        random.seed()
            
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return seed

    @staticmethod
    def add_keyframe_to_omni(omni_path, object_name, time, azimuth=None, elevation=None, speed=None, randomness=None):
        """Adiciona keyframe a um arquivo OMNI existente"""
        with open(omni_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        keyframe = {
            "time": time,
            "object_name": object_name
        }
        
        if azimuth is not None:
            keyframe["azimuth"] = azimuth
        if elevation is not None:
            keyframe["elevation"] = elevation
        if speed is not None:
            keyframe["speed"] = speed
        if randomness is not None:
            keyframe["randomness"] = randomness
        
        data["keyframes"].append(keyframe)
        
        with open(omni_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def get_keyframes_at_time(omni_path, time):
        """Retorna keyframes ativos em um determinado tempo"""
        with open(omni_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        active_keyframes = []
        for kf in data.get('keyframes', []):
            if kf['time'] <= time:
                active_keyframes.append(kf)
        
        return active_keyframes