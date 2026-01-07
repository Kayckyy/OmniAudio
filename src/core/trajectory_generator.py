import numpy as np
import random
try:
    from opensimplex import OpenSimplex
except ImportError:
    print("⚠ opensimplex não instalado. Instale: pip install opensimplex")
    OpenSimplex = None

class TrajectoryGenerator:
    """
    Gerador de траектórias procedurais com seeds, noise e keyframes
    """
    
    def __init__(self, seed=None):
        self.seed = seed if seed is not None else random.randint(0, 999999)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        if OpenSimplex:
            self.noise = OpenSimplex(seed=self.seed)
        else:
            self.noise = None
    
    def generate(self, duration, role='spatial', preset='organic', keyframes=None):
        """
        Gera траектória baseada em preset ou keyframes
        
        Args:
            duration: Duração em segundos
            role: 'anchor', 'spatial', 'ethereal', 'lfe_focused'
            preset: 'organic', 'chaotic', 'smooth', 'curious', 'aggressive'
            keyframes: Lista de dicts {time, azi, ele} pra interpolação manual
        
        Returns:
            dict com physics params
        """
        
        if keyframes:
            return self._generate_from_keyframes(keyframes, duration)
        
        return self._generate_procedural(duration, role, preset)
    
    def _generate_procedural(self, duration, role, preset):
        """Gera траектória procedural longa e variada"""
        
        # Presets definem comportamento base
        presets_config = {
            'organic': {
                'base_speed': 0.3,
                'noise_scale': 0.5,
                'noise_strength': 0.3,
                'harmonics': 3
            },
            'chaotic': {
                'base_speed': 1.2,
                'noise_scale': 2.0,
                'noise_strength': 0.8,
                'harmonics': 5
            },
            'smooth': {
                'base_speed': 0.15,
                'noise_scale': 0.2,
                'noise_strength': 0.1,
                'harmonics': 2
            },
            'curious': {
                'base_speed': 0.6,
                'noise_scale': 1.0,
                'noise_strength': 0.5,
                'harmonics': 4
            },
            'aggressive': {
                'base_speed': 1.5,
                'noise_scale': 3.0,
                'noise_strength': 0.9,
                'harmonics': 6
            }
        }
        
        config = presets_config.get(preset, presets_config['organic'])
        
        # Ajustes por role
        if role == 'anchor':
            config['base_speed'] = 0.05
            config['noise_strength'] = 0.02
        elif role == 'ethereal':
            config['base_speed'] *= 1.5
            config['noise_strength'] *= 1.3
        
        # Gerar parâmetros complexos pra траектória longa
        harmonics = []
        for i in range(config['harmonics']):
            harmonics.append({
                'freq': random.uniform(0.1, 2.0),
                'amp': random.uniform(0.3, 1.0),
                'phase': random.uniform(0, 2 * np.pi)
            })
        
        return {
            'type': 'procedural_long',
            'seed': self.seed,
            'preset': preset,
            'base_speed': config['base_speed'],
            'noise_scale': config['noise_scale'],
            'noise_strength': config['noise_strength'],
            'harmonics': harmonics,
            'direction': random.choice([1, -1])
        }
    
    def _generate_from_keyframes(self, keyframes, duration):
        """Gera траектória interpolando keyframes"""
        keyframes = sorted(keyframes, key=lambda k: k['time'])
        
        if keyframes[-1]['time'] < duration:
            keyframes.append({
                'time': duration,
                'azi': keyframes[0]['azi'],
                'ele': keyframes[0]['ele']
            })
        
        return {
            'type': 'keyframes',
            'keyframes': keyframes,
            'interpolation': 'catmull-rom'
        }
    
    def calculate_position(self, physics, time, role):
        """
        Calcula posição (azi, ele) em tempo real
        """
        
        if physics.get('type') == 'keyframes':
            return self._interpolate_keyframes(physics, time)
        
        # Траектória procedural longa
        base_speed = physics['base_speed']
        direction = physics['direction']
        noise_scale = physics['noise_scale']
        noise_strength = physics['noise_strength']
        harmonics = physics['harmonics']
        
        # Base circular com múltiplas frequências (evita loop óbvio)
        angle = 0
        for h in harmonics:
            angle += h['amp'] * np.sin(h['freq'] * time + h['phase'])
        
        angle = angle * direction * base_speed
        
        # Adicionar Perlin noise pra organicidade
        if self.noise:
            noise_x = self.noise.noise2(time * noise_scale, 0) * noise_strength
            noise_y = self.noise.noise2(time * noise_scale, 100) * noise_strength
        else:
            # Fallback se opensimplex não instalado
            noise_x = np.sin(time * 2.3 * noise_scale) * noise_strength
            noise_y = np.cos(time * 3.7 * noise_scale) * noise_strength
        
        x = np.cos(angle) + noise_x
        y = np.sin(angle) + noise_y
        
        # Elevação com variação
        if role == 'ethereal':
            base_ele = 45 + (np.sin(time * 0.4) * 20)
            if self.noise:
                base_ele += self.noise.noise2(time * 0.3, 200) * 15
            ele = base_ele
        else:
            base_ele = np.sin(time * 0.3) * 25
            if self.noise:
                base_ele += self.noise.noise2(time * 0.2, 300) * 10
            ele = base_ele
        
        # Converter pra azimuth
        azi = np.degrees(np.arctan2(y, x)) % 360
        
        return azi, ele
    
    def _interpolate_keyframes(self, physics, time):
        """Interpolação Catmull-Rom entre keyframes"""
        keyframes = physics['keyframes']
        
        for i in range(len(keyframes) - 1):
            if keyframes[i]['time'] <= time <= keyframes[i+1]['time']:
                t0 = keyframes[i]['time']
                t1 = keyframes[i+1]['time']
                
                t = (time - t0) / (t1 - t0)
                
                p1 = keyframes[i]
                p2 = keyframes[i+1]
                p0 = keyframes[i-1] if i > 0 else p1
                p3 = keyframes[i+2] if i+2 < len(keyframes) else p2
                
                azi = self._catmull_rom(
                    p0['azi'], p1['azi'], p2['azi'], p3['azi'], t
                )
                ele = self._catmull_rom(
                    p0['ele'], p1['ele'], p2['ele'], p3['ele'], t
                )
                
                return azi % 360, ele
        
        return keyframes[0]['azi'], keyframes[0]['ele']
    
    def _catmull_rom(self, p0, p1, p2, p3, t):
        """Interpolação Catmull-Rom (suave)"""
        return 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2*p0 - 5*p1 + 4*p2 - p3) * t**2 +
            (-p0 + 3*p1 - 3*p2 + p3) * t**3
        )