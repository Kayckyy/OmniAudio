import numpy as np
def cartesian_to_spherical(x, y, z):
    """
    Converte coordenadas Cartesianas para Esféricas.
    Sistema Omni: 0°=Frente, 90°=Direita, 180°=Trás, 270°=Esquerda.
    """
    # Usando arctan2(x, y) para que y+ seja 0 graus (Frente)
    azimuth = np.degrees(np.arctan2(x, y))
    azimuth = azimuth % 360 # Garante range [0, 360]
    
    distance = np.sqrt(x**2 + y**2 + z**2) 
    elevation = 0
    if distance > 0:
        elevation = np.degrees(np.arcsin(np.clip(z / distance, -1, 1)))
        
    return azimuth, elevation

def calculate_stereo_gains(azimuth):
    """
    Calcula ganhos Estéreo (L/R) usando 'Constant Power Pan Law'.
    Isso garante que o som não perca volume ao passar pelo centro.
    
    Mapeamento:
    0° (Frente)   -> L=70%, R=70% (Centro)
    90° (Direita) -> L=0%,   R=100%
    180° (Trás)   -> L=70%, R=70% (Centro)
    270° (Esquerda)-> L=100%, R=0%
    """
    # Converte azimute para radianos
    rads = np.radians(azimuth % 360)
    
    # Projeta o círculo 3D no eixo X estéreo (-1 a +1)
    # sin(0) = 0 (Centro), sin(90) = 1 (Dir), sin(270) = -1 (Esq)
    pan_position = np.sin(rads) 
    
    # Normaliza de [-1, 1] para [0, 1]
    # 0 = Esquerda Total, 1 = Direita Total, 0.5 = Centro
    norm_pan = (pan_position + 1) / 2.0 
    
    # Lei da Raiz Quadrada (Potência Constante)
    # Usa pi/2 (90 graus) para fazer a curva de ganho
    gain_l = np.cos(norm_pan * np.pi / 2)
    gain_r = np.sin(norm_pan * np.pi / 2)
    
    return gain_l, gain_r

def calculate_5_1_gains(azimuth):
    """
    Calcula ganhos 5.1 usando Interpolação Linear de Potência Constante.
    Canais: [0:L, 1:R, 2:C, 3:LFE, 4:Ls, 5:Rs]
    """
    azimuth = azimuth % 360
    gains = np.zeros(6)

    # Definição dos ângulos padrão ITU-R para 5.1
    # Centro=0, Direita=30, Surr.Dir=110, Surr.Esq=250, Esquerda=330
    # Tuplas: (Ângulo, Índice do Canal)
    speaker_nodes = [
        (0, 2),      # Center
        (30, 1),     # Right
        (110, 5),    # Right Surround
        (250, 4),    # Left Surround
        (330, 0),    # Left
        (360, 2)     # Center (fechamento do círculo)
    ]

    # Encontra entre quais caixas o azimute está
    for i in range(len(speaker_nodes) - 1):
        angle_start, ch_start = speaker_nodes[i]
        angle_end, ch_end = speaker_nodes[i+1]

        if angle_start <= azimuth <= angle_end:
            # Calcula a proporção (0 a 1) entre as duas caixas
            segment_range = angle_end - angle_start
            if segment_range == 0: segment_range = 1 # Evita div por zero
            
            t = (azimuth - angle_start) / segment_range
            
            # Pan de Potência Constante (Seno/Cosseno)
            gains[ch_start] = np.cos(t * np.pi / 2)
            gains[ch_end] = np.sin(t * np.pi / 2)
            break

    return gains