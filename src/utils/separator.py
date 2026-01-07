import os, subprocess, shutil
class AudioSeparator:
    @staticmethod
    def separate(audio_path, output_root):
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        demucs_out = os.path.join(output_root, "htdemucs_6s", filename)
        
        print(f"--- Rodando Demucs 6 Stems para: {filename} ---")
        subprocess.run(["demucs", "-n", "htdemucs_6s", audio_path, "-o", output_root], check=True)
        
        project_dir = os.path.join("projects", filename)
        os.makedirs(project_dir, exist_ok=True)

        stems_configs = []
        for file in os.listdir(demucs_out):
            old_path = os.path.join(demucs_out, file)
            new_path = os.path.join(project_dir, file)
            shutil.move(old_path, new_path)
            stems_configs.append({'name': os.path.splitext(file)[0], 'file': new_path})
        
        return project_dir, stems_configs