import os
import pandas as pd


class DataLoader:
    def __init__(self):
        # Caminho base do projeto (nível acima da pasta autosage/)
        self.base_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))

    def resolve_path(self, file_path):
        # Se já for caminho absoluto, retorna direto
        if os.path.isabs(file_path):
            return file_path

        # Caso contrário, assume que está dentro da pasta /data
        return os.path.join(self.base_dir, "data", file_path)

    def load(self, file_path):
        # Resolve caminho correto
        file_path = self.resolve_path(file_path)

        # Confere se arquivo existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

        # Tenta carregar CSV, depois Excel
        try:
            df = pd.read_csv(file_path)
        except Exception:
            try:
                df = pd.read_excel(file_path)
            except Exception:
                raise ValueError(
                    "Formato de arquivo não suportado. Use .csv ou .xlsx")

        return df

   