import os
import pandas as pd
import joblib


class SaveFiles:

    def save_splits(self, X_train, X_test, y_train, y_test,
                    output_dir=r"D:\u1\autosage\dev\preprocessing\data_explore\arquive\output"):
        """
            Salva os arquivos X_train, X_test, y_train, y_test em CSV no diretório especificado.
            """

        # cria pasta se não existir
        os.makedirs(output_dir, exist_ok=True)

        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        print(f"Arquivos salvos em: {output_dir}")

    # --- coloque dentro da sua classe, fora do normalization_data ---

    def save_normalized_data(
        self,
        X_train_norm,
        X_test_norm,
        scalers_used,
        output_dir=r"D:\u1\autosage\dev\preprocessing\data_explore\arquive\output"
    ):
        """
        Salva os arquivos normalizados X_train_norm, X_test_norm, 
        e todos os scalers usados no processo.

        - X_train_norm.csv
        - X_test_norm.csv
        - scalers.pkl (todos os scalers juntos)
        """

        # Criar pasta se não existir
        os.makedirs(output_dir, exist_ok=True)

        # ---------------------------------------------------------
        # 1. Salvar X_train_norm
        # ---------------------------------------------------------
        X_train_file = os.path.join(output_dir, "X_train_norm.csv")
        X_train_norm.to_csv(X_train_file, index=False)

        # ---------------------------------------------------------
        # 2. Salvar X_test_norm
        # ---------------------------------------------------------
        X_test_file = os.path.join(output_dir, "X_test_norm.csv")
        X_test_norm.to_csv(X_test_file, index=False)

        # ---------------------------------------------------------
        # 3. Salvar scalers (arquivo único .pkl)
        # ---------------------------------------------------------
        scalers_file = os.path.join(output_dir, "scalers_used.pkl")
        joblib.dump(scalers_used, scalers_file)

        print("\n✅ Arquivos salvos com sucesso em:")
        print(output_dir)
        print(" - X_train_norm.csv")
        print(" - X_test_norm.csv")
        print(" - scalers_used.pkl")
