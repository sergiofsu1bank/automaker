# =====================================================================
#                           IMPORTS DO PROJETO
# =====================================================================
from data_explore.data.data_explore import DataExplore
from data_explore.business.normalization_data import NormalizationData
from data_explore.util.save_files import SaveFiles

import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional, Any

# ----------------- Machine Learning -----------------
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import r2_score, f1_score
import os
import joblib


class DataBusiness:

    def __init__(self):
        self.normal = NormalizationData()
        self.save = SaveFiles()

    def _detect_target(self, df):
        """
        Detecta automaticamente qual coluna é o target no DataFrame.
        Estratégia:
        - Verifica nomes comuns de targets
        - Verifica colunas binárias (0/1)
        """
        # nomes comuns de coluna-alvo
        possible_targets = ["default", "target", "label", "classe", "class"]

        # 1) Verifica nomes comuns
        for col in df.columns:
            if col.lower() in possible_targets:
                return col

        # 2) Se não achou, procura colunas binárias
        for col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset({0, 1}) and len(unique_vals) > 1:
                return col

        # 3) Não conseguiu detectar
        return None

    def normalization_data(self, df, target=None):
        """
        Pipeline profissional:
        - Usuário pode informar o target manualmente
        - Caso não informe, o target é detectado automaticamente
        - Se não for possível detectar, erro amigável é lançado
        """

        print("\n" + "=" * 70)
        print("⭐ INICIANDO PIPELINE DE NORMALIZAÇÃO PROFISSIONAL")
        print("=" * 70 + "\n")

        # -------------------------------------------------------
        # 1) Determinar o target
        # -------------------------------------------------------
        if target is None:
            target = self._detect_target(df)

        if target is None:
            raise ValueError(
                "❌ Não foi possível identificar automaticamente o target.\n"
                "Por favor informe manualmente, exemplo:\n"
                "  app.business.normalization_data(df, target='default')"
            )

        print(f"✅ Target detectado: {target}\n")

        # -------------------------------------------------------
        # 2) Limpeza de inconsistências
        # -------------------------------------------------------
        df_clean, report_inconsist = self.normal.data_inconsistencies(
            df, target)

        print("\n✅ LIMPEZA DE INCONSISTÊNCIAS FINALIZADA")
        print("-" * 60)

        # -------------------------------------------------------
        # 3) Detecção e tratamento de outliers
        # -------------------------------------------------------
        df_out, report_outliers = self.normal.detect_and_fix_outliers(
            df_clean, target)

        print("\n✅ TRATAMENTO DE OUTLIERS FINALIZADO")
        print("-" * 60)

        # -------------------------------------------------------
        # 4) Split treino/teste
        # -------------------------------------------------------
        X_train, X_test, y_train, y_test = self.normal.split_train_test(
            df_out, target)

        print("\n✅ SPLIT DE TREINO E TESTE REALIZADO")
        print("-" * 60)

        self.save.save_splits(X_train, X_test, y_train, y_test)

        # -------------------------------------------------------
        # 5) NORMALIZAÇÃO AUTOMÁTICA
        # -------------------------------------------------------
        X_train_norm, X_test_norm, scalers_used = self.normal.auto_normalize(
            X_train, X_test)

        self.save.save_normalized_data(X_train_norm, X_test_norm, scalers_used)

        # -------------------------------------------------------
        # 5) Relatório final
        # -------------------------------------------------------
        print("\n📊 RELATÓRIO FINAL DE LIMPEZA")
        print("-" * 60)

        print("\n🔹 INCONSISTÊNCIAS:")
        for r in report_inconsist:
            print(r)

        print("\n🔹 OUTLIERS:")
        for r in report_outliers:
            print(r)

        print("\n✅ Pipeline profissional concluído!\n")

        print("\n📊 Relatório de Normalização:")
        for col, (_, method) in scalers_used.items():
            print(f" - {col}: {method}")

        print("\n✅ NORMALIZAÇÃO AUTOMÁTICA CONCLUÍDA")
        print("-" * 60)

        # Retorna tudo normalizado
        return X_train_norm, X_test_norm, y_train, y_test
