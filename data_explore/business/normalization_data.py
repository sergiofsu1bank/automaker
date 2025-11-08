# -*- coding: utf-8 -*-
"""
Classe NormalizationData (versão profissional)
Contém pipeline completo para detecção e correção de inconsistências
em DataFrames usando regras inteligentes, imputação avançada e relatório detalhado.
"""

import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional, Any

from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class NormalizationData:
    """
    Pipeline profissional para tratamento de inconsistências.
    Substitui completamente a lógica antiga.
    """

    def __init__(
        self,
        zero_is_inconsistent: bool = True,
        binary_valid_set: set = {0, 1},
        random_state: int = 42
    ):
        """
        Inicializa parâmetros da classe.
        """
        self.zero_is_inconsistent = zero_is_inconsistent
        self.binary_valid_set = set(binary_valid_set)
        self.random_state = random_state

    # =====================================================================
    #                           FUNÇÕES AUXILIARES
    # =====================================================================

    @staticmethod
    def _is_numeric(s: pd.Series) -> bool:
        """Retorna True se a coluna for numérica."""
        return pd.api.types.is_numeric_dtype(s)

    @staticmethod
    def _unique_count(s: pd.Series) -> int:
        """Conta valores únicos ignorando NaN."""
        return s.dropna().nunique()

    def _is_binary(self, s: pd.Series) -> bool:
        """Detecta se coluna é binária baseada no conjunto válido."""
        vals = set(s.dropna().unique())
        return len(vals) > 0 and vals.issubset(self.binary_valid_set)

    def _is_categorical(self, s: pd.Series) -> bool:
        """Detecta se a coluna é categórica."""
        return not self._is_numeric(s)

    def _percent(self, n: int, total: int) -> float:
        """Calcula percentual."""
        return (n / total) * 100 if total > 0 else 0.0

    # =====================================================================
    #                       DETECÇÃO DE INCONSISTÊNCIAS
    # =====================================================================

    def _inconsistency_mask(self, s: pd.Series, is_binary: bool) -> pd.Series:
        """
        Retorna uma máscara booleana com True para linhas inconsistentes.
        """
        mask = s.isna()

        if is_binary:
            return mask | ~s.isin(self.binary_valid_set)

        if self._is_numeric(s):
            if self.zero_is_inconsistent:
                mask |= (s == 0)
            return mask

        s_str = s.astype(str).str.strip()
        mask |= s_str.isin(["", "?", "nan"])
        return mask

    # =====================================================================
    #                       IMPORTÂNCIA DA COLUNA
    # =====================================================================

    def _importance(self, df: pd.DataFrame, col: str, target: Optional[str]) -> float:
        """
        Calcula a importância da coluna em relação ao target:
        - Correlação para colunas numéricas
        - Mutual Information para demais
        """
        if not target or target not in df.columns:
            return 0.0

        s = df[col]
        t = df[target]

        try:
            if self._is_numeric(s) and self._is_numeric(t):
                tmp = df[[col, target]].dropna()
                if tmp.shape[0] < 10:
                    return 0.0
                corr = tmp[col].corr(tmp[target])
                return abs(corr) if corr is not None else 0.0

            X = df[[col]].copy()
            y = df[target].copy()

            if not self._is_numeric(X[col]):
                X[col] = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1) \
                    .fit_transform(X[[col]])

            mask = ~X[col].isna() & ~y.isna()
            Xn = X.loc[mask, [col]].values
            yn = y.loc[mask]

            if self._is_numeric(t):
                mi = mutual_info_regression(Xn, yn)
            else:
                mi = mutual_info_classif(Xn, yn.astype(str))

            return float(mi[0])

        except Exception:
            return 0.0

    # =====================================================================
    #                          IMPUTAÇÃO SIMPLES
    # =====================================================================

    def _simple_impute(self, s: pd.Series, is_numeric: bool, is_binary: bool) -> pd.Series:
        """
        Imputação simples:
        - Numérica → mediana
        - Binária → valor mais frequente
        - Categórica → moda
        """
        ser = s.copy()

        if is_binary:
            mode = ser.mode(dropna=True)
            return ser.fillna(mode.iloc[0])

        if is_numeric:
            return ser.fillna(ser.median())

        mode = ser.mode(dropna=True)
        return ser.fillna(mode.iloc[0] if not mode.empty else None)

    # =====================================================================
    #                  IMPUTAÇÃO COM KNN (APENAS NUMÉRICAS)
    # =====================================================================

    def _knn_impute(self, df: pd.DataFrame, col: str, mask: pd.Series) -> pd.Series:
        work = df.copy()
        work.loc[mask, col] = np.nan

        numeric_cols = [c for c in work.columns if self._is_numeric(work[c])]
        if col not in numeric_cols:
            return self._simple_impute(work[col], True, False)

        imputer = KNNImputer(n_neighbors=5)
        imputed = imputer.fit_transform(work[numeric_cols])

        result_df = pd.DataFrame(imputed, columns=numeric_cols, index=df.index)
        return result_df[col]

    # =====================================================================
    #                IMPUTAÇÃO COM MODELO (RandomForest)
    # =====================================================================

    def _model_impute(self, df: pd.DataFrame, col: str, mask: pd.Series) -> Tuple[pd.Series, Dict]:
        X = df.drop(columns=[col]).copy()
        y = df[col].copy()

        valid_mask = ~mask & ~y.isna()

        if valid_mask.sum() < 30:
            filled = self._simple_impute(
                y.mask(mask), self._is_numeric(y), self._is_binary(y))
            return filled, {"model": "simple_impute_fallback"}

        cat_cols = [c for c in X.columns if self._is_categorical(X[c])]
        encoder = None

        if cat_cols:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1)
            X[cat_cols] = encoder.fit_transform(X[cat_cols])

        X_train = X.loc[valid_mask]
        y_train = y.loc[valid_mask]

        if self._is_numeric(y):
            model = RandomForestRegressor(
                n_estimators=200, random_state=self.random_state)
        else:
            y_train = y_train.astype(str)
            model = RandomForestClassifier(
                n_estimators=300, random_state=self.random_state)

        model.fit(X_train, y_train)

        X_pred = X.loc[mask]
        preds = model.predict(X_pred)

        y_out = y.copy()
        y_out.loc[mask] = preds

        metrics = {"model": type(model).__name__}
        try:
            pred_train = model.predict(X_train)
            if self._is_numeric(y_train):
                metrics["score"] = float(r2_score(y_train, pred_train))
            else:
                metrics["score"] = float(
                    f1_score(y_train, pred_train, average="macro"))
        except Exception:
            metrics["score"] = None

        return y_out, metrics

    # =====================================================================
    #                          DECISÃO PROFISSIONAL
    # =====================================================================

    def _decide(self, percent, importance, is_binary, is_numeric):
        """
        Retorna ação:
        - simple_impute
        - knn_impute
        - model_impute
        - drop_rows
        - drop_column
        """
        if is_binary:
            return "simple_impute"

        if importance >= 0.20:
            if percent < 5:
                return "simple_impute"
            if percent < 15:
                return "knn_impute" if is_numeric else "simple_impute"
            return "model_impute"

        if 0.05 <= importance < 0.20:
            if percent < 5:
                return "simple_impute"
            if percent < 20:
                return "drop_rows"
            return "drop_column"

        if importance < 0.05:
            if percent < 10:
                return "simple_impute"
            return "drop_column"

        return "simple_impute"

    # =====================================================================
    #                MÉTODO PRINCIPAL — NOVA VERSÃO PROFISSIONAL
    # =====================================================================
    def data_inconsistencies(self, df: pd.DataFrame, target: Optional[str] = None) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Método principal.
        Varre coluna por coluna, detecta inconsistências,
        decide ação e gera relatório completo.

        Retorna:
            df_corrigido, relatorio
        """

        df = df.copy()
        report = []

        for col in list(df.columns):

            if target and col == target:
                continue

            s = df[col]

            is_binary = self._is_binary(s)
            is_numeric = self._is_numeric(s)
            is_cat = not is_numeric

            mask = self._inconsistency_mask(s, is_binary)
            n_bad = int(mask.sum())
            total = df.shape[0]
            percent = self._percent(n_bad, total)

            importance = self._importance(df, col, target)

            action = self._decide(percent, importance, is_binary, is_numeric)

            log = {
                "coluna": col,
                "tipo": "binária" if is_binary else ("numérica" if is_numeric else "categórica"),
                "percentual_inconsistente": round(percent, 4),
                "importancia": round(float(importance), 6),
                "registros_afetados": int(n_bad),
                "acao": action,
                "metricas": None,
                "detalhes": None
            }

            try:
                if action == "simple_impute":
                    ser = s.copy()
                    ser.loc[mask] = np.nan
                    df[col] = self._simple_impute(ser, is_numeric, is_binary)
                    log["detalhes"] = "Imputação simples aplicada."

                elif action == "knn_impute":
                    df[col] = self._knn_impute(df, col, mask)
                    log["detalhes"] = "Imputação via KNN (k=5)."

                elif action == "model_impute":
                    filled, metrics = self._model_impute(df, col, mask)
                    df[col] = filled
                    log["metricas"] = metrics
                    log["detalhes"] = "Imputação via RandomForest."

                elif action == "drop_rows":
                    df = df.loc[~mask].reset_index(drop=True)
                    log["detalhes"] = "Linhas removidas."

                elif action == "drop_column":
                    df = df.drop(columns=[col])
                    log["detalhes"] = "Coluna removida."

            except Exception as e:
                log["detalhes"] = f"Erro: {e}. Fallback: imputação simples."
                ser = s.copy()
                ser.loc[mask] = np.nan
                df[col] = self._simple_impute(ser, is_numeric, is_binary)

            report.append(log)

        return df, report

    # =====================================================================
    #                DETECÇÃO E TRATAMENTO PROFISSIONAL DE OUTLIERS
    # =====================================================================

    def detect_and_fix_outliers(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None
    ):
        """
        Detecta e trata outliers usando o método IQR (Interquartile Range).
        Estratégia:
            - Apenas colunas numéricas são analisadas
            - Colunas com até 5% de outliers → winsorization (cap nos limites)
            - Colunas com mais de 5% de outliers → remoção de linhas
            - A coluna alvo (target) NUNCA é alterada

        Retorna:
            df_limpo, relatorio (list de dicts)
        """

        df = df.copy()
        relatorio = []

        for col in df.columns:

            # Nunca mexer no target
            if target and col == target:
                continue

            serie = df[col]

            # Outliers só existem em colunas numéricas
            if not pd.api.types.is_numeric_dtype(serie):
                continue

            # Calcula IQR
            Q1 = serie.quantile(0.25)
            Q3 = serie.quantile(0.75)
            IQR = Q3 - Q1

            lim_inf = Q1 - 1.5 * IQR
            lim_sup = Q3 + 1.5 * IQR

            # Máscara de outliers
            mask_out = (serie < lim_inf) | (serie > lim_sup)
            n_outliers = int(mask_out.sum())

            # Não há outliers → próxima coluna
            if n_outliers == 0:
                continue

            percent = (n_outliers / df.shape[0]) * 100

            log = {
                "coluna": col,
                "outliers_detectados": n_outliers,
                "percentual": round(percent, 3),
                "acao": None
            }

            # ------------------------------------------------------------------
            # REGRA PROFISSIONAL
            # ------------------------------------------------------------------

            if percent <= 5:
                # Winsorization — limitar valores para o limite inferior/superior
                df.loc[serie < lim_inf, col] = lim_inf
                df.loc[serie > lim_sup, col] = lim_sup

                log["acao"] = (
                    "winsorization (substituição dos outliers pelos limites inferiores/superiores)"
                )

            else:
                # Muitos outliers → remover linhas
                df = df.loc[~mask_out].reset_index(drop=True)
                log["acao"] = "remoção de linhas contendo outliers severos"

            relatorio.append(log)

        return df, relatorio

    # =====================================================================
    #                   SPLIT PROFISSIONAL TREINO / TESTE
    # =====================================================================
    from sklearn.model_selection import train_test_split

    def split_train_test(
        self,
        df: pd.DataFrame,
        target: str,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Realiza a separação entre treino e teste de forma profissional.

        Funcionalidades:
            - Separa X e y automaticamente
            - Mantém apenas features (X = df[target removido])
            - Aplica estratificação SOMENTE quando o target for categórico
            - Suporta classificação binária, multiclass e regressão
            - Retorna X_train, X_test, y_train, y_test

        Parâmetros:
            df         : DataFrame já limpo e pré-processado
            target     : nome da coluna alvo
            test_size  : percentual de teste (default 0.20)
            random_state : garante reprodutibilidade

        Retorna:
            X_train, X_test, y_train, y_test
        """

        # ------------------------------------------
        # 1) Validar target
        # ------------------------------------------
        if target not in df.columns:
            raise ValueError(
                f"Target '{target}' não encontrado no DataFrame. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

        # ------------------------------------------
        # 2) Separar features e alvo
        # ------------------------------------------
        X = df.drop(columns=[target])
        y = df[target]

        # ------------------------------------------
        # 3) Determinar se é classificação ou regressão
        # ------------------------------------------
        # Regressão = target numérico contínuo
        # Classificação = target categórico ou discreto
        is_regression = pd.api.types.is_numeric_dtype(y) and y.nunique() > 20

        # ------------------------------------------
        # 4) Usar estratificação apenas para CLASSIFICAÇÃO
        # ------------------------------------------
        stratify = None
        if not is_regression:
            # Apenas se tiver mais de 1 classe
            if y.nunique() > 1:
                stratify = y

        # ------------------------------------------
        # 5) Realiza o split
        # ------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        # ------------------------------------------
        # 6) Retorna tudo
        # ------------------------------------------
        return X_train, X_test, y_train, y_test

    # =====================================================================
    #                  AUTO NORMALIZATION (INTELIGENTE)
    # =====================================================================

    def auto_normalize(self, X_train, X_test):
        """
        Normalização automática inteligente.

        Estratégia:
            - Se coluna é binária → não normaliza
            - Se coluna é categórica → não normaliza
            - Se coluna tem outliers → RobustScaler
            - Se coluna é contínua normal → StandardScaler
            - Se coluna é contínua e muito assimétrica → MinMaxScaler

        Retorna:
            X_train_norm, X_test_norm, scalers_dict
        """

        X_train = X_train.copy()
        X_test = X_test.copy()

        scalers = {}

        for col in X_train.columns:

            s = X_train[col]

            # ---------------------------------------
            # 1) Ignorar colunas não numéricas
            # ---------------------------------------
            if not pd.api.types.is_numeric_dtype(s):
                continue

            # ---------------------------------------
            # 2) Ignorar colunas binárias
            # ---------------------------------------
            unique_vals = set(s.dropna().unique())
            if unique_vals.issubset({0, 1}):
                continue

            # ---------------------------------------
            # 3) Detectar outliers (IQR)
            # ---------------------------------------
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            lim_inf = Q1 - 1.5 * IQR
            lim_sup = Q3 + 1.5 * IQR

            has_outliers = ((s < lim_inf) | (s > lim_sup)).sum() > 0

            # ---------------------------------------
            # 4) Detectar assimetria
            # ---------------------------------------
            skew = abs(s.skew())

            # ---------------------------------------
            # 5) Escolher scaler automaticamente
            # ---------------------------------------
            if has_outliers:
                scaler = RobustScaler()
                method = "RobustScaler"
            elif skew > 1:
                scaler = MinMaxScaler()
                method = "MinMaxScaler"
            else:
                scaler = StandardScaler()
                method = "StandardScaler"

            scalers[col] = (scaler, method)

            # ---------------------------------------
            # 6) Fit NO X_train (sem data leakage)
            # ---------------------------------------
            X_train[col] = scaler.fit_transform(X_train[[col]])

            # ---------------------------------------
            # 7) Transform no X_test (mesmo scaler)
            # ---------------------------------------
            X_test[col] = scaler.transform(X_test[[col]])

        return X_train, X_test, scalers
