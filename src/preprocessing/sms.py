# 서드파티
import pandas as pd

# 프로젝트
from src.preprocessing.common import PreprocessingManager


class SMSDataPreprocessingManager(PreprocessingManager):
    def __init__(
        self,
        feature_column_name: str = 'message',
        label_column_name: str = 'label'
    ) -> None:
        super().__init__(feature_column_name, label_column_name)

    def read_sample_data(
        self,
        path: str
    ):
        df_sms = pd.read_csv(path, encoding='latin-1')
        df_sms.dropna(how="any", inplace=True, axis=1)
        df_sms.columns = [self.label_column_name, self.feature_column_name]
        return df_sms

    def split(
        self,
        df: pd.DataFrame,
        ratio: int = 0.8,
    ) -> list:
        cnt = int(len(df)*ratio)
        train_df = df[:cnt]
        test_df = df[cnt:]
        return train_df, test_df
