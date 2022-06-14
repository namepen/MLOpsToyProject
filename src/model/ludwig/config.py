# 내장
import os
from datetime import datetime

# 프로젝트
from src.preprocessing.common import PreprocessingManager


class LudwigConfigManager():
    def __init__(
        self,
        prep_manager: PreprocessingManager,
        experiment_name: str = "ludwig",
        model_name: str = "automl",
        model_id: str = None,
        time_based_id: bool = True,
    ) -> None:
        if time_based_id:
            assert model_id is None
            model_id = datetime.now().strftime("%Y%m%d%H%M%S")

        self.prep_manager = prep_manager
        self.experiment_name = f'{experiment_name}'
        self.model_name = f'{model_name}_{model_id}'

        self.saved_model_base_dir = f'./models'
        self.saved_model_file_name = f'{self.experiment_name}_{self.model_name}'

        self.cached_preprocessed_data_base_dir = os.path.join(
            './data', 'preprocessed')
        self.cached_preprocessed_data_dir = os.path.join(
            self.cached_preprocessed_data_base_dir,
            self.saved_model_file_name,
        )
        os.makedirs(self.cached_preprocessed_data_dir, exist_ok=True)

        self.model_info_output_base_dir = f"./results"
        self.model_info_output_dir = os.path.join(
            self.model_info_output_base_dir,
            self.saved_model_file_name,
        )

    def get_ludwig_config(
        self
    ) -> dict:
        config = {
            "input_features": [
                {
                    "name": self.prep_manager.feature_column_name,
                    "type": "text",             # Data type of the input column
                    "encoder": "parallel_cnn",  # The model architecture we should use for
                                                # encoding this column
                }
            ],
            "output_features": [
                {
                    "name": self.prep_manager.label_column_name,
                    "type": "category",
                }
            ],
            "trainer": {
                "epochs": 5,
            },
            "backend": {
                "type": "local",
                "cache_dir": self.cached_preprocessed_data_dir
            },
        }
        return config
