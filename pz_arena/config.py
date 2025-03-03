from typing import Final

class Configuration:
	MODEL_DIRECTORY: Final[str] = "model"
	MODEL_TEMP_DIRECTORY: Final[str] = "model-temp"
	TENSORBOARD_LOG: Final[str] = "tensorboard"
	MODEL_UPDATE_FREQUENCY: Final[int] = 15
	TOTAL_TIME_STEPS: Final[int] = 1_000_000
	MODEL_FILE_LIMIT: Final[int] = 3