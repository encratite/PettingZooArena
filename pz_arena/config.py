from typing import Final

class Configuration:
	MODEL_PATH: Final[str] = "model"
	TENSORBOARD_LOG: Final[str] = "tensorboard"
	MODEL_UPDATE_FREQUENCY: Final[int] = 15
	TOTAL_TIME_STEPS: Final[int] = 1_000_000