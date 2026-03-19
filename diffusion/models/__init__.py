try:
    from .option_i import OptionIModel
    from .option_ii import OptionIIModel

    __all__ = ["OptionIModel", "OptionIIModel"]
except ModuleNotFoundError:
    # Allows `unittest` discovery to run in environments without PyTorch installed.
    __all__ = []

