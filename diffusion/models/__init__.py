try:
    from .option_i import OptionIModel
    from .option_ii import OptionIIModel
    from .option_iii import OptionIIIModel

    __all__ = ["OptionIModel", "OptionIIModel", "OptionIIIModel"]
except ModuleNotFoundError:
    # Allows `unittest` discovery to run in environments without PyTorch installed.
    __all__ = []

