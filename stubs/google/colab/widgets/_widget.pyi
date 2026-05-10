from google.colab import errors as errors, output as output

WidgetException = errors.WidgetException

class OutputAreaWidget:
    def __init__(self) -> None: ...
    def remove(self, wait: bool = False) -> None: ...
