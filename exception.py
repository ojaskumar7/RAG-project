# exception.py
import traceback

class CustomException(Exception):
    def __init__(self, original_exception, sys_module=None):
        self.original_exception = original_exception
        self.trace = None
        if sys_module is not None:
            try:
                self.trace = traceback.format_exc()
            except Exception:
                self.trace = "Trace unavailable"
        message = f"{original_exception} - Trace: {self.trace}"
        super().__init__(message)

    def __str__(self):
        return f"CustomException: {self.original_exception}\n{self.trace}"
