# Global flag to suppress printing inside decorated functions
_SUPPRESS_PRINTING = False

def suppress_printing():
    global _SUPPRESS_PRINTING
    _SUPPRESS_PRINTING = True

def enable_printing():
    global _SUPPRESS_PRINTING
    _SUPPRESS_PRINTING = False

def is_printing_suppressed():
    return _SUPPRESS_PRINTING
