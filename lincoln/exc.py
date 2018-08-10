""" Module to hold exceptions, errors, etc. """

class MatchError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

class DimensionError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

class BackwardError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message