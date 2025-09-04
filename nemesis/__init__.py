import os
import logging
from typing import Any, Callable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

old_factory = logging.getLogRecordFactory()


def logger_record_factory(run: str) -> Callable[..., logging.LogRecord]:
    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)
        record.run = run  # type: ignore
        return record
    return record_factory
