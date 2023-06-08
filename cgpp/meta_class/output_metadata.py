from dataclasses import dataclass
from pathlib import PurePath


@dataclass
class OutputMetadata:
    encoding: str
    directory: str

    def set_base_dir(self, meta_dir: PurePath):
        self._base_dir = meta_dir
