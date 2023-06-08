from dataclasses import dataclass
from pathlib import Path


@dataclass
class AaRootMetadata:
    encoding: str
    input_meta_fn: str
    output_meta_fn: str
    log_filename: str
    log_encoding: str
    log_format: str

    @property
    def get_base_dir(self) -> Path:
        return self._base_dir

    def set_base_dir(self, meta_dir: Path):
        self._base_dir = meta_dir

    @property
    def get_meta_dir(self) -> Path:
        return self._meta_dir

    def set_meta_dir(self, meta_dir: Path):
        self._meta_dir = meta_dir
