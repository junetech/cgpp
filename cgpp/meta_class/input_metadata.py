from dataclasses import dataclass
from pathlib import PurePath, Path
from typing import Iterable


@dataclass
class InputMetadata:
    encoding: str
    directory: str
    input_ext: str

    fn_splitter: str
    model_type_t1: str
    model_type_t2: str
    model_type_t3: str

    solver_name: str
    timelimit: int

    def set_base_dir(self, meta_dir: PurePath):
        self._base_dir = meta_dir

    def input_dir(self) -> Path:
        return Path(self._base_dir, self.directory)

    def input_data_f_loc_iter(self) -> Iterable[Path]:
        # input_path.mkdir(parents=True, exist_ok=True)
        for fp in self.input_dir().iterdir():
            if fp.suffix == self.input_ext:
                yield fp
