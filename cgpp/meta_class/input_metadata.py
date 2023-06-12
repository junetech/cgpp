from dataclasses import dataclass
from pathlib import PurePath, Path
from typing import Iterable


@dataclass
class InputMetadata:
    encoding: str
    directory: str
    input_ext: str
    input_model_type_list: list[str]

    fn_splitter: str
    model_type_t1: str
    model_type_t2: str
    model_type_t3: str
    model_type_obj_idx_list_dict: dict[str, list[int]]

    solver_name: str
    timelimit: int

    def set_base_dir(self, meta_dir: PurePath):
        self._base_dir = meta_dir

    def get_abs_dir(self, rel_dir: PurePath) -> Path:
        return Path(self._base_dir, rel_dir)

    def input_dir(self) -> Path:
        return Path(self._base_dir, self.directory)

    def input_data_f_loc_iter(self) -> Iterable[Path]:
        for fp in self.input_dir().iterdir():
            if fp.suffix == self.input_ext:
                yield fp

    def obj_idx_list(self) -> list[int]:
        return sorted(
            obj_idx
            for obj_idx_list in self.model_type_obj_idx_list_dict.values()
            for obj_idx in obj_idx_list
        )
