import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path, PurePath
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

    n_crops_seq: list[int]
    n_shelves_seq: list[int]
    demand_mult_seq: list[float]
    n_shelves_dict: dict[int, str]
    time_horizon_seq: list[int]

    solver_name: str
    timelimit: int

    def set_base_dir(self, meta_dir: PurePath):
        self._base_dir = meta_dir

    def get_abs_dir(self, rel_dir: PurePath) -> Path:
        return Path(self._base_dir, rel_dir)

    def input_dir(self) -> Path:
        return Path(self._base_dir, self.directory)

    def input_data_f_loc_iter(self) -> Iterable[Path]:
        for n_shelves, n_crops, time_horizon, demand_mult in product(
            self.n_shelves_seq,
            self.n_crops_seq,
            self.time_horizon_seq,
            self.demand_mult_seq,
        ):
            regex_str = f"{n_shelves}-{n_crops}-*"
            regex_str += "." * n_crops
            regex_str += f"-{time_horizon}-{demand_mult}-*"
            fp_list = [
                fp for fp in self.input_dir().iterdir() if re.match(regex_str, fp.name)
            ]
            for fp in fp_list:
                yield fp

    def obj_idx_list(self) -> list[int]:
        return sorted(
            obj_idx
            for obj_idx_list in self.model_type_obj_idx_list_dict.values()
            for obj_idx in obj_idx_list
        )
