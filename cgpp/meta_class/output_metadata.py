from dataclasses import dataclass
from pathlib import Path, PurePath


@dataclass
class OutputMetadata:
    encoding: str
    directory: str
    crop_schedule_fn_prefix: str
    crop_schedule_fn_ext: str

    keystone_val: str
    first_col_wth: int
    sch_col_wth: int

    schedule_sheetname: str
    shelf_config_sheetname: str

    def set_base_dir(self, meta_dir: PurePath):
        self._base_dir = meta_dir

    def crop_schedule_f_loc(self, problem_name: str, obj_idx: int) -> PurePath:
        fn: str = (
            self.crop_schedule_fn_prefix
            + problem_name
            + f"_obj{obj_idx}"
            + self.crop_schedule_fn_ext
        )

        return_path = PurePath(self._base_dir, self.directory, fn)
        Path(return_path.parent).mkdir(parents=True, exist_ok=True)
        return return_path
