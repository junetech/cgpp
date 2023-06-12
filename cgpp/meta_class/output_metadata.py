import csv
import logging
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
    germination_sheetname: str
    harvest_sheetname: str

    summary_fn_prefix: str
    summary_colhead_list: list[str]

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

    def summary_file_location(self) -> Path:
        fn: str = self.summary_fn_prefix + self.summary_fn_suffix + ".csv"
        return_path = Path(self._base_dir, self.directory, fn)
        return_path.parent.mkdir(parents=True, exist_ok=True)
        return return_path

    def init_summary_file(self, summary_fn_suffix: str):
        self.summary_fn_suffix: str = summary_fn_suffix
        sf_path = self.summary_file_location()
        if sf_path.exists():
            logging.info(f"Appending to the existing summary file {sf_path}")
        else:
            with open(sf_path, "w", newline="", encoding=self.encoding) as f:
                csv_writer = csv.DictWriter(f, fieldnames=self.summary_colhead_list)
                csv_writer.writeheader()
            logging.info(f"Created new summary file {sf_path}")

    def append_summary(self, summary_dict: dict[str, float]):
        sf_path = self.summary_file_location()
        with open(sf_path, "a", newline="", encoding=self.encoding) as _file:
            csv_writer = csv.DictWriter(_file, fieldnames=self.summary_colhead_list)
            csv_writer.writerow(summary_dict)
