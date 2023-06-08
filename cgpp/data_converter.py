import ast
import datetime
import json
from pathlib import Path, PurePath
from typing import Iterable

from input_class import (
    ProbInsS21,
    ProbInsS21T1,
    ProbInsS21T2,
    ProbInsS21T3,
    config_id_str,
    shelf_id_str,
    shelf_type_str,
)


def from_file_to_dict(
    fp: PurePath, item_splitter: str, line_splitter: str, key_val_splitter: str
) -> dict[str, str]:
    with open(fp, "r") as f:
        contents = f.read().replace(line_splitter, "")
    return {
        item.split(key_val_splitter)[0]: item.split(key_val_splitter)[1]
        for item in contents.split(item_splitter)
        if item != ""
    }


def convert_dat_to_json(
    input_dir: PurePath,
    input_ext: str,
    output_dir: PurePath,
    output_ext: str,
    fn_splitter: str,
    output_encoding: str,
):
    for p_ins in generate_p_ins_from_dat(input_dir, input_ext, fn_splitter):
        fn = p_ins.problem_name + output_ext
        dump_dict = p_ins.make_json_dump_dict()
        with open(output_dir.joinpath(fn), "w", encoding=output_encoding) as f:
            json.dump(
                dump_dict,
                f,
                ensure_ascii=False,
                indent=None,
                separators=(",", ":"),
                sort_keys=False,
            )


def generate_p_ins_from_dat(
    input_dir: PurePath,
    input_ext: str,
    fn_splitter: str,
) -> Iterable[ProbInsS21]:
    input_path = Path(input_dir)
    fp_list: list[Path] = [fp for fp in input_path.iterdir() if fp.suffix == input_ext]
    print(f"{len(fp_list)} files found in {input_path}")
    for fp in fp_list:
        filename = fp.stem
        # instance name information
        filename_info_list = filename.split(fn_splitter)
        [
            n_shelves_fn,
            n_crops_fn,
            crop_id_string,
            n_days_fn,
            demand_mult,
            id,
            model_type,
        ] = filename_info_list[:7]
        if len(filename_info_list) == 7:
            model_type = filename_info_list[6]
        else:
            model_type = f"{filename_info_list[6]}_{filename_info_list[7]}"

        input_dict = from_file_to_dict(fp, ";", "\n", " = ")

        n_days = int(input_dict["n_days"])
        n_crops = int(input_dict["n_crops"])
        crop_growth_days = ast.literal_eval(input_dict["crop_growth_days"])
        n_configurations = int(input_dict["n_configurations"])
        crop_growth_day_config = ast.literal_eval(input_dict["crop_growth_day_config"])
        capacity = ast.literal_eval(input_dict["capacity"])
        demand = ast.literal_eval(input_dict["demand"])
        if model_type == "s":
            n_shelves = int(input_dict["n_shelves"])
            crop_shelf_compatible = ast.literal_eval(
                input_dict["crop_shelf_compatible"]
            )
        else:
            n_shelf_types = int(input_dict["n_shelf_types"])
            num_shelves = ast.literal_eval(input_dict["num_shelves"])
            crop_shelf_type_compatible = ast.literal_eval(
                input_dict["crop_shelf_type_compatible"]
            )
            if model_type == "st_pen":
                missed_demand_penalty = ast.literal_eval(
                    input_dict["missed_demand_penalty"]
                )

        # Problem instance
        p_ins: ProbInsS21
        if model_type == "s":
            p_ins = ProbInsS21T1()
        elif model_type == "st":
            p_ins = ProbInsS21T2()
        else:
            p_ins = ProbInsS21T3()

        # instance name information
        p_ins.info_from_filename(
            int(n_shelves_fn),
            int(n_crops_fn),
            crop_id_string,
            int(n_days_fn),
            float(demand_mult),
            int(id),
            model_type,
        )

        # crop information
        if n_crops != p_ins.n_crops:
            print(
                UserWarning(
                    f"n_crops of filename {p_ins.n_crops} != n_crops of data {n_crops}"
                )
            )
            continue
        p_ins.create_crop_id()

        if n_days != p_ins.n_days:
            print(
                UserWarning(
                    f"n_days of filename {p_ins.n_days} != n_days of data {n_days}"
                )
            )
            continue
        p_ins.create_t_list()

        p_ins.crop_growth_days = dict()
        for c_index, crop_id in enumerate(p_ins.crop_id_list):
            p_ins.crop_growth_days[crop_id] = crop_growth_days[c_index]

        p_ins.n_configurations = n_configurations
        p_ins.create_config_id()

        p_ins.crop_growth_day_config = dict()
        p_ins.demand = dict()
        for c_index, crop_id in enumerate(p_ins.crop_id_list):
            p_ins.crop_growth_day_config[crop_id] = [
                config_id_str(crop_growth_day_config[c_index][growth_day])
                for growth_day in range(p_ins.crop_growth_days[crop_id])
                if crop_growth_day_config[c_index][growth_day] != -1
            ]
            p_ins.demand[crop_id] = {
                demand_day: demand[c_index][demand_day]
                for demand_day in p_ins.t_list
                if demand[c_index][demand_day] > 0
            }

        # shelf information
        p_ins.capacity = dict()
        if model_type == "s":
            if n_shelves != p_ins.n_shelves:
                print(
                    UserWarning(
                        f"n_shelves of filename {p_ins.n_shelves} != n_shelves of data {n_shelves}"
                    )
                )
                continue
            p_ins.create_shelf_id()
            p_ins.crop_shelf_compatible = {
                crop_id: dict() for crop_id in p_ins.crop_id_list
            }
            for s_index, shelf_id in enumerate(p_ins.shelf_id_list):
                for c_index, crop_id in enumerate(p_ins.crop_id_list):
                    p_ins.crop_shelf_compatible[crop_id][shelf_id] = (
                        crop_shelf_compatible[c_index][s_index] == 1
                    )

                p_ins.capacity[shelf_id] = dict()
                for cfg_index, config_id in enumerate(p_ins.config_id_list):
                    p_ins.capacity[shelf_id][config_id] = capacity[s_index][cfg_index]
        else:
            p_ins.n_shelf_types = n_shelf_types
            p_ins.create_shelf_type()

            p_ins.num_shelves = dict()
            for st_index, shelf_type in enumerate(p_ins.shelf_type_list):
                p_ins.num_shelves[shelf_type] = num_shelves[st_index]
            try:
                p_ins.check_integrity()
            except ValueError:
                continue

            p_ins.crop_shelf_type_compatible = {
                crop_id: dict() for crop_id in p_ins.crop_id_list
            }
            for st_index, shelf_type in enumerate(p_ins.shelf_type_list):
                for c_index, crop_id in enumerate(p_ins.crop_id_list):
                    p_ins.crop_shelf_type_compatible[crop_id][shelf_type] = (
                        crop_shelf_type_compatible[c_index][st_index] == 1
                    )

                p_ins.capacity[shelf_type] = dict()
                for cfg_index, config_id in enumerate(p_ins.config_id_list):
                    p_ins.capacity[shelf_type][config_id] = capacity[st_index][
                        cfg_index
                    ]

            p_ins.create_shelf_id()
            if model_type == "st_pen":
                p_ins.missed_demand_penalty = dict()
                for c_index, crop_id in enumerate(p_ins.crop_id_list):
                    p_ins.missed_demand_penalty[crop_id] = dict()
                    for demand_day in p_ins.t_list:
                        penalty_val = missed_demand_penalty[c_index][demand_day]
                        if penalty_val > 0:
                            p_ins.missed_demand_penalty[crop_id][
                                demand_day
                            ] = penalty_val

        yield p_ins


def main():
    start_d = datetime.datetime.now()
    fn_splitter = "-"
    input_foldername, input_ext = "../input_data/dat_files", ".dat"
    # input_foldername, input_ext = (
    #     "C:/Users/jt/code/crop-growth-planning-vf/data",
    #     ".dat",
    # )
    output_foldername, output_ext = "../input_data/json_files", ".json"
    output_encoding = "utf-8"

    convert_dat_to_json(
        PurePath(input_foldername),
        input_ext,
        PurePath(output_foldername),
        output_ext,
        fn_splitter,
        output_encoding,
    )

    end_d = datetime.datetime.now()
    elapsed_d = end_d - start_d
    print(f"{__name__} program end @ {end_d}"[:-3] + f"; took total {elapsed_d}"[:-3])


if __name__ == "__main__":
    main()
