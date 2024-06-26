import ast
import datetime
import json
import logging
import sys
from pathlib import Path, PurePath
from typing import Iterable

from input_class import ProbInsS21, ProbInsS21T1, ProbInsS21T2, ProbInsS21T3
from main import create_aaroot_meta_ins
from meta_class import InputMetadata, create_input_meta_ins

FN_SPLITTER = "-"
INPUT_DIR, INPUT_EXT = "input_data/dat_files/", ".dat"
# INPUT_DIR, INPUT_EXT = (
#     "C:/Users/jt/code/crop-growth-planning-vf/data/",
#     ".dat",
# )

N_CROPS_LIST: list[int] = [1, 2, 3, 4, 5, 6]
N_SHELVES_DICT: dict[int, str] = {7: "small", 9: "medium", 12: "large"}
DEMAND_MULT_LIST: list[float] = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
TIME_HORIZON_LIST: list[int] = [60, 80, 100]

N_CROPS_COUNT: dict[int, int] = {v: 0 for v in N_CROPS_LIST}
N_SHELVES_COUNT: dict[str, int] = {v: 0 for v in N_SHELVES_DICT.values()}
DEMAND_MULT_COUNT: dict[float, int] = {v: 0 for v in DEMAND_MULT_LIST}
TIME_HORIZON_COUNT: dict[int, int] = {v: 0 for v in TIME_HORIZON_LIST}


def shelf_id_str(idx: int) -> str:
    return f"s_{idx}"


def shelf_type_str(idx: int) -> str:
    return f"st_{idx}"


def config_id_str(idx: int) -> str:
    return f"cf_{idx}"


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


def write_p_ins_as_json(p_ins: ProbInsS21, output_meta: InputMetadata):
    output_dir = output_meta.input_dir()
    output_ext = output_meta.input_ext
    output_encoding = output_meta.encoding

    fn = f"{p_ins.problem_name}-{p_ins.model_type}{output_ext}"
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


def convert_dat_to_json(
    input_dir: PurePath, input_ext: str, fn_splitter: str, input_meta: InputMetadata
):
    input_abs_dir = input_meta.get_abs_dir(input_dir)
    input_meta.input_dir().mkdir(parents=True, exist_ok=True)

    conversion_count = 0
    for p_ins in generate_p_ins_from_dat(
        input_abs_dir, input_ext, fn_splitter, input_meta.input_model_type_list
    ):
        write_p_ins_as_json(p_ins, input_meta)
        conversion_count += 1
    logging.info(f"{conversion_count} files converted")
    logging.info(f"The number of crops: {N_CROPS_COUNT}")
    logging.info(f"The capacity by shelves: {N_SHELVES_COUNT}")
    logging.info(f"Demand multipliers: {DEMAND_MULT_COUNT}")
    logging.info(f"Time horizons: {TIME_HORIZON_COUNT}")


def generate_p_ins_from_dat(
    input_dir: PurePath,
    input_ext: str,
    fn_splitter: str,
    input_model_type_list: list[str],
) -> Iterable[ProbInsS21]:
    input_path = Path(input_dir)
    fp_list: list[Path] = [fp for fp in input_path.iterdir() if fp.suffix == input_ext]
    logging.info(f"{len(fp_list)} files found in {input_path}")
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
            ins_id,
            model_type,
        ] = (
            int(filename_info_list[0]),
            int(filename_info_list[1]),
            filename_info_list[2],
            int(filename_info_list[3]),
            float(filename_info_list[4]),
            int(filename_info_list[5]),
            "_".join(filename_info_list[6:]),
        )
        DEMAND_MULT_COUNT[demand_mult] += 1

        if model_type not in input_model_type_list:
            continue

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

        # crop information
        if n_crops != n_crops_fn:
            logging.info(
                UserWarning(
                    f"n_crops of filename {n_crops_fn} != n_crops of data {n_crops}"
                )
            )
            continue
        N_CROPS_COUNT[n_crops] += 1

        if n_days != n_days_fn:
            logging.info(
                UserWarning(
                    f"n_days of filename {n_days_fn} != n_days of data {n_days}"
                )
            )
            continue
        TIME_HORIZON_COUNT[n_days] += 1

        crop_id_list = [char for char in crop_id_string]
        config_id_list = [config_id_str(idx + 1) for idx in range(n_configurations)]
        day_list = [idx + 1 for idx in range(n_days)]

        crop_growth_days_dict = dict()
        for c_index, crop_id in enumerate(crop_id_list):
            crop_growth_days_dict[crop_id] = crop_growth_days[c_index]

        cgdc_dict: dict[str, list[str]] = dict()
        demand_dict = dict()
        for c_index, crop_id in enumerate(crop_id_list):
            cgdc_dict[crop_id] = [
                config_id_str(crop_growth_day_config[c_index][growth_day])
                for growth_day in range(crop_growth_days_dict[crop_id])
                if crop_growth_day_config[c_index][growth_day] != -1
            ]
            demand_dict[crop_id] = {
                demand_day: demand[c_index][day_idx]
                for day_idx, demand_day in enumerate(day_list)
                if demand[c_index][day_idx] > 0
            }

        # instance to be returned
        p_ins: ProbInsS21

        # shelf information
        capa_dict: dict[str, dict[str, int]] = dict()
        if model_type == "s":
            if n_shelves != n_shelves_fn:
                logging.info(
                    UserWarning(
                        f"n_shelves of filename {n_shelves_fn} != n_shelves of data {n_shelves}"
                    )
                )
                continue
            assert n_shelves in N_SHELVES_DICT.keys()
            N_SHELVES_COUNT[N_SHELVES_DICT[n_shelves]] += 1

            shelf_id_list: list[str] = [
                shelf_id_str(idx + 1) for idx in range(n_shelves)
            ]
            csc_dict: dict[str, dict[str, bool]] = {
                crop_id: dict() for crop_id in crop_id_list
            }
            for s_index, shelf_id in enumerate(shelf_id_list):
                for c_index, crop_id in enumerate(crop_id_list):
                    csc_dict[crop_id][shelf_id] = (
                        crop_shelf_compatible[c_index][s_index] == 1
                    )

                capa_dict[shelf_id] = dict()
                for cfg_index, config_id in enumerate(config_id_list):
                    capa_dict[shelf_id][config_id] = capacity[s_index][cfg_index]
            p_ins = ProbInsS21T1(
                n_shelves=n_shelves,
                n_crops=n_crops,
                crop_id_string=crop_id_string,
                n_days=n_days,
                demand_mult=demand_mult,
                id=ins_id,
                model_type=model_type,
                cabinet=N_SHELVES_DICT[n_shelves],
                crop_id_list=crop_id_list,
                config_id_list=config_id_list,
                crop_growth_days=crop_growth_days_dict,
                n_configurations=n_configurations,
                crop_growth_day_config=cgdc_dict,
                capacity=capa_dict,
                demand=demand_dict,
                shelf_id_list=shelf_id_list,
                crop_shelf_compatible=csc_dict,
            )
        else:
            shelf_type_list: list[str] = [
                shelf_type_str(idx + 1) for idx in range(n_shelf_types)
            ]
            shelf_id_dict: dict[str, list[str]] = {
                shelf_type: list() for shelf_type in shelf_type_list
            }
            ns_dict = dict()
            for st_index, shelf_type in enumerate(shelf_type_list):
                ns_dict[shelf_type] = num_shelves[st_index]

            idx = 0
            for shelf_type in shelf_type_list:
                shelf_type_count = ns_dict[shelf_type]
                shelf_id_dict[shelf_type].extend(
                    [shelf_id_str(n + idx + 1) for n in range(shelf_type_count)]
                )
                idx += shelf_type_count
            n_shelves = idx
            assert n_shelves in N_SHELVES_DICT.keys()
            N_SHELVES_COUNT[N_SHELVES_DICT[n_shelves]] += 1

            cstc_dict: dict[str, dict[str, bool]] = {
                crop_id: dict() for crop_id in crop_id_list
            }
            for st_index, shelf_type in enumerate(shelf_type_list):
                for c_index, crop_id in enumerate(crop_id_list):
                    cstc_dict[crop_id][shelf_type] = (
                        crop_shelf_type_compatible[c_index][st_index] == 1
                    )

                capa_dict[shelf_type] = dict()
                for cfg_index, config_id in enumerate(config_id_list):
                    capa_dict[shelf_type][config_id] = capacity[st_index][cfg_index]

            if model_type == "st_pen":
                mdp_dict: dict[str, dict[int, int]] = dict()
                for c_index, crop_id in enumerate(crop_id_list):
                    mdp_dict[crop_id] = dict()
                    for day_idx, demand_day in enumerate(day_list):
                        penalty_val = missed_demand_penalty[c_index][day_idx]
                        if penalty_val > 0:
                            mdp_dict[crop_id][demand_day] = penalty_val
                p_ins = ProbInsS21T3(
                    n_shelves=n_shelves,
                    n_crops=n_crops,
                    crop_id_string=crop_id_string,
                    n_days=n_days,
                    demand_mult=demand_mult,
                    id=ins_id,
                    model_type=model_type,
                    cabinet=N_SHELVES_DICT[n_shelves],
                    crop_id_list=crop_id_list,
                    config_id_list=config_id_list,
                    crop_growth_days=crop_growth_days_dict,
                    n_configurations=n_configurations,
                    crop_growth_day_config=cgdc_dict,
                    capacity=capa_dict,
                    demand=demand_dict,
                    n_shelf_types=n_shelf_types,
                    shelf_type_list=shelf_type_list,
                    num_shelves=ns_dict,
                    shelf_id_dict=shelf_id_dict,
                    crop_shelf_type_compatible=cstc_dict,
                    missed_demand_penalty=mdp_dict,
                )
            else:
                p_ins = ProbInsS21T2(
                    n_shelves=n_shelves,
                    n_crops=n_crops,
                    crop_id_string=crop_id_string,
                    n_days=n_days,
                    demand_mult=demand_mult,
                    id=ins_id,
                    model_type=model_type,
                    cabinet=N_SHELVES_DICT[n_shelves],
                    crop_id_list=crop_id_list,
                    config_id_list=config_id_list,
                    crop_growth_days=crop_growth_days_dict,
                    n_configurations=n_configurations,
                    crop_growth_day_config=cgdc_dict,
                    capacity=capa_dict,
                    demand=demand_dict,
                    n_shelf_types=n_shelf_types,
                    shelf_type_list=shelf_type_list,
                    num_shelves=ns_dict,
                    shelf_id_dict=shelf_id_dict,
                    crop_shelf_type_compatible=cstc_dict,
                )

        yield p_ins


def main():
    start_d = datetime.datetime.now()
    # read root metadata
    root_meta = create_aaroot_meta_ins()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        root_meta.log_filename, encoding=root_meta.log_encoding
    )
    handler.setFormatter(logging.Formatter(root_meta.log_format))
    root_logger.addHandler(handler)

    # show log messages on terminal as well
    root_logger.addHandler(logging.StreamHandler(sys.stdout))

    logging.info(f"{__name__} program start @ {start_d}"[:-3])

    input_meta: InputMetadata = create_input_meta_ins(create_aaroot_meta_ins())
    convert_dat_to_json(PurePath(INPUT_DIR), INPUT_EXT, FN_SPLITTER, input_meta)

    end_d = datetime.datetime.now()
    elapsed_d = end_d - start_d
    logging.info(
        f"{__name__} program end @ {end_d}"[:-3] + f"; took total {elapsed_d}"[:-3]
    )


if __name__ == "__main__":
    main()
