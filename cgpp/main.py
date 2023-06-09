import datetime
import json
import logging
import sys
from pathlib import Path


from meta_class import (
    AaRootMetadata,
    OutputMetadata,
    create_input_meta_ins,
    create_output_meta_ins,
)
import input_func
from input_class import ProbInsS21
from output_func import schedule_by_santini_21_milp

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)


def create_aaroot_meta_ins() -> AaRootMetadata:
    meta_dir = "metadata"
    aaroot_meta_fn = "aaroot_metadata.json"
    aaroot_meta_enc = "utf-8"

    pkg_location = Path(__file__).parent
    base_location = pkg_location.parent
    meta_location = base_location.joinpath(meta_dir)

    aaroot_meta_f_loc = meta_location.joinpath(aaroot_meta_fn)
    aaroot_meta_file = json.load(open(aaroot_meta_f_loc, "r", encoding=aaroot_meta_enc))
    return_ins = AaRootMetadata(**aaroot_meta_file)

    return_ins.set_base_dir(base_location)
    return_ins.set_meta_dir(meta_location)
    return return_ins


def solve_one_prob_ins(
    p_ins: ProbInsS21,
    output_meta: OutputMetadata,
    solver_name: str,
    timelimit: int,
):
    logging.info(p_ins.problem_name)
    schedule_by_santini_21_milp(p_ins, solver_name, timelimit, output_meta)


def read_and_solve_all(root_meta: AaRootMetadata):
    # read metadata
    input_meta = create_input_meta_ins(root_meta)
    output_meta = create_output_meta_ins(root_meta)
    # read problem parameters
    for p_ins in input_func.p_ins_iter(input_meta):
        solve_one_prob_ins(
            p_ins, output_meta, input_meta.solver_name, input_meta.timelimit
        )


def main():
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

    start_d = datetime.datetime.now()
    logging.info(f"{__name__} program start @ {start_d}"[:-3])

    read_and_solve_all(root_meta)

    end_d = datetime.datetime.now()
    elapsed_d = end_d - start_d
    logging.info(
        f"{__name__} program end @ {end_d}"[:-3] + f"; took total {elapsed_d}"[:-3]
    )


if __name__ == "__main__":
    main()
