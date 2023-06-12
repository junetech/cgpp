import json
from typing import Iterable
from meta_class import InputMetadata
from input_class import ProbInsS21, ProbInsS21T1, ProbInsS21T2, ProbInsS21T3


def p_ins_iter(input_meta: InputMetadata) -> Iterable[ProbInsS21]:
    for fp in input_meta.input_data_f_loc_iter():
        input_dict = json.load(open(fp, "r", encoding=input_meta.encoding))

        filename = fp.stem
        # instance name information
        filename_info_list = filename.split(input_meta.fn_splitter)
        if len(filename_info_list) == 7:
            model_type = filename_info_list[6]
        else:
            model_type = f"{filename_info_list[6]}_{filename_info_list[7]}"

        if model_type not in input_meta.input_model_type_list:
            continue

        p_ins: ProbInsS21
        if model_type == input_meta.model_type_t1:
            p_ins = ProbInsS21T1(**input_dict)
        elif model_type == input_meta.model_type_t2:
            p_ins = ProbInsS21T2(**input_dict)
        elif model_type == input_meta.model_type_t3:
            p_ins = ProbInsS21T3(**input_dict)
        else:
            raise ValueError(f"Unknown model type ({model_type}) of file ({filename})")
        yield p_ins
