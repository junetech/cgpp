import json
from pathlib import PurePath
from . import AaRootMetadata, InputMetadata, OutputMetadata


def create_input_meta_ins(root_metadata: AaRootMetadata) -> InputMetadata:
    meta_fn = root_metadata.input_meta_fn
    meta_enc = root_metadata.encoding

    input_meta_f_loc = PurePath(root_metadata.get_meta_dir, meta_fn)
    input_meta_file = json.load(open(input_meta_f_loc, "r", encoding=meta_enc))
    return_ins = InputMetadata(**input_meta_file)
    return_ins.set_base_dir(root_metadata.get_base_dir)
    return return_ins


def create_output_meta_ins(root_metadata: AaRootMetadata) -> OutputMetadata:
    meta_fn = root_metadata.output_meta_fn
    meta_enc = root_metadata.encoding

    input_meta_f_loc = PurePath(root_metadata.get_meta_dir, meta_fn)
    input_meta_file = json.load(open(input_meta_f_loc, "r", encoding=meta_enc))
    return_ins = OutputMetadata(**input_meta_file)
    return_ins.set_base_dir(root_metadata.get_base_dir)
    return return_ins
