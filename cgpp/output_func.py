from input_class import ProbInsS21, ProbInsS21T1, ProbInsS21T2, ProbInsS21T3
from meta_class import OutputMetadata
from openpyxl.utils import get_column_letter
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from output_class import Variables
from santini_21_milp_t1 import solve_santini_21_milp_t1
from santini_21_milp_t2 import solve_santini_21_milp_t2
from santini_21_milp_t3 import solve_santini_21_milp_t3


def schedule_by_santini_21_milp_t1(
    p_ins: ProbInsS21T1,
    solver_name: str,
    timelimit: int,
):
    _, sol = solve_santini_21_milp_t1(
        p_ins,
        solver_name,
        timelimit,
    )


def schedule_by_santini_21_milp_t2(
    p_ins: ProbInsS21T2,
    solver_name: str,
    timelimit: int,
):
    _, sol = solve_santini_21_milp_t2(p_ins, solver_name, timelimit, 2)


def schedule_by_santini_21_milp_t3(
    p_ins: ProbInsS21T3, solver_name: str, timelimit: int, output_meta: OutputMetadata
):
    _, sol = solve_santini_21_milp_t3(p_ins, solver_name, timelimit)
    make_santini_21_sch_xlsx(sol, p_ins, output_meta)


def schedule_by_santini_21_milp(
    p_ins: ProbInsS21, solver_name: str, timelimit: int, output_meta: OutputMetadata
):
    p_ins.create_t_idx_list()
    if p_ins.model_type == "s":
        # schedule_by_santini_21_milp_t1(p_ins, solver_name, timelimit)
        pass
    elif p_ins.model_type == "st":
        # schedule_by_santini_21_milp_t2(p_ins, solver_name, timelimit)
        pass
    elif p_ins.model_type == "st_pen":
        schedule_by_santini_21_milp_t3(p_ins, solver_name, timelimit, output_meta)
    else:
        raise ValueError(f"Unknown model type {p_ins.model_type}")


def make_santini_21_sch_xlsx(
    solution: Variables, p_ins: ProbInsS21, output_meta: OutputMetadata
):
    # Indices
    c_list = p_ins.crop_id_list  # c\in C in paper
    t_list: list[str]
    if type(p_ins) == ProbInsS21T1:
        t_list = p_ins.shelf_id_list  # s\in S in paper
    else:
        t_list = p_ins.shelf_type_list  # t\in T in paper
    k_list = p_ins.config_id_list  # k\in K in paper
    demand_d_list = p_ins.t_idx_list  # d\in D in paper; set of demand days
    growth_d_list = p_ins.t_idx_list[:-1]  # d\in D in paper; set of demand days
    d_prime_list = [0] + growth_d_list  # extended time horizon

    # Parameters
    # c -> d -> demand quantity
    p_dict = p_ins.make_demand_dict()
    # c -> t -> the crop can grow on the shelf type
    delta_dict: dict[str, dict[int, bool]]
    if type(p_ins) == ProbInsS21T1:
        delta_dict = p_ins.crop_shelf_compatible
    else:
        delta_dict = p_ins.crop_shelf_type_compatible
    # c -> list of compatible t
    T_dict = {c: [t for t in t_list if delta_dict[c][t]] for c in c_list}
    # t -> number of shelves
    n_dict: dict[str, int]
    if type(p_ins) == ProbInsS21T1:
        n_dict = {t: 1 for t in t_list}
    else:
        n_dict = p_ins.num_shelves
    # c -> the number of growth days
    gamma_dict = p_ins.crop_growth_days
    # c -> g (growth day) -> configuration required
    k_dict = p_ins.crop_growth_day_config_dict
    # s -> t -> capacity
    q_dict = p_ins.capacity

    # seed vault index
    sigma = "SeedVault"
    # produce stage index
    tau = "ProduceStage"
    t_prime_list: list[str]
    if type(p_ins) == ProbInsS21T1:
        t_prime_list = [sigma] + p_ins.shelf_id_list + [tau]
    else:
        t_prime_list = [sigma] + p_ins.shelf_type_list + [tau]
    for c in c_list:
        delta_dict[c][sigma] = True
        delta_dict[c][tau] = True
    T_prime_dict = {c: [sigma] + T_dict[c] + [tau] for c in c_list}

    # create empty workbook
    wb = Workbook()

    cursor = {"row": 1, "column": 1}  # each index starts from 1
    # schedule worksheet
    sch_ws: Worksheet = wb.worksheets[0]
    sch_ws.title = output_meta.schedule_sheetname
    # shelf configuration worksheet
    cfg_ws: Worksheet = wb.create_sheet(output_meta.shelf_config_sheetname)
    # germination worksheet
    grm_ws: Worksheet = wb.create_sheet(output_meta.germination_sheetname)
    # harvest worksheet
    hv_ws: Worksheet = wb.create_sheet(output_meta.harvest_sheetname)

    crop_ws_list: list[Worksheet] = [sch_ws, grm_ws, hv_ws]
    shelf_ws_list: list[Worksheet] = [cfg_ws]
    ws_list: list[Worksheet] = crop_ws_list + shelf_ws_list

    cursor["row"] = 2
    time_row = [output_meta.keystone_val] + [0] + p_ins.t_idx_list

    for ws in ws_list:
        ws.cell(**cursor).value = p_ins.problem_name
        ws.freeze_panes = ws["B2"]
        for idx, val in enumerate(time_row):
            col = idx + cursor["column"]
            # define column width
            ws.column_dimensions[get_column_letter(col)].width = output_meta.sch_col_wth
            # write time index row
            ws.cell(column=col, row=cursor["row"]).value = val
        # first column width
        ws.column_dimensions[
            get_column_letter(cursor["column"])
        ].width = output_meta.first_col_wth

    # Shelf configuration worksheet
    for shelf_id in t_list:
        cursor["row"] += 1
        cfg_ws.cell(**cursor).value = shelf_id
        for t_idx in d_prime_list:
            for cfg_id in k_list:
                if solution.y[shelf_id][t_idx][cfg_id] == 0:
                    continue
                col = (
                    cursor["column"] + 1 + t_idx
                )  # +1 since column t_idx starts from 0
                cfg_ws.cell(column=col, row=cursor["row"]).value = cfg_id

    # Crop information worksheets
    cursor["row"] = 2
    for shelf_id in t_list:
        cursor["row"] += 1
        for ws in crop_ws_list:
            ws.cell(**cursor).value = shelf_id

        row_key_list: list[str] = list()
        grm_row_list_dict: dict[str, list[int]] = dict()  # row_key -> row_list
        sch_row_list_dict: dict[str, list[int]] = dict()  # row_key -> row_list
        hv_row_list_dict: dict[str, list[int]] = dict()  # row_key -> row_list

        for crop_id in c_list:
            if shelf_id not in T_prime_dict[crop_id]:
                continue
            for t_idx in growth_d_list:
                row_key = f"{crop_id}_{t_idx}"
                if row_key not in row_key_list:
                    row_key_list.append(row_key)
                grm_row_list_dict[row_key] = [0] * p_ins.n_days
                grm_unit_count = solution.x[crop_id][0][sigma][t_idx][shelf_id]
                if grm_unit_count > 0:
                    grm_row_list_dict[row_key][t_idx] = grm_unit_count

                for g in range(gamma_dict[crop_id] + 1):
                    crop_unit_count = sum(
                        solution.x[crop_id][g][shelf_id][t_idx][t2]
                        for t2 in T_prime_dict[crop_id]
                    )
                    if crop_unit_count == 0:
                        continue
                    tp = t_idx - g
                    row_key = f"{crop_id}_{tp}"
                    if row_key not in row_key_list:
                        row_key_list.append(row_key)
                    if row_key not in sch_row_list_dict:
                        sch_row_list_dict[row_key] = [0] * p_ins.n_days
                    sch_row_list_dict[row_key][t_idx] = crop_unit_count

                row_key = f"{crop_id}_{t_idx - gamma_dict[c]}"
                if row_key not in row_key_list:
                    row_key_list.append(row_key)
                hv_row_list_dict[row_key] = [0] * p_ins.n_days
                hv_unit_count = solution.x[crop_id][gamma_dict[c]][shelf_id][t_idx][tau]
                if hv_unit_count > 0:
                    hv_row_list_dict[row_key][t_idx + 1] = hv_unit_count

        for row_key in row_key_list:
            cursor["row"] += 1
            for ws in crop_ws_list:
                ws.cell(**cursor).value = row_key
            if row_key in sch_row_list_dict:
                sch_row_list = sch_row_list_dict[row_key]
            else:
                sch_row_list = [0] * p_ins.n_days
            if row_key in grm_row_list_dict:
                grm_row_list = grm_row_list_dict[row_key]
            else:
                grm_row_list = [0] * p_ins.n_days
            if row_key in hv_row_list_dict:
                hv_row_list = hv_row_list_dict[row_key]
            else:
                hv_row_list = [0] * p_ins.n_days

            if sum(sch_row_list) + sum(grm_row_list) + sum(hv_row_list) == 0:
                break

            for c_idx, sch_row_val in enumerate(sch_row_list):
                if sch_row_val == 0:
                    continue
                col = cursor["column"] + 1 + c_idx
                sch_ws.cell(column=col, row=cursor["row"]).value = sch_row_val
            for c_idx, grm_row_val in enumerate(grm_row_list):
                if grm_row_val == 0:
                    continue
                col = cursor["column"] + 1 + c_idx
                grm_ws.cell(column=col, row=cursor["row"]).value = grm_row_val
            for c_idx, hv_row_val in enumerate(hv_row_list):
                if hv_row_val == 0:
                    continue
                hv_ws.cell(column=col, row=cursor["row"]).value = hv_row_val

    f_loc = output_meta.crop_schedule_f_loc(p_ins.problem_name, 3)
    wb.save(filename=f_loc)
