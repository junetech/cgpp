from input_class import ProbInsS21, ProbInsS21T1, ProbInsS21T2, ProbInsS21T3
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
    p_ins: ProbInsS21T3,
    solver_name: str,
    timelimit: int,
):
    _, sol = solve_santini_21_milp_t3(p_ins, solver_name, timelimit)


def schedule_by_santini_21_milp(
    p_ins: ProbInsS21,
    solver_name: str,
    timelimit: int,
):
    p_ins.create_t_idx_list()
    if p_ins.model_type == "s":
        # schedule_by_santini_21_milp_t1(p_ins, solver_name, timelimit)
        pass
    elif p_ins.model_type == "st":
        # schedule_by_santini_21_milp_t2(p_ins, solver_name, timelimit)
        pass
    elif p_ins.model_type == "st_pen":
        schedule_by_santini_21_milp_t3(p_ins, solver_name, timelimit)
    else:
        raise ValueError(f"Unknown model type {p_ins.model_type}")
