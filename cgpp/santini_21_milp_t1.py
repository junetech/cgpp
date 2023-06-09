import logging
from itertools import product

from input_class import ProbInsS21T1
from ortools.linear_solver.linear_solver_natural_api import SumArray
from ortools.linear_solver.pywraplp import Solver


def t_list_before(t: int, t_list: list[int]) -> list[int]:
    return [u for u in t_list if u <= t]


def t_list_after(t: int, t_list: list[int]) -> list[int]:
    return [u for u in t_list if u >= t]


def solve_santini_21_milp_t1(p_ins: ProbInsS21T1, solver_name: str, timelimit: int):
    # Indices
    c_list = p_ins.crop_id_list  # c\in C in paper
    s_list = p_ins.shelf_id_list  # s\in S in paper
    k_list = p_ins.config_id_list  # k\in K in paper
    full_d_list = p_ins.t_idx_list  # d\in D in paper; set of demand days
    d_list = full_d_list[:-1]  # d\in D in paper; set of demand days
    d_prime_list = [0] + d_list  # extended time horizon

    # Parameters
    # c -> d -> demand quantity
    p_dict = p_ins.make_demand_dict()
    # c -> t -> the crop can grow on the shelf type
    delta_dict = p_ins.crop_shelf_type_compatible
    # c -> list of compatible t
    T_dict = {c: [t for t in t_list if delta_dict[c][t]] for c in c_list}
    # t -> number of shelves
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
    t_prime_list = [sigma] + p_ins.shelf_type_list + [tau]
    for c in c_list:
        delta_dict[c][sigma] = True
        delta_dict[c][tau] = True
    T_prime_dict = {c: [sigma] + T_dict[c] + [tau] for c in c_list}

    solver: Solver = Solver.CreateSolver(solver_name)
    infty = Solver.infinity()

    # Variables
    # the number of units of crop c in their day of growth g,
    # growing on shelves of type t1 on day d and going to shelves of type t2 on day d+1
    x = {
        c: {
            g: {
                t1: {
                    d: {
                        t2: solver.IntVar(
                            lb=0, ub=infty, name=f"x^{c},{g}_{t1},{d},{t2}"
                        )
                        for t2 in T_prime_dict[c]
                    }
                    for d in d_prime_list
                }
                for t1 in T_prime_dict[c]
            }
            for g in range(gamma_dict[c] + 1)
        }
        for c in c_list
    }
    # the number of shelves of type t with configuration k on day d
    # TODO: shelf type에서 쓰이는 configuration list의 dictionary를 쓰자
    y = {
        t: {
            d: {k: solver.IntVar(lb=0, ub=infty, name=f"y_{t},{d},{k}") for k in k_list}
            for d in d_list
        }
        for t in t_list
    }

    def add_obj_3():
        # Objective 3: minimize unmet demand
        omega_dict = p_ins.missed_demand_penalty
        # c -> d -> the amount of unmet demand of crop c on day d
        u = {
            c: {
                d: solver.IntVar(lb=0, ub=infty, name=f"u_{c},{d}") for d in full_d_list
            }
            for c in c_list
        }
        obj_func = solver.Objective()
        for c, d in product(c_list, full_d_list):
            if d not in omega_dict[c]:
                continue
            obj_func.SetCoefficient(u[c][d], omega_dict[c][d])
        obj_func.SetMinimization()

        # constraint 9 instead of constraint 1
        for c, d in product(c_list, full_d_list):
            solver.Add(
                SumArray(x[c][gamma_dict[c]][t][d - 1][tau] for t in T_dict[c])
                + u[c][d]
                == p_dict[c][d]
            )
        # constraint 10 instead of constrain 3
        for c in c_list:
            for d in t_list_after(gamma_dict[c] + 1, full_d_list):
                solver.Add(
                    SumArray(
                        x[c][0][sigma][d - gamma_dict[c] - 1][t] for t in T_dict[c]
                    )
                    + u[c][d]
                    == p_dict[c][d]
                )

    if obj_idx == 3:
        add_obj_3()
    else:
        # constraint 1: demand satisfaction
        for c, d in product(c_list, full_d_list):
            solver.Add(
                SumArray(x[c][gamma_dict[c]][t][d - 1][tau] for t in T_dict[c])
                == p_dict[c][d]
            )

        # constraint 3: planting constraints
        for c in c_list:
            for d in t_list_after(gamma_dict[c] + 1, full_d_list):
                solver.Add(
                    SumArray(
                        x[c][0][sigma][d - gamma_dict[c] - 1][t] for t in T_dict[c]
                    )
                    == p_dict[c][d]
                )

    # constraint 2: capacity constraints
    for t1, d, k in product(t_list, d_list, k_list):
        lhs = SumArray(
            x[c][g][t1][d][t2]
            for t2 in t_prime_list
            for c in c_list
            if delta_dict[c][t1] and delta_dict[c][t2]
            for g in t_list_before(gamma_dict[c], full_d_list)
            if k_dict[c][g] == k
        )
        solver.Add(lhs <= q_dict[t1][k] * y[t1][d][k])

    # constraint 4: flow-balance constraints
    for c in c_list:
        # fix: gamma_dict[c] in paper -> gamma_dict[c] - 1
        for g in t_list_before(gamma_dict[c] - 1, d_prime_list):
            for t_1, d in product(t_list, d_list):
                lhs = SumArray(x[c][g][t2][d - 1][t1] for t2 in T_prime_dict[c])
                rhs = SumArray(x[c][g + 1][t1][d][t2] for t2 in T_prime_dict[c])
                solver.Add(lhs == rhs)

    # constraint 5: configuration constraints
    for t, d in product(t_list, d_list):
        solver.Add(SumArray(y[t][d][k] for k in k_list) == n_dict[t])

    obj_val: float = 0.0
    solution = dict()
    # solve
    solver.set_time_limit(timelimit * 1000)
    phase = solver.Solve()
    wall_sec = solver.wall_time() / 1000

    _str: str = "solver phase log placeholder"
    if phase == Solver.OPTIMAL:
        _str = (
            f"*Optimal* objective value = {solver.Objective().Value()}\t"
            + f"found in {wall_sec:.3f} seconds"
        )
    elif phase == Solver.FEASIBLE:
        _str = (
            f"An incumbent objective value = {solver.Objective().Value()}\t"
            + f"and best bound = {solver.Objective().BestBound()}\t"
            + f"found in {wall_sec:.3f} seconds"
        )
    elif phase == Solver.INFEASIBLE:
        _str = f"The problem is found to be infeasible in {wall_sec:.3f} seconds"
    elif phase == Solver.UNBOUNDED:
        _str = f"The problem is found to be unbounded in {wall_sec:.3f} seconds"
    elif phase == Solver.NOT_SOLVED:
        _str = f"No solution found in {wall_sec:.3f} seconds"
    logging.info(_str)

    if phase == Solver.OPTIMAL or phase == Solver.FEASIBLE:
        obj_val = solver.Objective().Value()
        _str = f"Problem solved in {solver.iterations()} iterations"
        _str += f" & {solver.nodes()} branch-and-bound nodes"
        logging.info(_str)

    return obj_val, solution
