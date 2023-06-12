import logging
import math
from itertools import product

from input_class import ProbInsS21T2
from ortools.linear_solver.linear_solver_natural_api import SumArray
from ortools.linear_solver.pywraplp import Objective, Solver
from output_class import Variables, VariablesObj2, VariablesObj4


def t_list_before(t: int, t_list: list[int]) -> list[int]:
    return [u for u in t_list if u <= t]


def t_list_after(t: int, t_list: list[int]) -> list[int]:
    return [u for u in t_list if u >= t]


def solve_santini_21_milp_t2(
    p_ins: ProbInsS21T2, solver_name: str, timelimit: int, obj_idx: int
) -> Variables:
    if obj_idx == 1:
        return solve_santini_21_milp_t2_obj1(p_ins, solver_name, timelimit)
    elif obj_idx == 2:
        return solve_santini_21_milp_t2_obj2(p_ins, solver_name, timelimit)
    elif obj_idx == 4:
        return solve_santini_21_milp_t2_obj4(p_ins, solver_name, timelimit)
    else:
        raise ValueError(f"Wrong obj_idx {obj_idx} as input")


def solve_santini_21_milp_t2_obj1(
    p_ins: ProbInsS21T2, solver_name: str, timelimit: int
) -> Variables:
    print("Under contruction")
    return Variables()


def solve_santini_21_milp_t2_obj2(
    p_ins: ProbInsS21T2, solver_name: str, timelimit: int
) -> VariablesObj2:
    # from pprint import pprint

    # Indices
    C_list = p_ins.crop_id_list  # c\in C in paper
    # S_list = p_ins.shelf_id_list
    D_list = p_ins.t_idx_list[:-1]  # d\in D in paper; set of growth days
    d_bar = p_ins.t_idx_list[-1]  # d_bar in paper
    D_prime_list = [0] + D_list  # extended time horizon
    # day 0 is the earliest day seeds can be planted, day 1 is the earliest day of growth counted
    K_list = p_ins.config_id_list  # k\in K in paper
    T_list = p_ins.shelf_type_list  # t\in T in paper

    # print(C_list)
    # print(T_list)
    # print(K_list)
    # print(demand_D_list)
    # print(growth_D_list)
    # print(D_prime_list)

    # Parameters
    # c -> d -> demand quantity
    p_dict = p_ins.make_demand_dict()
    # c -> s -> the crop can grow on the shelf type
    # delta_dict = p_ins.crop_shelf_compatible
    # c -> list of compatible s
    # S_dict = {c: [s for s in S_list if delta_dict[c][s]] for c in C_list}
    # c -> the number of growth days
    gamma_dict = p_ins.crop_growth_days
    # c -> g (growth day) -> configuration required
    k_dict = p_ins.crop_growth_day_config_dict
    # s -> t -> capacity
    q_dict = p_ins.capacity
    # c -> t -> the crop can grow on the shelf type
    delta_dict = p_ins.crop_shelf_type_compatible
    # c -> list of compatible t
    T_dict = {c: [t for t in T_list if delta_dict[c][t]] for c in C_list}
    # t -> number of shelves
    n_dict = p_ins.num_shelves

    # seed vault index
    sigma = "SeedVault"
    # produce stage index
    tau = "Produce"
    # S_prime_list = [sigma, tau] + S_list
    # S_prime_dict = {c: [sigma, tau] + S_dict[c] for c in C_list}
    T_prime_list = [sigma, tau] + T_list
    for c in C_list:
        delta_dict[c][sigma] = True
        delta_dict[c][tau] = True
    T_prime_dict = {c: [sigma, tau] + T_dict[c] for c in C_list}

    # pprint(p_dict)
    # pprint(delta_dict)
    # pprint(T_dict)
    # pprint(n_dict)
    # pprint(gamma_dict)
    # pprint(k_dict)
    # pprint(q_dict)
    # pprint(T_prime_list)
    # pprint(T_prime_dict)

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
                    for d in D_prime_list
                }
                for t1 in T_prime_dict[c]
            }
            for g in range(0, gamma_dict[c] + 1)
        }
        for c in C_list
    }
    # the number of shelves of type t with configuration k on day d
    # TODO: shelf type에서 쓰이는 configuration list의 dictionary를 쓰자
    y = {
        t: {
            d: {k: solver.IntVar(lb=0, ub=infty, name=f"y_{t},{d},{k}") for k in K_list}
            for d in D_list
        }
        for t in T_list
    }
    # constraint 1: demand satisfaction
    # note: the due date is a day after the last growth day
    for c, d in product(C_list, D_list + [d_bar]):
        solver.Add(
            SumArray(x[c][gamma_dict[c]][t][d - 1][tau] for t in T_dict[c])
            == p_dict[c][d]
        )

    # constraint 2: capacity constraints
    for t1, d, k in product(T_list, D_list, K_list):
        var_list = list()
        for t2 in T_prime_list:
            for c in C_list:
                if delta_dict[c][t1] and delta_dict[c][t2]:
                    for g in range(1, gamma_dict[c] + 1):
                        if k_dict[c][g] == k:
                            var_list.append(x[c][g][t1][d][t2])

        lhs = SumArray(var_list)
        solver.Add(lhs <= q_dict[t1][k] * y[t1][d][k])

    # constrain 3: planting constraints
    for c in C_list:
        for d in range(gamma_dict[c] + 1, d_bar + 1):
            solver.Add(
                SumArray(x[c][0][sigma][d - gamma_dict[c] - 1][t] for t in T_dict[c])
                == p_dict[c][d]
            )

    # constraint 4: flow-balance constraints
    for c in C_list:
        # fix: is gamma_dict[c]+1 in the paper
        for g in range(0, gamma_dict[c]):
            for t1 in T_dict[c]:
                for d in D_list:
                    lhs = SumArray(x[c][g][t2][d - 1][t1] for t2 in T_prime_dict[c])
                    rhs = SumArray(x[c][g + 1][t1][d][t2] for t2 in T_prime_dict[c])
                    solver.Add(lhs == rhs)

    # constraint 5: configuration constraints
    # TODO: better be equality
    for t, d in product(T_list, D_list):
        solver.Add(SumArray(y[t][d][k] for k in K_list) <= n_dict[t])

    # Objective 2: minimize # reconfiguration
    # t -> d -> k -> # shelves of type t reconfigured to or from configuration k on day d
    w = {
        t: {
            d: {k: solver.IntVar(lb=0, ub=infty, name=f"w_{t},{d},{k}") for k in K_list}
            for d in range(1, d_bar - 1)
        }
        for t in T_list
    }
    obj_func: Objective = solver.Objective()
    for t, d, k in product(T_list, range(1, d_bar - 1), K_list):
        obj_func.SetCoefficient(w[t][d][k], 1)
        solver.Add(w[t][d][k] >= y[t][d][k] - y[t][d + 1][k])
        solver.Add(w[t][d][k] >= y[t][d + 1][k] - y[t][d][k])
    obj_func.SetMinimization()

    # Variable fixing
    # TODO: these are constraints, not variable fixing
    for c, d in product(C_list, D_prime_list):
        for t, g in product(T_prime_dict[c], range(0, gamma_dict[c] + 1)):
            solver.Add(x[c][g][t][d][sigma] == 0)
            solver.Add(x[c][g][tau][d][t] == 0)
    for c, d in product(C_list, D_prime_list):
        for g in range(0, gamma_dict[c] + 1):
            solver.Add(x[c][g][sigma][d][tau] == 0)

    # TODO: these are constraints, not variable fixing
    for c, d in product(C_list, D_prime_list):
        for t, g in product(T_prime_dict[c], range(1, gamma_dict[c] + 1)):
            solver.Add(x[c][g][sigma][d][t] == 0)
    for c, d in product(C_list, D_prime_list):
        for t, g in product(T_prime_dict[c], range(0, gamma_dict[c])):
            solver.Add(x[c][g][t][d][tau] == 0)
    for c, d in product(C_list, D_list):
        for t1, t2, g in product(T_dict[c], T_dict[c], [0, gamma_dict[c]]):
            solver.Add(x[c][g][t1][d][t2] == 0)

    # TODO: these are constraints, not variable fixing
    for c, d in product(C_list, D_prime_list):
        # fix: is T (no subscript) in the paper
        for t1, t2 in product(T_dict[c] + [tau], T_prime_dict[c]):
            solver.Add(x[c][0][t1][d][t2] == 0)
    for c, d in product(C_list, D_prime_list):
        # fix: is T (no subscript) in the paper
        for t1, t2 in product(T_prime_dict[c], [sigma] + T_dict[c]):
            solver.Add(x[c][gamma_dict[c]][t1][d][t2] == 0)

    # TODO: these are constraints, not variable fixing
    # d_prime_dict[c] is the day of delivery;
    # d_prime_dict[c]-1 is the last day of growth for g=gamma_dict[c]
    d_prime_dict = p_ins.make_last_demand_date_dict()
    for c in C_list:
        for t1, t2 in product(T_dict[c], T_dict[c]):
            # added: if no demand, all x for the crop should be 0
            if c not in d_prime_dict:
                for d in D_prime_list:
                    for g in range(1, gamma_dict[c] + 1):
                        solver.Add(x[c][g][t1][d][t2] == 0)
            else:
                for d in range(
                    d_prime_dict[c] - gamma_dict[c] + 1, d_prime_dict[c] + 1
                ):
                    for g in range(1, gamma_dict[c] - (d_prime_dict[c] - d) + 1):
                        solver.Add(x[c][g][t1][d][t2] == 0)

    # TODO: these are constraints, not variable fixing
    for c in C_list:
        for t1, t2 in product(T_prime_dict[c], T_prime_dict[c]):
            for d in t_list_before(gamma_dict[c] - 1, D_prime_list):
                for g in range(d + 1, gamma_dict[c] + 1):
                    solver.Add(x[c][g][t1][d][t2] == 0)

    for c in C_list:
        for t, d in product(T_dict[c], range(gamma_dict[c] + 1, d_bar + 1)):
            solver.Add(x[c][0][sigma][d - gamma_dict[c] - 1][t] <= p_dict[c][d])
    for c in C_list:
        for t, d in product(T_dict[c], D_list + [d_bar]):
            solver.Add(x[c][gamma_dict[c]][t][d - 1][tau] <= p_dict[c][d])

    # d -> k -> the number of units of crop c that requires configuration k on day d
    eta_dict: dict[int, dict[str, int]] = {d: {k: 0 for k in K_list} for d in D_list}
    for d, k in product(D_list, K_list):
        for c in C_list:
            # fix: is range(d + 1, d + gamma_dict[c])  in paper
            for d_prime in range(d + 1, min(d + gamma_dict[c], d_bar) + 1):
                if k == k_dict[c][gamma_dict[c] - (d_prime - d - 1)]:
                    # fix: is p_dict[c][d] in paper
                    eta_dict[d][k] += p_dict[c][d_prime]
    for d, k in product(D_list, K_list):
        if eta_dict[d][k] == 0:
            for t in T_list:
                solver.Add(y[t][d][k] == 0)

    # Valid inequalities
    q_bar_dict = {k: max(q_dict[t][k] for t in T_list) for k in K_list}
    for k in K_list:
        if q_bar_dict[k] == 0:
            continue
        for d in D_list:
            if eta_dict[d][k] == 0:
                continue
            rhs = math.ceil(eta_dict[d][k] / q_bar_dict[k])
            solver.Add(SumArray(y[t][d][k] for t in T_list) >= rhs)

    # solve
    solver.set_time_limit(timelimit * 1000)
    status = solver.Solve()
    wall_sec = solver.wall_time() / 1000

    solution = VariablesObj2()
    solution.wall_sec = wall_sec

    _str: str = "solver status log placeholder"
    if status == Solver.OPTIMAL:
        _str = (
            f"*Optimal* objective value = {solver.Objective().Value()}\t"
            + f"found in {wall_sec:.3f} seconds"
        )
        solution.found_feasible = True
        solution.is_optimal = True
    elif status == Solver.FEASIBLE:
        _str = (
            f"An incumbent objective value = {solver.Objective().Value()}\t"
            + f"and best bound = {solver.Objective().BestBound()}\t"
            + f"found in {wall_sec:.3f} seconds"
        )
        solution.found_feasible = True
    elif status == Solver.INFEASIBLE:
        _str = f"The problem is found to be infeasible in {wall_sec:.3f} seconds"
        solution.is_infeasible = True
    elif status == Solver.UNBOUNDED:
        _str = f"The problem is found to be unbounded in {wall_sec:.3f} seconds"
        solution.is_unbounded = True
    elif status == Solver.NOT_SOLVED:
        _str = f"No solution found in {wall_sec:.3f} seconds"
        solution.not_solved = True
    logging.info(_str)

    if solution.found_feasible:
        solution.obj_val = solver.Objective().Value()
        solution.obj_bound = solver.Objective().BestBound()
        _str = f"Problem solved in {solver.iterations()} iterations"
        _str += f" & {solver.nodes()} branch-and-bound nodes"
        logging.info(_str)
        solution.x = {
            c: {
                g: {
                    t1: {
                        d: {
                            t2: x[c][g][t1][d][t2].solution_value()
                            for t2 in T_prime_dict[c]
                        }
                        for d in D_prime_list
                    }
                    for t1 in T_prime_dict[c]
                }
                for g in t_list_before(gamma_dict[c], D_prime_list)
            }
            for c in C_list
        }
        solution.y = {
            t: {d: {k: y[t][d][k].solution_value() for k in K_list} for d in D_list}
            for t in T_list
        }
        solution.w = {
            t: {
                d: {k: w[t][d][k].solution_value() for k in K_list}
                for d in range(1, d_bar - 1)
            }
            for t in T_list
        }

    return solution


def solve_santini_21_milp_t2_obj4(
    p_ins: ProbInsS21T2, solver_name: str, timelimit: int
) -> VariablesObj4:
    # from pprint import pprint

    # Indices
    C_list = p_ins.crop_id_list  # c\in C in paper
    # S_list = p_ins.shelf_id_list
    D_list = p_ins.t_idx_list[:-1]  # d\in D in paper; set of growth days
    d_bar = p_ins.t_idx_list[-1]  # d_bar in paper
    D_prime_list = [0] + D_list  # extended time horizon
    # day 0 is the earliest day seeds can be planted, day 1 is the earliest day of growth counted
    K_list = p_ins.config_id_list  # k\in K in paper
    T_list = p_ins.shelf_type_list  # t\in T in paper

    # print(C_list)
    # print(T_list)
    # print(K_list)
    # print(demand_D_list)
    # print(growth_D_list)
    # print(D_prime_list)

    # Parameters
    # c -> d -> demand quantity
    p_dict = p_ins.make_demand_dict()
    # c -> s -> the crop can grow on the shelf type
    # delta_dict = p_ins.crop_shelf_compatible
    # c -> list of compatible s
    # S_dict = {c: [s for s in S_list if delta_dict[c][s]] for c in C_list}
    # c -> the number of growth days
    gamma_dict = p_ins.crop_growth_days
    # c -> g (growth day) -> configuration required
    k_dict = p_ins.crop_growth_day_config_dict
    # s -> t -> capacity
    q_dict = p_ins.capacity
    # c -> t -> the crop can grow on the shelf type
    delta_dict = p_ins.crop_shelf_type_compatible
    # c -> list of compatible t
    T_dict = {c: [t for t in T_list if delta_dict[c][t]] for c in C_list}
    # t -> number of shelves
    n_dict = p_ins.num_shelves

    # seed vault index
    sigma = "SeedVault"
    # produce stage index
    tau = "Produce"
    # S_prime_list = [sigma, tau] + S_list
    # S_prime_dict = {c: [sigma, tau] + S_dict[c] for c in C_list}
    T_prime_list = [sigma, tau] + T_list
    for c in C_list:
        delta_dict[c][sigma] = True
        delta_dict[c][tau] = True
    T_prime_dict = {c: [sigma, tau] + T_dict[c] for c in C_list}

    # pprint(p_dict)
    # pprint(delta_dict)
    # pprint(T_dict)
    # pprint(n_dict)
    # pprint(gamma_dict)
    # pprint(k_dict)
    # pprint(q_dict)
    # pprint(T_prime_list)
    # pprint(T_prime_dict)

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
                    for d in D_prime_list
                }
                for t1 in T_prime_dict[c]
            }
            for g in range(0, gamma_dict[c] + 1)
        }
        for c in C_list
    }
    # the number of shelves of type t with configuration k on day d
    # TODO: shelf type에서 쓰이는 configuration list의 dictionary를 쓰자
    y = {
        t: {
            d: {k: solver.IntVar(lb=0, ub=infty, name=f"y_{t},{d},{k}") for k in K_list}
            for d in D_list
        }
        for t in T_list
    }
    # constraint 1: demand satisfaction
    # note: the due date is a day after the last growth day
    for c, d in product(C_list, D_list + [d_bar]):
        solver.Add(
            SumArray(x[c][gamma_dict[c]][t][d - 1][tau] for t in T_dict[c])
            == p_dict[c][d]
        )

    # constraint 2: capacity constraints
    for t1, d, k in product(T_list, D_list, K_list):
        var_list = list()
        for t2 in T_prime_list:
            for c in C_list:
                if delta_dict[c][t1] and delta_dict[c][t2]:
                    for g in range(1, gamma_dict[c] + 1):
                        if k_dict[c][g] == k:
                            var_list.append(x[c][g][t1][d][t2])

        lhs = SumArray(var_list)
        solver.Add(lhs <= q_dict[t1][k] * y[t1][d][k])

    # constrain 3: planting constraints
    for c in C_list:
        for d in range(gamma_dict[c] + 1, d_bar + 1):
            solver.Add(
                SumArray(x[c][0][sigma][d - gamma_dict[c] - 1][t] for t in T_dict[c])
                == p_dict[c][d]
            )

    # constraint 4: flow-balance constraints
    for c in C_list:
        # fix: is gamma_dict[c]+1 in the paper
        for g in range(0, gamma_dict[c]):
            for t1 in T_dict[c]:
                for d in D_list:
                    lhs = SumArray(x[c][g][t2][d - 1][t1] for t2 in T_prime_dict[c])
                    rhs = SumArray(x[c][g + 1][t1][d][t2] for t2 in T_prime_dict[c])
                    solver.Add(lhs == rhs)

    # constraint 5: configuration constraints
    # TODO: better be equality
    for t, d in product(T_list, D_list):
        solver.Add(SumArray(y[t][d][k] for k in K_list) <= n_dict[t])

    # Objective 4: minimize # shelves used
    cfg_for_unused_shelf_id = "cf_unused"
    for t in T_list:
        q_dict[t][cfg_for_unused_shelf_id] = 0

    # t -> # shelves of type t used
    nu = {t: solver.IntVar(lb=0, ub=infty, name=f"nu_{t}") for t in T_list}
    obj_func: Objective = solver.Objective()
    for t in T_list:
        obj_func.SetCoefficient(nu[t], 1)
        for d in D_list:
            solver.Add(nu[t] >= SumArray(y[t][d][k] for k in K_list))
    obj_func.SetMinimization()

    # Variable fixing
    # TODO: these are constraints, not variable fixing
    for c, d in product(C_list, D_prime_list):
        for t, g in product(T_prime_dict[c], range(0, gamma_dict[c] + 1)):
            solver.Add(x[c][g][t][d][sigma] == 0)
            solver.Add(x[c][g][tau][d][t] == 0)
    for c, d in product(C_list, D_prime_list):
        for g in range(0, gamma_dict[c] + 1):
            solver.Add(x[c][g][sigma][d][tau] == 0)

    # TODO: these are constraints, not variable fixing
    for c, d in product(C_list, D_prime_list):
        for t, g in product(T_prime_dict[c], range(1, gamma_dict[c] + 1)):
            solver.Add(x[c][g][sigma][d][t] == 0)
    for c, d in product(C_list, D_prime_list):
        for t, g in product(T_prime_dict[c], range(0, gamma_dict[c])):
            solver.Add(x[c][g][t][d][tau] == 0)
    for c, d in product(C_list, D_list):
        for t1, t2, g in product(T_dict[c], T_dict[c], [0, gamma_dict[c]]):
            solver.Add(x[c][g][t1][d][t2] == 0)

    # TODO: these are constraints, not variable fixing
    for c, d in product(C_list, D_prime_list):
        # fix: is T (no subscript) in the paper
        for t1, t2 in product(T_dict[c] + [tau], T_prime_dict[c]):
            solver.Add(x[c][0][t1][d][t2] == 0)
    for c, d in product(C_list, D_prime_list):
        # fix: is T (no subscript) in the paper
        for t1, t2 in product(T_prime_dict[c], [sigma] + T_dict[c]):
            solver.Add(x[c][gamma_dict[c]][t1][d][t2] == 0)

    # TODO: these are constraints, not variable fixing
    # d_prime_dict[c] is the day of delivery;
    # d_prime_dict[c]-1 is the last day of growth for g=gamma_dict[c]
    d_prime_dict = p_ins.make_last_demand_date_dict()
    for c in C_list:
        for t1, t2 in product(T_dict[c], T_dict[c]):
            # added: if no demand, all x for the crop should be 0
            if c not in d_prime_dict:
                for d in D_prime_list:
                    for g in range(1, gamma_dict[c] + 1):
                        solver.Add(x[c][g][t1][d][t2] == 0)
            else:
                for d in range(
                    d_prime_dict[c] - gamma_dict[c] + 1, d_prime_dict[c] + 1
                ):
                    for g in range(1, gamma_dict[c] - (d_prime_dict[c] - d) + 1):
                        solver.Add(x[c][g][t1][d][t2] == 0)

    # TODO: these are constraints, not variable fixing
    for c in C_list:
        for t1, t2 in product(T_prime_dict[c], T_prime_dict[c]):
            for d in t_list_before(gamma_dict[c] - 1, D_prime_list):
                for g in range(d + 1, gamma_dict[c] + 1):
                    solver.Add(x[c][g][t1][d][t2] == 0)

    for c in C_list:
        for t, d in product(T_dict[c], range(gamma_dict[c] + 1, d_bar + 1)):
            solver.Add(x[c][0][sigma][d - gamma_dict[c] - 1][t] <= p_dict[c][d])
    for c in C_list:
        for t, d in product(T_dict[c], D_list + [d_bar]):
            solver.Add(x[c][gamma_dict[c]][t][d - 1][tau] <= p_dict[c][d])

    # d -> k -> the number of units of crop c that requires configuration k on day d
    eta_dict: dict[int, dict[str, int]] = {d: {k: 0 for k in K_list} for d in D_list}
    for d, k in product(D_list, K_list):
        for c in C_list:
            # fix: is range(d + 1, d + gamma_dict[c])  in paper
            for d_prime in range(d + 1, min(d + gamma_dict[c], d_bar) + 1):
                if k == k_dict[c][gamma_dict[c] - (d_prime - d - 1)]:
                    # fix: is p_dict[c][d] in paper
                    eta_dict[d][k] += p_dict[c][d_prime]
    for d, k in product(D_list, K_list):
        if eta_dict[d][k] == 0:
            for t in T_list:
                solver.Add(y[t][d][k] == 0)

    # Valid inequalities
    q_bar_dict = {k: max(q_dict[t][k] for t in T_list) for k in K_list}
    for k in K_list:
        if q_bar_dict[k] == 0:
            continue
        for d in D_list:
            if eta_dict[d][k] == 0:
                continue
            rhs = math.ceil(eta_dict[d][k] / q_bar_dict[k])
            solver.Add(SumArray(y[t][d][k] for t in T_list) >= rhs)

    # solve
    solver.set_time_limit(timelimit * 1000)
    status = solver.Solve()
    wall_sec = solver.wall_time() / 1000

    solution = VariablesObj4()
    solution.wall_sec = wall_sec

    _str: str = "solver status log placeholder"
    if status == Solver.OPTIMAL:
        _str = (
            f"*Optimal* objective value = {solver.Objective().Value()}\t"
            + f"found in {wall_sec:.3f} seconds"
        )
        solution.found_feasible = True
        solution.is_optimal = True
    elif status == Solver.FEASIBLE:
        _str = (
            f"An incumbent objective value = {solver.Objective().Value()}\t"
            + f"and best bound = {solver.Objective().BestBound()}\t"
            + f"found in {wall_sec:.3f} seconds"
        )
        solution.found_feasible = True
    elif status == Solver.INFEASIBLE:
        _str = f"The problem is found to be infeasible in {wall_sec:.3f} seconds"
        solution.is_infeasible = True
    elif status == Solver.UNBOUNDED:
        _str = f"The problem is found to be unbounded in {wall_sec:.3f} seconds"
        solution.is_unbounded = True
    elif status == Solver.NOT_SOLVED:
        _str = f"No solution found in {wall_sec:.3f} seconds"
        solution.not_solved = True
    logging.info(_str)

    if solution.found_feasible:
        solution.obj_val = solver.Objective().Value()
        solution.obj_bound = solver.Objective().BestBound()
        _str = f"Problem solved in {solver.iterations()} iterations"
        _str += f" & {solver.nodes()} branch-and-bound nodes"
        logging.info(_str)
        solution.x = {
            c: {
                g: {
                    t1: {
                        d: {
                            t2: x[c][g][t1][d][t2].solution_value()
                            for t2 in T_prime_dict[c]
                        }
                        for d in D_prime_list
                    }
                    for t1 in T_prime_dict[c]
                }
                for g in t_list_before(gamma_dict[c], D_prime_list)
            }
            for c in C_list
        }
        solution.y = {
            t: {d: {k: y[t][d][k].solution_value() for k in K_list} for d in D_list}
            for t in T_list
        }
        solution.nu = {t: nu[t].solution_value() for t in T_list}

    return solution
