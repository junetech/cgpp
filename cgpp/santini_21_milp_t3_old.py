import logging
from itertools import product

from input_class import ProbInsS21T3
from ortools.linear_solver.linear_solver_natural_api import SumArray
from ortools.linear_solver.pywraplp import Solver
from output_class import VariablesObj3


def t_list_before(t: int, t_list: list[int]) -> list[int]:
    return [u for u in t_list if u <= t]


def t_list_after(t: int, t_list: list[int]) -> list[int]:
    return [u for u in t_list if u >= t]


def solve_santini_21_milp_t3(p_ins: ProbInsS21T3, solver_name: str, timelimit: int):
    # from pprint import pprint

    # Indices
    C_list = p_ins.crop_id_list  # c\in C in paper
    T_list = p_ins.shelf_type_list  # t\in T in paper
    K_list = p_ins.config_id_list  # k\in K in paper
    demand_D_list = p_ins.t_idx_list  # set of demand days
    growth_D_list = p_ins.t_idx_list[:-1]  # d\in D in paper; set of growth days
    D_prime_list = [0] + growth_D_list  # extended time horizon
    # day 0 is the earliest day seeds can be planted, day 1 is the earliest day of growth counted

    # print(C_list)
    # print(T_list)
    # print(K_list)
    # print(demand_D_list)
    # print(growth_D_list)
    # print(D_prime_list)

    # Parameters
    # c -> d -> demand quantity
    p_dict = p_ins.make_demand_dict()
    # c -> t -> the crop can grow on the shelf type
    delta_dict = p_ins.crop_shelf_type_compatible
    # c -> list of compatible t
    T_dict = {c: [t for t in T_list if delta_dict[c][t]] for c in C_list}
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
    for c in C_list:
        delta_dict[c][sigma] = True
        delta_dict[c][tau] = True
    T_prime_dict = {c: [sigma] + T_dict[c] + [tau] for c in C_list}

    # pprint(p_dict)
    # pprint(delta_dict)
    # pprint(T_dict)
    # pprint(n_dict)
    # pprint(gamma_dict)
    # pprint(k_dict)
    # pprint(q_dict)
    # pprint(t_prime_list)
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
            for g in t_list_before(gamma_dict[c], D_prime_list)
        }
        for c in C_list
    }
    # the number of shelves of type t with configuration k on day d
    # TODO: shelf type에서 쓰이는 configuration list의 dictionary를 쓰자
    y = {
        t: {
            d: {k: solver.IntVar(lb=0, ub=infty, name=f"y_{t},{d},{k}") for k in K_list}
            for d in D_prime_list
        }
        for t in T_list
    }

    # Objective 3: minimize unmet demand
    omega_dict = p_ins.missed_demand_penalty
    # pprint(omega_dict)
    # c -> d -> the amount of unmet demand of crop c on day d
    u = {
        c: {d: solver.IntVar(lb=0, ub=infty, name=f"u_{c},{d}") for d in demand_D_list}
        for c in C_list
    }
    obj_func = solver.Objective()
    for c, d in product(C_list, demand_D_list):
        if d not in omega_dict[c]:
            continue
        obj_func.SetCoefficient(u[c][d], omega_dict[c][d])
    obj_func.SetMinimization()

    # constraint 9 instead of constraint 1
    # note: the due date is a day after the last growth day
    for c, d in product(C_list, demand_D_list):
        solver.Add(
            SumArray(x[c][gamma_dict[c]][t][d - 1][tau] for t in T_dict[c]) + u[c][d]
            == p_dict[c][d]
        )

    # constraint 10 instead of constrain 3
    for c in C_list:
        for d in t_list_after(gamma_dict[c] + 1, demand_D_list):
            solver.Add(
                SumArray(x[c][0][sigma][d - gamma_dict[c] - 1][t] for t in T_dict[c])
                + u[c][d]
                == p_dict[c][d]
            )

    # constraint 2: capacity constraints
    for t1, d, k in product(T_list, growth_D_list, K_list):
        var_list = list()
        for t2 in t_prime_list:
            for c in C_list:
                if delta_dict[c][t1] and delta_dict[c][t2]:
                    for g in t_list_before(gamma_dict[c], growth_D_list):
                        if k_dict[c][g] == k:
                            var_list.append(x[c][g][t1][d][t2])

        lhs = SumArray(var_list)
        solver.Add(lhs <= q_dict[t1][k] * y[t1][d][k])

    # constraint 4: flow-balance constraints
    for c in C_list:
        # fix
        for t1, d in product(T_dict[c], growth_D_list):
            lhs = x[c][0][sigma][d - 1][t1]
            rhs = SumArray(x[c][1][t1][d][t2] for t2 in T_dict[c])
            solver.Add(lhs == rhs)
            for g in t_list_before(gamma_dict[c] - 1, growth_D_list[:-1]):
                lhs = SumArray(x[c][g][t2][d - 1][t1] for t2 in T_prime_dict[c])
                rhs = SumArray(x[c][g + 1][t1][d][t2] for t2 in T_prime_dict[c])
                solver.Add(lhs == rhs)
            lhs = SumArray(x[c][gamma_dict[c] - 1][t2][d - 1][t1] for t2 in T_dict[c])
            rhs = x[c][gamma_dict[c]][t1][d][tau]
            solver.Add(lhs == rhs)

    # constraint 5: configuration constraints
    for t, d in product(T_list, growth_D_list):
        solver.Add(SumArray(y[t][d][k] for k in K_list) == n_dict[t])

    # Variable fixing
    for c, d in product(C_list, D_prime_list):
        for t, g in product(
            T_prime_dict[c], t_list_before(gamma_dict[c], D_prime_list)
        ):
            solver.Add(x[c][g][t][d][sigma] == 0)
            solver.Add(x[c][g][tau][d][t] == 0)
    for c, d in product(C_list, D_prime_list):
        for g in t_list_before(gamma_dict[c], D_prime_list):
            solver.Add(x[c][g][sigma][d][tau] == 0)

    for c, d in product(C_list, D_prime_list):
        for t, g in product(
            T_prime_dict[c], t_list_before(gamma_dict[c], growth_D_list)
        ):
            solver.Add(x[c][g][sigma][d][t] == 0)
    for c, d in product(C_list, D_prime_list):
        for t, g in product(
            T_prime_dict[c], t_list_before(gamma_dict[c] - 1, D_prime_list)
        ):
            solver.Add(x[c][g][t][d][tau] == 0)
    for c, d in product(C_list, growth_D_list):
        for t1, t2, g in product(
            T_dict[c], T_dict[c], t_list_before(gamma_dict[c], D_prime_list)
        ):
            solver.Add(x[c][g][t1][d][t2] == 0)

    for c, d in product(C_list, D_prime_list):
        for t1, t2 in product(T_dict[c] + [tau], T_prime_dict[c]):
            solver.Add(x[c][0][t1][d][t2] == 0)
        for t1, t2 in product(T_prime_dict[c], [sigma] + T_dict[c]):
            solver.Add(x[c][gamma_dict[c]][t1][d][t2] == 0)

    obj_val: float = 0.0
    solution = VariablesObj3()
    # solve
    solver.set_time_limit(timelimit * 1000)
    status = solver.Solve()
    wall_sec = solver.wall_time() / 1000

    _str: str = "solver status log placeholder"
    if status == Solver.OPTIMAL:
        _str = (
            f"*Optimal* objective value = {solver.Objective().Value()}\t"
            + f"found in {wall_sec:.3f} seconds"
        )
    elif status == Solver.FEASIBLE:
        _str = (
            f"An incumbent objective value = {solver.Objective().Value()}\t"
            + f"and best bound = {solver.Objective().BestBound()}\t"
            + f"found in {wall_sec:.3f} seconds"
        )
    elif status == Solver.INFEASIBLE:
        _str = f"The problem is found to be infeasible in {wall_sec:.3f} seconds"
    elif status == Solver.UNBOUNDED:
        _str = f"The problem is found to be unbounded in {wall_sec:.3f} seconds"
    elif status == Solver.NOT_SOLVED:
        _str = f"No solution found in {wall_sec:.3f} seconds"
    logging.info(_str)

    if status == Solver.OPTIMAL or status == Solver.FEASIBLE:
        obj_val = solver.Objective().Value()
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
            t: {
                d: {k: y[t][d][k].solution_value() for k in K_list}
                for d in D_prime_list
            }
            for t in T_list
        }
        solution.u = {
            c: {d: u[c][d].solution_value() for d in demand_D_list} for c in C_list
        }

    return obj_val, solution
