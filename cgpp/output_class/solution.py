class Variables:
    # crop -> time -> shelf type -> time -> shelf type -> the number of units of crop
    x: dict[str, dict[int, dict[str, dict[int, dict[str, int]]]]]
    # shelf type -> time -> configuration -> the number of shelves
    y: dict[str, dict[int, dict[str, int]]]

    wall_sec: float
    obj_val: float
    obj_bound: float

    def __init__(self):
        self.not_solved: bool = False
        self.is_unbounded: bool = False
        self.is_infeasible: bool = False
        self.found_feasible: bool = False
        self.is_optimal: bool = False


class VariablesObj2(Variables):
    w: dict


class VariablesObj3(Variables):
    # crop -> time -> the number of unmet demand units of crop
    u: dict[str, dict[int, int]]


class VariablesObj4(Variables):
    v: dict
