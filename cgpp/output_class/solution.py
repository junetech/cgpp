class Variables:
    # crop -> time -> shelf type -> time -> shelf type -> the number of units of crop
    x: dict[str, dict[int, dict[str, dict[int, dict[str, int]]]]]
    # shelf type -> time -> configuration -> the number of shelves
    y: dict[str, dict[int, dict[str, int]]]


class VariablesObj2(Variables):
    w: dict


class VariablesObj3(Variables):
    # crop -> time -> the number of unmet demand units of crop
    u: dict[str, dict[int, int]]


class VariablesObj4(Variables):
    v: dict
