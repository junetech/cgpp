from typing import Any


def shelf_id_str(idx: int) -> str:
    return f"s_{idx}"


def shelf_type_str(idx: int) -> str:
    return f"st_{idx}"


def config_id_str(idx: int) -> str:
    return f"cfg_{idx}"


class ProbInsS21:
    # instance name information
    n_shelves: int
    n_crops: int
    crop_id_string: str  # actual crops in the instance (named from A to F)
    n_days: int  # the number of days
    demand_mult: float  # demand multiplier
    id: int  # instance id
    model_type: str

    # instance data
    # crop ID -> growth days
    crop_growth_days: dict[str, int]
    n_configurations: int
    # TODO: create configuration ID list
    # crop ID -> list of required configuration for each growth day
    crop_growth_day_config: dict[str, list[str]]
    # shelf ID (type) -> configuration ID -> capaciity
    capacity: dict[str, dict[str, int]]
    # crop ID -> day -> demand
    demand: dict[str, dict[int, int]]

    @property
    def problem_name(self):
        _str = f"{self.n_shelves}-{self.n_crops}-{self.crop_id_string}-"
        _str += f"{self.n_days}-{self.demand_mult}-{self.id}-{self.model_type}"
        return _str

    def info_from_filename(
        self,
        n_shelves: int,
        n_crops: int,
        crop_id_string: str,
        n_days: int,
        demand_mult: float,
        id: int,
        model_type: str,
    ):
        self.n_shelves = n_shelves
        self.n_crops = n_crops
        self.crop_id_string = crop_id_string
        self.n_days = n_days
        self.demand_mult = demand_mult
        self.id = id
        self.model_type = model_type

    def create_crop_id(self):
        self.crop_id_list: list[str] = [char for char in self.crop_id_string]

    def create_t_list(self):
        self.t_list: list[int] = [t for t in range(self.n_days)]

    def create_config_id(self):
        self.config_id_list: list[str] = [
            config_id_str(idx) for idx in range(self.n_configurations)
        ]

    def make_val_dict(self) -> dict[str, Any]:
        return_dict: dict[str, Any] = dict()
        return_dict["n_shelves"] = self.n_shelves
        return_dict["n_crops"] = self.n_crops
        return_dict["crop_id_string"] = self.crop_id_string
        return_dict["n_days"] = self.n_days
        return_dict["demand_mult"] = self.demand_mult
        return_dict["id"] = self.id
        return_dict["model_type"] = self.model_type
        return return_dict


class ProbInsS21T1(ProbInsS21):
    # crop ID -> shelf ID -> compatibility
    crop_shelf_compatible: dict[str, dict[str, bool]]

    def create_shelf_id(self):
        self.shelf_id_list: list[str] = [
            shelf_id_str(idx) for idx in range(self.n_shelves)
        ]

    def make_val_dict(self) -> dict[str, Any]:
        return_dict: dict[str, Any] = super().make_val_dict()

        return_dict["n_days"] = self.n_days
        return_dict["n_shelves"] = self.n_shelves
        return_dict["n_crops"] = self.n_crops
        return_dict["crop_shelf_compatible"] = self.crop_shelf_compatible
        return_dict["crop_growth_days"] = self.crop_growth_days
        return_dict["n_configurations"] = self.n_configurations
        return_dict["crop_growth_day_config"] = self.crop_growth_day_config
        return_dict["capacity"] = self.capacity
        return_dict["demand"] = self.demand

        return return_dict

    def make_json_dump_dict(self) -> dict[str, Any]:
        return_dict: dict[str, Any] = self.make_val_dict()
        return_dict["crop_id_list"] = self.crop_id_list
        return_dict["shelf_id_list"] = self.shelf_id_list
        return_dict["config_id_list"] = self.config_id_list

        return return_dict


class ProbInsS21T2(ProbInsS21):
    n_shelf_types: int
    # TODO: create shelf type list
    # shelf ID -> the number of shelves
    num_shelves: dict[str, int]
    # crop ID -> shelf type -> compatibility
    crop_shelf_type_compatible: dict[str, dict[str, bool]]

    def create_shelf_type(self):
        self.shelf_type_list: list[str] = [
            shelf_type_str(idx) for idx in range(self.n_shelf_types)
        ]

    def check_integrity(self):
        if self.n_shelves != sum(self.num_shelves.values()):
            _str = f"The number of all shelves {self.n_shelves} != the sum of the"
            _str += f" number of all types of shelves {sum(self.num_shelves.values())}"
            raise ValueError(_str)

    def create_shelf_id(self):
        self.shelf_id_dict: dict[str, list[str]] = {
            shelf_type: list() for shelf_type in self.shelf_type_list
        }
        idx = 0
        for shelf_type in self.shelf_type_list:
            shelf_type_count = self.num_shelves[shelf_type]
            self.shelf_id_dict[shelf_type].extend(
                [shelf_id_str(n + idx) for n in range(shelf_type_count)]
            )
            idx += shelf_type_count

    @property
    def shelf_id_list(self):
        return [
            shelf_id
            for shelf_type in self.shelf_type_list
            for shelf_id in self.shelf_id_dict[shelf_type]
        ]

    def make_val_dict(self) -> dict[str, Any]:
        return_dict: dict[str, Any] = super().make_val_dict()

        return_dict["n_days"] = self.n_days
        return_dict["n_shelf_types"] = self.n_shelf_types
        return_dict["n_crops"] = self.n_crops
        return_dict["num_shelves"] = self.num_shelves
        return_dict["crop_shelf_type_compatible"] = self.crop_shelf_type_compatible
        return_dict["crop_growth_days"] = self.crop_growth_days
        return_dict["n_configurations"] = self.n_configurations
        return_dict["crop_growth_day_config"] = self.crop_growth_day_config
        return_dict["capacity"] = self.capacity
        return_dict["demand"] = self.demand

        return return_dict

    def make_json_dump_dict(self) -> dict[str, Any]:
        return_dict: dict[str, Any] = self.make_val_dict()
        return_dict["crop_id_list"] = self.crop_id_list
        return_dict["shelf_type_list"] = self.shelf_type_list
        return_dict["shelf_id_dict"] = self.shelf_id_dict
        return_dict["config_id_list"] = self.config_id_list

        return return_dict


class ProbInsS21T3(ProbInsS21T2):
    # crop ID -> day -> missed demand penalty
    missed_demand_penalty: dict[str, dict[int, int]]

    def make_val_dict(self) -> dict[str, Any]:
        return_dict: dict[str, Any] = super().make_val_dict()
        return_dict["missed_demand_penalty"] = self.missed_demand_penalty

        return return_dict

    def make_json_dump_dict(self) -> dict[str, Any]:
        return_dict: dict[str, Any] = self.make_val_dict()
        return_dict["crop_id_list"] = self.crop_id_list
        return_dict["shelf_type_list"] = self.shelf_type_list
        return_dict["shelf_id_dict"] = self.shelf_id_dict
        return_dict["config_id_list"] = self.config_id_list

        return return_dict
