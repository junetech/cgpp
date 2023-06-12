from dataclasses import dataclass
from typing import Any


@dataclass(kw_only=True)
class ProbInsS21:
    # instance name information
    n_shelves: int
    n_crops: int
    crop_id_string: str  # actual crops in the instance (named from A to F)
    n_days: int  # the number of days
    demand_mult: float  # demand multiplier
    id: int  # instance id
    model_type: str

    cabinet: str

    crop_id_list: list[str]
    config_id_list: list[str]

    # instance data
    # crop ID -> growth days
    crop_growth_days: dict[str, int]
    n_configurations: int
    # crop ID -> list of required configuration for each growth day
    crop_growth_day_config: dict[str, list[str]]
    # shelf ID (type) -> configuration ID -> capacity
    capacity: dict[str, dict[str, int]]
    # crop ID -> day -> demand
    demand: dict[str, dict[int, int]]

    @property
    def problem_name(self):
        _str = f"{self.n_shelves}-{self.n_crops}-{self.crop_id_string}-"
        _str += f"{self.n_days}-{self.demand_mult}-{self.id}"
        return _str

    def set_prob_obj_name(self, obj_idx: int):
        self.prob_obj_name: str = f"{self.problem_name}-o{obj_idx}"

    def get_prob_obj_name(self) -> str:
        return self.prob_obj_name

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

    def create_t_idx_list(self):
        self.t_idx_list: list[int] = [t + 1 for t in range(self.n_days)]
        # crop id -> growth day -> configuration required
        self.crop_growth_day_config_dict: dict[str, dict[int, str]] = {
            crop_id: {
                idx + 1: val
                for idx, val in enumerate(self.crop_growth_day_config[crop_id])
            }
            for crop_id in self.crop_id_list
        }
        self.demand = {
            crop_id: {int(t_idx): val for t_idx, val in t_idx_dict.items() if val > 0}
            for crop_id, t_idx_dict in self.demand.items()
        }

    def make_val_dict(self) -> dict[str, Any]:
        return_dict: dict[str, Any] = dict()
        return_dict["n_shelves"] = self.n_shelves
        return_dict["n_crops"] = self.n_crops
        return_dict["crop_id_string"] = self.crop_id_string
        return_dict["n_days"] = self.n_days
        return_dict["demand_mult"] = self.demand_mult
        return_dict["id"] = self.id
        return_dict["model_type"] = self.model_type
        return_dict["cabinet"] = self.cabinet
        return return_dict

    def make_demand_dict(self) -> dict[str, dict[int, int]]:
        return_dict = {
            crop_id: {t_idx: 0 for t_idx in self.t_idx_list}
            for crop_id in self.crop_id_list
        }
        for crop_id, t_idx_dict in self.demand.items():
            for t_idx, demand in t_idx_dict.items():
                return_dict[crop_id][t_idx] = demand
        return return_dict

    def make_last_demand_date_dict(self) -> dict[str, int]:
        """

        Returns:
            dict[str, int]: crop ID -> last demand date
        """
        return_dict: dict[str, int] = dict()
        for crop_id, t_idx_dict in self.demand.items():
            if t_idx_dict:
                return_dict[crop_id] = max(t_idx_dict.keys())
        return return_dict


@dataclass(kw_only=True)
class ProbInsS21T1(ProbInsS21):
    shelf_id_list: list[str]
    # crop ID -> shelf ID -> compatibility
    crop_shelf_compatible: dict[str, dict[str, bool]]

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


@dataclass(kw_only=True)
class ProbInsS21T2(ProbInsS21):
    n_shelf_types: int
    shelf_type_list: list[str]
    # shelf type -> the number of shelves
    num_shelves: dict[str, int]
    shelf_id_dict: dict[str, list[str]]
    # crop ID -> shelf type -> compatibility
    crop_shelf_type_compatible: dict[str, dict[str, bool]]

    def check_integrity(self):
        if self.n_shelves != sum(self.num_shelves.values()):
            _str = f"The number of all shelves {self.n_shelves} != the sum of the"
            _str += f" number of all types of shelves {sum(self.num_shelves.values())}"
            raise ValueError(_str)

    @property
    def shelf_id_list(self) -> list[str]:
        return [
            shelf_id
            for shelf_type in self.shelf_type_list
            for shelf_id in self.shelf_id_dict[shelf_type]
        ]

    @property
    def crop_shelf_compatible(self) -> dict[str, dict[str, bool]]:
        return {
            crop_id: {
                shelf_id: shelf_type_dict[shelf_type]
                for shelf_type, shelf_id_list in self.shelf_id_dict.items()
                for shelf_id in shelf_id_list
            }
            for crop_id, shelf_type_dict in self.crop_shelf_type_compatible.items()
        }

    @property
    def shelf_id_capacity(self) -> dict[str, int]:
        return {shelf_id: 1 for shelf_id in self.shelf_id_list}

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


@dataclass(kw_only=True)
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

    def create_t_idx_list(self):
        super().create_t_idx_list()
        self.missed_demand_penalty = {
            crop_id: {int(t_idx): val for t_idx, val in t_idx_dict.items()}
            for crop_id, t_idx_dict in self.missed_demand_penalty.items()
        }


def from_t1_to_t2(p_ins_t1: ProbInsS21T1) -> ProbInsS21T2:
    n_shelves = p_ins_t1.n_shelves
    n_crops = p_ins_t1.n_crops
    crop_id_string = p_ins_t1.crop_id_string
    n_days = p_ins_t1.n_days
    demand_mult = p_ins_t1.demand_mult
    id = p_ins_t1.id
    model_type = p_ins_t1.model_type
    crop_id_list = p_ins_t1.crop_id_list
    config_id_list = p_ins_t1.config_id_list
    crop_growth_days_dict = p_ins_t1.crop_growth_days
    n_configurations = p_ins_t1.n_configurations
    cgdc_dict = p_ins_t1.crop_growth_day_config
    capa_dict = p_ins_t1.capacity
    demand_dict = p_ins_t1.demand
    n_shelf_types = 1
    shelf_type_list = p_ins_t1.shelf_id_list
    ns_dict = {shelf_id: 1 for shelf_id in p_ins_t1.shelf_id_list}
    shelf_id_dict = {shelf_id: [shelf_id] for shelf_id in p_ins_t1.shelf_id_list}
    cstc_dict = p_ins_t1.crop_shelf_compatible

    return ProbInsS21T2(
        n_shelves=n_shelves,
        n_crops=n_crops,
        crop_id_string=crop_id_string,
        n_days=n_days,
        demand_mult=demand_mult,
        id=id,
        model_type=model_type,
        crop_id_list=crop_id_list,
        config_id_list=config_id_list,
        crop_growth_days=crop_growth_days_dict,
        n_configurations=n_configurations,
        crop_growth_day_config=cgdc_dict,
        capacity=capa_dict,
        demand=demand_dict,
        n_shelf_types=n_shelf_types,
        shelf_type_list=shelf_type_list,
        num_shelves=ns_dict,
        shelf_id_dict=shelf_id_dict,
        crop_shelf_type_compatible=cstc_dict,
    )
