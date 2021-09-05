import os
from pydantic import BaseModel

base = os.getcwd()


class Inputs(BaseModel):
    city: str = "bogota"
    n_school_going_preschool: int = 150
    classroom_size_preschool: int = 15
    n_teachers_preschool: int = 5
    height_room_preschool: float = 3.1
    width_room_preschool: float = 7.0
    length_room_preschool: float = 7.0
    n_school_going_primary: int = 200
    classroom_size_primary: int = 35
    n_teachers_primary: int = 6
    height_room_primary: float = 3.1
    width_room_primary: float = 10.0
    length_room_primary: float = 10.0
    n_school_going_highschool: int = 200
    classroom_size_highschool: int = 35
    n_teachers_highschool: int = 7
    height_room_highschool: float = 3.1
    width_room_highschool: float = 10.0
    length_room_highschool: float = 10.0
    school_type: bool = False
    masks_type: str = "N95"
    ventilation_level: str = "alto"
    class_duration: int = 6


class Args:
    def __init__(self, params):
        inputs = Inputs(**params)
        ## USER
        self.city = inputs.city  # City

        # School parameters
        self.n_teachers_preschool = inputs.n_teachers_preschool
        self.n_school_going_preschool = inputs.n_school_going_preschool
        self.preschool_size = inputs.classroom_size_preschool
        self.height_room_preschool = inputs.height_room_preschool
        self.width_room_preschool = inputs.width_room_preschool
        self.length_room_preschool = inputs.length_room_preschool

        self.n_teachers_primary = inputs.n_teachers_primary
        self.n_school_going_primary = inputs.n_school_going_primary
        self.primary_size = inputs.classroom_size_primary
        self.height_room_primary = inputs.height_room_primary
        self.width_room_primary = inputs.width_room_primary
        self.length_room_primary = inputs.length_room_primary

        self.n_teachers_highschool = inputs.n_teachers_highschool
        self.n_school_going_highschool = inputs.n_school_going_highschool
        self.highschool_size = inputs.classroom_size_highschool
        self.height_room_highschool = inputs.height_room_highschool
        self.width_room_highschool = inputs.width_room_highschool
        self.length_room_highschool = inputs.length_room_highschool

        ventilation_level = inputs.ventilation_level

        if ventilation_level == "bajo":
            self.ventilation_out = 2  # Ventilation values (h-1) that define how much ventilated is a classroom [2-15]
        elif ventilation_level == "medio":
            self.ventilation_out = 7  # Ventilation values (h-1) that define how much ventilated is a classroom [2-15]
        elif ventilation_level == "alto":
            self.ventilation_out = 15  # Ventilation values (h-1) that define how much ventilated is a classroom [2-15]

        # Type of masks that individuals are using. Options are: cloth, surgical, N95
        self.masks_type = inputs.masks_type
        # Duration of event (i.e. classes/lectures) in hours over a day
        self.duration_event = inputs.class_duration

        ## DEFAULTS

        # Nodes
        self.population = 100  # Speficy the number of individials
        self.number_trials = 10  # Number of iterations per step

        self.intervention = 0.6  # Intervention efficiancy
        # Define the type of intervention [no_intervention,internvention,school_alternancy]
        self.intervention_type = "intervention"
        self.work_occupation = 0.6  # Percentage of occupation at workplaces over intervention
        self.school_occupation = 0.35  # Percentage of occupation at classrooms over intervention
        self.school_openings = 0  # Day of the simulation where schools are open
        self.fraction_people_masks = 1.0  # Fraction value of people wearing masks
        self.Tmax = 200  # Length of simulation (days)
        self.delta_t = 0.08  # Time steps

        self.preschool_mean = 9.4  # Preschool degree distribution (mean)
        self.preschool_std = 1.8  # preschool degree distribution (standard deviation)
        self.preschool_r = 1  # Correlation in preschool layer
        self.primary_mean = 9.4  # Primary degree distribution (mean)
        self.primary_std = 1.8  # Primary degree distribution (standard deviation)
        self.primary_r = 1  # Correlation in primary layer
        self.highschool_mean = 9.4  # Highschool degree distribution (mean)
        self.highschool_std = 1.8  # Highschool degree distribution (standard deviation)
        self.highschool_r = 1  # Correlation in highschool layer
        self.work_mean = 14.4 / 3  # Work degree distribution (mean)
        self.work_std = 6.2 / 3  # Work degree distribution (standard deviation)
        self.work_size = 10  # Approximation of a work place size
        self.work_r = 1  # Correlation in work layer
        self.community_mean = 4.3 / 2  # Community degree distribution (mean)
        self.community_std = 1.9 / 2  # Community degree distribution (standard deviation)
        self.community_n = 1  # Number of community
        self.community_r = 0  # Correlation in community layer

        ############ Paths ############
        self.figures_path = "".join([base, "/figures_new"])  # Directory to save figures
        self.results_path = "".join([base, "/results"])  # Directory to save results .csv files
        self.params_data_path = "".join([base, "/data/params.csv"])  # Path to ages parameters data
        self.ages_data_path = "".join([base, "/data/BogotaAgeStructure.txt"])  # Path to ages distgribution data
        self.houses_data_path = "".join([base, "/data/HousesCensusBOG.CSV"])  # Path to city census data
        # Path to teachers level distribution data
        self.teachers_data_path = "".join([base, "/data/teachers/teacher_level_distribution_bogota.csv"])
