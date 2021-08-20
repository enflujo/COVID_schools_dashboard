from fastapi import FastAPI
from run import process
import os

base = os.getcwd()
app = FastAPI()


class Args:
    def __init__(self, city, 
                n_school_going_preschool, classroom_size_preschool, n_teachers_preschool,
                height_room_preschool, width_room_preschool, length_room_preschool,
                n_school_going_primary, classroom_size_primary, n_teachers_primary,
                height_room_primary, width_room_primary, length_room_primary,
                n_school_going_highschool, classroom_size_highschool, n_teachers_highschool,
                height_room_highschool, width_room_highschool, length_room_highschool,
                ventilation_level, masks_type, class_duration):
        ## USER
        self.city = city  # City

        # School parameters
        self.n_teachers_preschool = n_teachers_preschool
        self.n_school_going_preschool = n_school_going_preschool
        # self.n_classrooms_preschool = n_classrooms_preschool
        self.preschool_size = classroom_size_preschool
        self.height_room_preschool = height_room_preschool
        self.width_room_preschool = width_room_preschool
        self.length_room_preschool = length_room_preschool

        self.n_teachers_primary = n_teachers_primary
        self.n_school_going_primary = n_school_going_primary
        # self.n_classrooms_primary = n_classrooms_primary
        self.primary_size = classroom_size_primary
        self.height_room_primary = height_room_primary
        self.width_room_primary = width_room_primary
        self.length_room_primary = length_room_primary

        self.n_teachers_highschool = n_teachers_highschool
        self.n_school_going_highschool = n_school_going_highschool
        # self.n_classrooms_highschool = n_classrooms_highschool
        self.highschool_size = classroom_size_highschool
        self.height_room_highschool = height_room_highschool
        self.width_room_highschool = width_room_highschool
        self.length_room_highschool = length_room_highschool

        if ventilation_level == 'bajo':
            self.ventilation_out = 2 # Ventilation values (h-1) that define how much ventilated is a classroom [2-15]
        elif ventilation_level == 'medio':
            self.ventilation_out = 7 # Ventilation values (h-1) that define how much ventilated is a classroom [2-15]
        elif ventilation_level == 'alto':
            self.ventilation_out = 15 # Ventilation values (h-1) that define how much ventilated is a classroom [2-15]

        self.masks_type = masks_type  # Type of masks that individuals are using. Options are: cloth, surgical, N95
        self.duration_event = class_duration  # Duration of event (i.e. classes/lectures) in hours over a day

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
        # self.length_room_preschool = 7.0  # Preschool length of classroom
        # self.width_room_preschool= 7.0  # Preschool width of classroom
        # self.length_room_primary = 10.0  # Primary length of classroom
        # self.width_room_primary= 10.0  # Primary width of classroom
        # self.length_room_highschool = 10.0  # Highschool length of classroom
        # self.width_room_highschool= 10.0  # Highschool width of classroom
        self.Tmax = 200  # Length of simulation (days)
        self.delta_t = 0.08  # Time steps

        self.preschool_mean = 9.4  # Preschool degree distribution (mean)
        self.preschool_std = 1.8  # preschool degree distribution (standard deviation)
        # self.preschool_size = 15  # Number of students per classroom
        self.preschool_r = 1  # Correlation in preschool layer
        self.primary_mean = 9.4  # Primary degree distribution (mean)
        self.primary_std = 1.8  # Primary degree distribution (standard deviation)
        # self.primary_size = 35  # Number of students per classroom
        self.primary_r = 1  # Correlation in primary layer
        self.highschool_mean = 9.4  # Highschool degree distribution (mean)
        self.highschool_std = 1.8  # Highschool degree distribution (standard deviation)
        # self.highschool_size = 35  # Number of students per classroom
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


@app.get("/")
async def root(
    city: str = "bogota",
    # n_teachers_vacc: int = 0, not yet implemented
    n_school_going_preschool: int = 150,
    # n_classrooms_preschool: int = 0,
    classroom_size_preschool: int = 15,
    n_teachers_preschool: int = 5,
    height_room_preschool: float = 3.1,
    width_room_preschool: float = 7.0,
    length_room_preschool: float = 7.0,

    n_school_going_primary: int = 200,
    # n_classrooms_primary: int = 0,
    classroom_size_primary: int = 35,
    n_teachers_primary: int = 6,
    height_room_primary: float = 3.1,
    width_room_primary: float = 10.0,
    length_room_primary: float = 10.0,

    n_school_going_highschool: int = 200,
    # n_classrooms_highschool: int = 0,
    classroom_size_highschool: int = 35,
    n_teachers_highschool: int = 7,
    height_room_highschool: float = 3.1,
    width_room_highschool: float = 10.0,
    length_room_highschool: float = 10.0,

    school_type: bool = False,

    masks_type: str = "N95",
    ventilation_level: str = 'alto',
    class_duration: int = 6,
):
    args = Args(city, 
                n_school_going_preschool, classroom_size_preschool, n_teachers_preschool,
                height_room_preschool, width_room_preschool, length_room_preschool,
                n_school_going_primary, classroom_size_primary, n_teachers_primary,
                height_room_primary, width_room_primary, length_room_primary,
                n_school_going_highschool, classroom_size_highschool, n_teachers_highschool,
                height_room_highschool, width_room_highschool, length_room_highschool,
                ventilation_level, masks_type, class_duration)
    # print(args.__dict__)
    await process(args)
    return {"message": args}
