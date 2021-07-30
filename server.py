from fastapi import FastAPI
from run import process
import os

base = os.getcwd()
app = FastAPI()


class Args:
    def __init__(self, city, ventilation_level, masks_type, class_duration, height_room):
        self.city = city  # City
        # Ventilation values (h-1) that define how much ventilated is a classroom [2-15]
        self.ventilation_out = ventilation_level
        self.masks_type = masks_type  # Type of masks that individuals are using. Options are: cloth, surgical, N95
        self.duration_event = class_duration  # Duration of event (i.e. classes/lectures) in hours over a day
        self.height_room = height_room  # Schools height of classroom

        # Nodes
        self.population = 100  # Speficy the number of individials
        self.number_trials = 2  # Number of iterations per step

        ############ Defaults ############
        self.res_id = "ND"  # Result ID for simulation save

        self.intervention = 0.6  # Intervention efficiancy
        # Define the type of intervention [no_intervention,internvention,school_alternancy]
        self.intervention_type = "intervention"
        self.work_occupation = 0.6  # Percentage of occupation at workplaces over intervention
        self.school_occupation = 0.35  # Percentage of occupation at classrooms over intervention
        self.school_openings = 20  # Day of the simulation where schools are open
        self.fraction_people_masks = 1.0  # Fraction value of people wearing masks
        self.preschool_length_room = 7.0  # Preschool length of classroom
        self.preschool_width_room = 7.0  # Preschool width of classroom
        self.primary_length_room = 10.0  # Primary length of classroom
        self.primary_width_room = 10.0  # Primary width of classroom
        self.highschool_length_room = 10.0  # Highschool length of classroom
        self.highschool_width_room = 10.0  # Highschool width of classroom
        self.Tmax = 200  # Length of simulation (days)
        self.delta_t = 0.08  # Time steps

        self.preschool_mean = 9.4  # Preschool degree distribution (mean)
        self.preschool_std = 1.8  # preschool degree distribution (standard deviation)
        self.preschool_size = 15  # Number of students per classroom
        self.preschool_r = 1  # Correlation in preschool layer
        self.primary_mean = 9.4  # Primary degree distribution (mean)
        self.primary_std = 1.8  # Primary degree distribution (standard deviation)
        self.primary_size = 35  # Number of students per classroom
        self.primary_r = 1  # Correlation in primary layer
        self.highschool_mean = 9.4  # Highschool degree distribution (mean)
        self.highschool_std = 1.8  # Highschool degree distribution (standard deviation)
        self.highschool_size = 35  # Number of students per classroom
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
    n_teachers: int = 0,
    n_teachers_vacc: int = 0,
    n_school_going: int = 0,
    n_classrooms: int = 0,
    classroom_size: int = 0,
    school_type: bool = False,
    height_room: float = 3.1,
    width_room: float = 0.0,
    length_room: float = 0.0,
    masks_type: str = "N95",
    ventilation_level: int = 3,
    class_duration: int = 6,
):
    args = Args(city, ventilation_level, masks_type, class_duration, height_room)
    # print(args.__dict__)
    await process(args)
    return {"message": args}
