import pandas as pd


def build(args):
    teachers_data_BOG = pd.read_csv(args.teachers_data_path, encoding="unicode_escape", delimiter=",")
    total_teachers_BOG = int(teachers_data_BOG["Total"][1])

    teachers_preschool_ = [int(teachers_data_BOG["Preescolar"][1])]
    teachers_primary_ = [int(teachers_data_BOG["Basica_primaria"][1])]
    teachers_highschool_ = [int(teachers_data_BOG["Basica_secundaria"][1])]

    return {
        "preschool": [teachers_preschool_, sum(teachers_preschool_) / total_teachers_BOG],
        "primary": [teachers_primary_, sum(teachers_primary_) / total_teachers_BOG],
        "highschool": [teachers_highschool_, sum(teachers_highschool_) / total_teachers_BOG],
        "total": total_teachers_BOG,
    }


def cache(args):
    ########################### Static teachers distribution ################################################
    teachers = {
        "bogota": {
            "preschool": [[9701], 0.1507443204773596],
            "primary": [[22662], 0.3521459427541412],
            "highschool": [[19927], 0.3096466420113746],
            "total": 64354,
        }
    }

    return teachers[args.city]
