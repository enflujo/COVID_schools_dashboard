from fastapi import FastAPI
from run import process
import os
from utils.get_cache import run as build_cache
from utils.Args import Args

base = os.getcwd()
app = FastAPI()


@app.get("/")
async def root(
    city: str = "bogota",
    n_school_going_preschool: int = 150,
    classroom_size_preschool: int = 15,
    n_teachers_preschool: int = 5,
    height_room_preschool: float = 3.1,
    width_room_preschool: float = 7.0,
    length_room_preschool: float = 7.0,
    n_school_going_primary: int = 200,
    classroom_size_primary: int = 35,
    n_teachers_primary: int = 6,
    height_room_primary: float = 3.1,
    width_room_primary: float = 10.0,
    length_room_primary: float = 10.0,
    n_school_going_highschool: int = 200,
    classroom_size_highschool: int = 35,
    n_teachers_highschool: int = 7,
    height_room_highschool: float = 3.1,
    width_room_highschool: float = 10.0,
    length_room_highschool: float = 10.0,
    school_type: bool = False,
    masks_type: str = "N95",
    ventilation_level: str = "alto",
    class_duration: int = 6,
):
    args = Args(
        city,
        n_school_going_preschool,
        classroom_size_preschool,
        n_teachers_preschool,
        height_room_preschool,
        width_room_preschool,
        length_room_preschool,
        n_school_going_primary,
        classroom_size_primary,
        n_teachers_primary,
        height_room_primary,
        width_room_primary,
        length_room_primary,
        n_school_going_highschool,
        classroom_size_highschool,
        n_teachers_highschool,
        height_room_highschool,
        width_room_highschool,
        length_room_highschool,
        ventilation_level,
        masks_type,
        class_duration,
    )

    # print(args.__dict__)
    # build_cache(args)
    results = await process(args)

    return {"prediccion": results}
