# COVID_schools_dashboard

## Servidor

Instalar dependencias:

```bash
pip install fastapi
```

```bash
pip install uvicorn[standard]
```

Iniciar el servidor con:

```bash
uvicorn server:app
```

Para desarrollo local se puede iniciar con `--reload` para reiniciar cuando se hacen cambios en el código:

```bash
uvicorn server:app --reload
```

## Sitio

**La página queda en <a href="http://localhost:8000" target="_blank">localhost:8000</a>**

Para pasar parámetros al API:

```bash
http://localhost:8000/?city=cali&n_teachers=90
```

### Parametros

| nombre | type | default | *opciones* |
| ------ | ---- | ------- | -------- |
| city | `str` | `"bogota"` | |
| n_school_going_preschool | `int` | `150` | |
| classroom_size_preschool | `int` | `15` | |
| n_teachers_preschool | `int` | `5` | |
| height_room_preschool | `float` | `3.1` | |
| width_room_preschool | `float` | `7.0` | |
| length_room_preschool | `float` | `7.0` | |
| n_school_going_primary | `int` | `200` | |
| classroom_size_primary | `int` | `35` | |
| n_teachers_primary | `int` | `6` | |
| height_room_primary | `float` | `3.1` | |
| width_room_primary | `float` | `10.0` | |
| length_room_primary | `float` | `10.0` | |
| n_school_going_highschool | `int` | `200` | |
| classroom_size_highschool | `int` | `35` | |
| n_teachers_highschool | `int` | `7` | |
| height_room_highschool | `float` | `3.1` | |
| width_room_highschool | `float` | `10.0` | |
| length_room_highschool | `float` | `10.0` | |
| school_type | `bool` | `False` | |
| masks_type | `str` | `"N95"` | `"cloth", "surgical" o "N95"`|
| ventilation_level | `str` | `"alto"` | `"bajo", "medio" o "alto"`|
| class_duration | `int` | `6` | |
