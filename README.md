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

| nombre | type | default |
| ------ | ---- | ------- |
| city | `str` | `"bogota"` |
| n_teachers | `int` | `0` |
| n_teachers_vacc | `int` | `0` |
| n_school_going | `int` | `0` |
| n_classrooms | `int` | `0` |
| classroom_size | `int` | `0` |
| school_type | `bool` | `False` |
| height_room | `float` | `3.1` |
| width_room | `float` | `0.0` |
| length_room | `float` | `0.0` |
| masks_type | `str` | `"N95"` |
| ventilation_level | `int` | `3` |
| class_duration | `int` | `6` |


## Notas

- Las variables que tenemos como input no coinciden del todo con las que recibe el modelo. Ver en `server.py` las variables que recibe el API y las que se construyen con la clase `Args`.
- **Falta definir post-procesamiento de resultados antes de enviar respuesta al cliente.**
- ¿ Necesitamos pedir ciudad en formulario?
- Si vamos a procesar por ciudades, toca buscar los datos de cada una, de momento sólo tenemos de Bogotá
- Para crear los datos estáticos de cada cache hay unas funciones definidas llamadas `build` en cada archivo `/modules/graph_....` pero aún no tienen implementación para guardar los resultados automáticamente (lo estoy haciendo manual por ahora).

## Problemas
- Los resultados dan valores sin el mismo número de decimales que antes y muchos quedan en 0.0. Por ahora asumo que tiene que ver con los valores estáticos, me faltan pruebas para resolver esto.

## Posibles optimizaciones

- Revisar diferencia entre pandas y Dask (*parallel processing* si armamos un cluster en el servidor)

## Sin Servidor

El modelo se puede correr igual que antes directamente con python.

Por ejemplo:

```bash
python run_sim_glob.py --population 100
```