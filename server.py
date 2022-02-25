import json
from fastapi import FastAPI, WebSocket

from utils.Args import Args

from utils.NumpyEncoder import NumpyEncoder
from modules.simulate import simulate

from modules.graphs import create_graph_matrix
from networks.network_dynamics import create_day_intervention_dynamics as dynamics

from utils.get_cache import run as build_cache

app = FastAPI()

# TODO:
# - Eliminar logs cuando se desconecta usuario.


@app.websocket("/ws")
async def socket(websocket: WebSocket):
    await websocket.accept()

    while True:
        # Socket por donde se reciben los inputs del usuario
        data = await websocket.receive_json()

        if data["tipo"] == "inicio":
            # Inputs entran en JSON y se suman a los args
            args = Args(data)

            # Crear caches
            # build_cache(args)

            # Iniciar proceso
            pop = args.population * 2
            Tmax = args.Tmax
            days_intervals = [1] * Tmax
            delta_t = args.delta_t
            step_intervals = [int(x / delta_t) for x in days_intervals]
            total_steps = sum(step_intervals)

            # Crear nodos
            print("nodos")
            nodes, multilayer_matrix = create_graph_matrix(args)

            # Pasa los datos por un convertidor de numpy a json string.
            nodesString = json.dumps(nodes, cls=NumpyEncoder)
            # Envía nodos al cliente - toca convertirlo de json string a json para que lleguen estructurados al front.
            await websocket.send_json({"tipo": "nodos", "datos": json.loads(nodesString)})
            print("creando dinamicas")
            # Crear capas
            time_intervals, ws = dynamics(
                multilayer_matrix,
                Tmax,
                total_steps,
                args.school_openings,
                args.intervention,
                args.school_occupation,
                args.work_occupation,
            )

            trials = simulate(args, total_steps, pop, ws, time_intervals)
            trialI = 0
            stateI = 0
            trialsLen = args.number_trials
            statesLen = 600
            trial = next(trials)

        elif data["tipo"] == "sim":
            if trialI < trialsLen and stateI < statesLen:
                state = next(trial)
                stateI = stateI + 1

                stateString = json.dumps(state, cls=NumpyEncoder)
                await websocket.send_json(
                    {
                        "tipo": "estado",
                        "datos": json.loads(stateString),
                        "trialI": trialI,
                        "stateI": stateI,
                    }
                )

                if stateI == statesLen:
                    trialI = trialI + 1
                    stateI = 0
                    try:
                        trial = next(trials)
                    except StopIteration:
                        await websocket.send_json({"tipo": "fin"})
                        print("fin")
