import networks.network_dynamics_no_interventions as nd_ni
import networks.network_dynamics as nd


def create_dynamics(args, multilayer_matrix, Tmax, total_steps):
    if args.intervention_type == "no_intervention":
        time_intervals, ws = nd_ni.create_day_intervention_dynamics(
            multilayer_matrix,
            Tmax=Tmax,
            total_steps=total_steps,
            schools_day_open=0,
            interv_glob=0,
            schl_occupation=1.0,
            work_occupation=1.0,
        )

    elif args.intervention_type == "intervention":
        time_intervals, ws = nd.create_day_intervention_dynamics(
            multilayer_matrix,
            Tmax=Tmax,
            total_steps=total_steps,
            schools_day_open=args.school_openings,
            interv_glob=args.intervention,
            schl_occupation=args.school_occupation,
            work_occupation=args.work_occupation,
        )

    elif args.intervention_type == "school_alternancy":
        time_intervals, ws = nd.create_day_intervention_altern_schools_dynamics(
            multilayer_matrix,
            Tmax=Tmax,
            total_steps=total_steps,
            schools_day_open=args.school_openings,
            interv_glob=args.intervention,
            schl_occupation=args.school_occupation,
            work_occupation=args.work_occupation,
        )
    else:
        print("No valid intervention type")

    return time_intervals, ws
