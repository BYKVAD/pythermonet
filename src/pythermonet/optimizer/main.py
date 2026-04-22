from pythermonet.optimizer.configure_thermonet import execute_dimensioning
from lcc5gdhc.system_model import evaluate_project_ts
from pythermonet.optimizer.configure_lcoe import manipulate_lcoe_input, set_init_lcoe

lcoe_model = set_init_lcoe()


def cost():
    dims = execute_dimensioning() # Generate some kind of heating demand and insert it as function call
    
    # lcoe_model = manipulate_lcoe_input()
    
    # res = evaluate_project_ts(
    #     years=lcoe_model.years, steps_per_year=lcoe_model.steps_per_year, real_discount_rate=lcoe_model.r,
    #     capex_ts=lcoe_model.capex, opex_fixed_ts=lcoe_model.ofix, opex_variable_ts=lcoe_model.ovar,
    #     price_paths_ts=lcoe_model.ppaths, loads_ts=lcoe_model.loads, debt_ts=lcoe_model.debt_cfg
    # )

    print(dims)


if __name__ == "__main__":
    cost()

