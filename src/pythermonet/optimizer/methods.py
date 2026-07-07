import numpy as np
from scipy.optimize import brentq

def fit_curves(common_x: list[float | int], y1: list[float |int], y2: list[float | int]):
    # Fit should probably be tested with various types of fits up to a maximum of degrees?
    model_1 = np.poly1d(np.polyfit(common_x, y1, deg=2))
    model_2 = np.poly1d(np.polyfit(common_x, y2, deg=2))

    return model_1, model_2


def find_root(model_1, model_2, low, high):
    def diff(x):
        return model_1(x) - model_2(x)

    if diff(low) * diff(high) < 0:
        x_intersect = brentq(diff, low, high)
        print("Crosses at", x_intersect)
        return x_intersect

    print(f"No sign change on [{low}, {high}]: diff(low)={diff(low):.2f}, diff(high)={diff(high):.2f}")
    return None


# Change to x-axis being total watt*h (i guess this should be based on the electricity?)
#   x-axis will be defined as the sum of electricity load for the number of heatpumps
# Håndter intersection uden at skulle bruge fit af linjer (muligvis simplere når det er watt*h grundet ikke afbøjnoing til sidste)
#   alternativt, når vi har fikset watt*h, så kan vi ved hjælp af matematik udregne skæringen
# Sørg for at energiberegning fra monthly_elec_loads også er den samme energiberegning som indgå i execute_dimensioning (pythermonet)
# clean code when we achieved a starting point
# revisit heat_pump.py -> Does it make sense with a capacity factor to determine the difference or can we do this more dynamic using cop and building loads? (Should be coupled with fast load estimator)