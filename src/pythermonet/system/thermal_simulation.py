class ThermalSimulation:
    def __init__(self, dhcSystem, soil, climate, gFunctions, flowResults):
        self.dhcSystem = dhcSystem
        self.soil = soil
        self.climate = climate
        self.gFunctions = gFunctions
        self.flowResults = flowResults

        self.thermalResults = None
        self.thermalResistance = None

    def computeThermalResistances(self):
        pass
