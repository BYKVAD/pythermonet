class DHCSystem:
    def __init__(self, heatPumps, connectionPipes, distributionPipes, heatCarrier, gheFields):
        self.heatPumps = heatPumps
        self.connectionPipes = connectionPipes
        self.distributionPipes = distributionPipes
        self.heatCarrier = heatCarrier
        self.gheFields = gheFields

        self.ID = -1

    def assignID(self) -> int:
        self.ID += 1
        return self.ID