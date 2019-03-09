import sys

EMPTY = 0
FULL = 1
NUMBER_OF_SPURS = 4
NUMBER_OF_CARS = 5

class RailYardEnv2:
    
    def __init__(self):
        self.tracks = [Spur(5), Spur(2), Spur(1), Spur(5)]
        self.period = 0
        for i in range(NUMBER_OF_CARS):
            self.tracks[0].push(RailCar(EMPTY))

    def step(self,action):
        self.period += 1
        #switch k cars from spur to spur                         
        for i in range(NUMBER_OF_SPURS):
            for j in range(NUMBER_OF_SPURS):
                for k in range(action[i][j]):
                    self.tracks[j].push(self.tracks[i].pop())
        return True if self.tracks[NUMBER_OF_SPURS-1].size() == NUMBER_OF_CARS else False #delovered all cars

    def render(self,mode="human"):
        for i in range(NUMBER_OF_SPURS):
            sys.stdout.write(str(self.tracks[i]) + "\n")
        sys.stdout.write("\n")

class Spur:
    def __init__(self, length):
        self.cars = []
        self.length = length

    def push(self,car):
        self.cars.append(car)

    def pop(self):
        return self.cars.pop()

    def isEmpty(self):
        return self.cars == []

    def isFull(self):
        return True if self.size() == self.length else False

    def size(self):
        return len(self.cars)

    def __str__(self):
        return str([str(car) for car in self.cars])

class RailCar:
    def __init__(self, empty_full):
        self.empty_or_full = empty_full

    def __str__(self):
        return str(self.empty_or_full)

class Rack(Spur):
    def load(car,period):
        pass
    
    def isLoading(period):
        pass

class SimpleOneByOnePolicyAgent:
    def __init__(self,rail_yard):
        self.rail_yard = rail_yard

    def play(self):
        action = [[0 for j in range(NUMBER_OF_SPURS)] for i in range(NUMBER_OF_SPURS)]
        for i in reversed(range(NUMBER_OF_SPURS-1)):
            if not self.rail_yard.tracks[i].isEmpty() and not self.rail_yard.tracks[i+1].isFull():
                action[i][i+1] = 1
        return action

#Main
rail_yard = RailYardEnv2()
ss = SimpleOneByOnePolicyAgent(rail_yard)
done = False
rail_yard.render()
while not done:
    action = ss.play()
    done = rail_yard.step(action)
    rail_yard.render()