import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

class RailCarSet:

    def __init__(self, name, capacity, starting_car_count):
        self.name = name
        self.capacity = capacity
        self.curr_car_count = starting_car_count

    def add_cars(self, num_cars):
        #assert(self.curr_car_count + num_cars <= self.capacity), "Rail car set {} has insufficient capacity for {} cars".format(self.name, num_cars)
        self.curr_car_count += num_cars

    def remove_cars(self, num_cars):
       #assert (self.curr_car_count - num_cars >= 0),"Rail car set {} has less than {} cars".format(self.name, num_cars)
       self.curr_car_count -= num_cars 

    def car_count(self):
        return self.curr_car_count

    def get_next_cars(self):
        num_cars = np.random.normal(size = 1, loc = 100, scale = 20)[0]
        self.remove_cars(num_cars)
        return num_cars

class RailCarInventoryGame:

    def __init__(self, num_periods):
        self.ext = RailCarSet("External", 1000, 400)
        self.ery = RailCarSet("ERY", 500, 200)
        self.num_periods = num_periods
    
    def run(self):
        ERY_inventory_history = []
        EXT_inventory_history = []
        for i in range(self.num_periods):
            cars_delivered = self.ext.get_next_cars()
            cars_loaded = self.ery.get_next_cars()        
            self.ery.add_cars(cars_delivered)
            self.ext.add_cars(cars_loaded)
            ERY_inventory_history.append(self.ery.car_count())
            EXT_inventory_history.append(self.ext.car_count())
            print("Period {}: {} were delivered and {} were loaded. ERY: {} EXT: {}".format(i, cars_delivered, cars_loaded, self.ery.car_count(), self.ext.car_count()))
        plt.plot(range(self.num_periods), ERY_inventory_history, 'r^', range(self.num_periods), EXT_inventory_history, 'bs')
        plt.show()

def main():
    RailCarInventoryGame(30).run()

main()