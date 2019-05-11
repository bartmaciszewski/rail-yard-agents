import numpy as np

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
       self.curr_car_count += num_cars 

    def car_count(self):
        return self.curr_car_count

    def get_next_cars(self):
        num_cars = np.random.normal(size = 1, loc = 100, scale = 20)
        self.remove_cars(num_cars)
        return num_cars

class RailCarInventoryGame:

    def __init__(self, num_periods):
        self.ext = RailCarSet("External", 1000, 200)
        self.ery = RailCarSet("ERY", 1000, 200)
        self.num_periods = 30
    
    def run(self):
        for i in range(self.num_periods):
            cars_delivered = self.ext.get_next_cars()
            cars_loaded = self.ery.get_next_cars()        
            self.ery.add_cars(cars_delivered)
            self.ext.add_cars(cars_loaded)
            print("Period {}: {} were delivered and {} were loaded. ERY: {} EXT: {}".format(i, cars_delivered, cars_loaded, self.ery.car_count(), self.ext.car_count()))

def main():
    RailCarInventoryGame(10).run()

main()