class RailYard:
    EMPTY = 0
    FULL = 1
    SWITCH_POS_A = 0
    SWITCH_POS_B = 1
    PRODUCTS = {0 : "M", 1 : "D", 2 : "J", 4 : "A"}

    def __init__(self):
        self.NUMBER_OF_TRACKS = 7
        self.MAX_TRACK_LENGTH = 10
        self.NUMBER_OF_CARS = 4
        self.NUMBER_OF_SETS = 2
        self.INBOUND_TRACK_ID = 2
        self.OUTBOUND_TRACK_ID = 7

        #Create tracks
        self.lead1 = Track(1,5)
        self.inbound = Track(2,5)
        self.marshalling_track1 = MarshallingTrack(3,5,RailYard.PRODUCTS[0])
        self.marshalling_track2 = MarshallingTrack(4,5,RailYard.PRODUCTS[1])
        self.rack1 = Rack(5,2,RailYard.PRODUCTS[0],2)
        self.rack2 = Rack(6,2,RailYard.PRODUCTS[1],2)    
        self.outbound = Track(7,5)

        #Connect tracks to form network
        self.lead1.connect(self.inbound)
        self.lead1.connect(self.marshalling_track1)
        self.lead1.connect(self.marshalling_track2)
        self.lead1.connect(self.rack1)
        self.lead1.connect(self.rack2)
        self.lead1.connect(self.outbound)
        #self.switch1 = Switch(self.lead1, self.spur1, self.spur2)
        
        #Create track reference lists
        self.tracks = {1 : self.lead1, 2 : self.inbound, 3 :  self.marshalling_track1, 4 : self.marshalling_track2, 5 : self.rack1, 6 : self.rack2, 7 : self.outbound}
        self.racks = {1 : self.rack1, 2 : self.rack2}
        self.marshalling_tracks = [self.marshalling_track1, self.marshalling_track2]

        #Create cars
        self.cars = []
        for i in range(2):
            self.cars.append(RailCar(i,"m" + str(i+1),RailYard.EMPTY, RailYard.PRODUCTS[0]))
            self.inbound.push(self.cars[i])
        for j in range(2,4):
            self.cars.append(RailCar(j,"d" + str(j+1),RailYard.EMPTY, RailYard.PRODUCTS[1]))
            self.inbound.push(self.cars[j])

        #Create locomotive
        self.loco = Locomotive()
        
        #Build load schedule
        self.loading_schedule = LoadingSchedule()
        self.loading_schedule.add_to_schedule(1, self.cars[1], self.cars[1].product)
        self.loading_schedule.add_to_schedule(1, self.cars[2], self.cars[2].product)
        self.loading_schedule.add_to_schedule(2, self.cars[3], self.cars[3].product)
        
class RailCar:
    def __init__(self, ID, number, empty_full, product):
        self.number = number
        self.ID = ID
        self.empty_or_full = empty_full
        self.product = product

    def is_empty(self):
        return True if self.empty_or_full == RailYard.EMPTY else False

    def __str__(self):
        if self.empty_or_full == RailYard.EMPTY:
            return self.number.lower()
        else:
            return self.number.capitalize()

class Track:
    def __init__(self, ID, length):
        self.ID = ID
        self.cars = []
        self.length = length
        self.connected_tracks = []

    def push(self,car):
        self.cars.append(car)

    def pop(self):
        return self.cars.pop()

    def peek(self):
        return self.cars[-1]

    def is_empty(self):
        return self.cars == []

    def is_full(self):
        return True if self.number_of_cars() == self.length else False

    def number_of_cars(self):
        return len(self.cars)

    def get_cars(self):
        return self.cars

    def number_of_empty_spots(self):
        return self.length - self.number_of_cars()

    def derail_up(self):
        return False

    def __str__(self):
        return str(self.ID) + ": " + str([str(car) for car in self.cars])

    def connect(self, track):
        self.connected_tracks.append(track)
        track.connected_tracks.append(self)

"""
"NOT USED"
#A switch is a track of size 1, track C diverges from the A and B straight ahead
class Switch(Track):
    def __init__(self, track_A, track_B, track_C):
        self.track_A = track_A
        self.track_B = track_B
        self.track_C = track_C
        self.A_B = SWITCH_POS_A
        super(self.__class__, self).__init__(1)
    
    #throw the switch to either A or B position
    def switch(self, switch_pos):
        self.A_B = switch_pos
"""

class MarshallingTrack(Track):
    """A track that is designated for a specific product.
    
    Attributes:
        ID: the unique inteer identifier of this track in the yard
        length: the length of the track in car units
        product the type of proudct for which  this track is intended 
    """

    def __init__(self, ID, length, product):
        self.product = product
        super(self.__class__, self).__init__(ID, length)

class Rack(Track):
    def __init__(self, ID, num_bays, product, load_time):
        self.load_time = load_time
        self.product = product
        self.current_load_time = 0
        self.is_loading = False
        super(self.__class__, self).__init__(ID, num_bays)

    def start_load(self):
        """Starts the process of loading cars under a rack if at least one car is product compatible."""

        for car in self.cars:
            if car.product == self.product:
                self.current_load_time = 0
                self.is_loading = True
                break

    def load_step(self):
        """Continues loading cars on rack. Finish loading cars if sufficient time has passed."""

        if self.is_loading == True and self.current_load_time >= self.load_time:
            #we are done loading
            for car in self.cars:
                #only fill up car for the rack’s product
                if car.product == self.product:
                    car.empty_or_full = RailYard.FULL 
            self.is_loading = False
            self.current_load_time = 0

        self.current_load_time += 1

    def is_currently_loading(self):
        return self.is_loading

    def derail_up(self):
        return self.is_loading

class Locomotive:
    """An engine that can pull cars from track to track
    """
    def __init__(self):
        self.active = False

    def move_cars(self, tracks, source_track, destination_track, num_cars):
        self.path = self.shortest_path(len(tracks), tracks, source_track, destination_track, num_cars)
        self.current_hop = 0
        self.active = True

    def shortest_path(self, max_hops, tracks, source_track, destination_track, num_cars):
        if destination_track.ID == 1:
            #destination is the lead track
            return [(source_track.ID, destination_track.ID, num_cars)]
        else:
            #first hop is the source to lead then lead to destination
            return [(source_track.ID, 1, num_cars), (1, destination_track.ID, num_cars)]    


    def is_active(self):
        return self.active
    
    def next_hop(self):
        hop = self.path[self.current_hop]
        self.current_hop += 1
        if self.current_hop == len(self.path):
            self.active = False
        return hop

class LoadingSchedule:
    """A list of tuples that represents the rail cars and products that need to be loaded for a given set on a given day
    """
    def __init__(self):
        self.loading_schedule = [[]]
        self.cars = [[]] 
    
    def add_to_schedule(self, set, car, product):
        #are we adding another set?
        if set > len(self.loading_schedule):
            self.loading_schedule.append([])
            self.cars.append([])    
        self.loading_schedule[set - 1].append([car, product])
        self.cars[set - 1].append(car)
        
    def get_cars(self,set):
        return self.cars[set - 1]
    
    def is_on_set_schedule(self, car, set, product):
        for car_product in self.loading_schedule[set-1]:
            if car_product[0] == car and car_product[1] == product:
                return True
        return False

    def number_of_sets(self):
        return(len(self.loading_schedule))

    def __str__(self):
        print_string = ""
        for i in range(len(self.loading_schedule)):
            print_string += "Set: " + str(i+1) + "\n"
            for car_product in self.loading_schedule[i]:
                print_string += "Car: " + car_product[0].number + " Product:" + car_product[1] + "\n"
        return print_string

