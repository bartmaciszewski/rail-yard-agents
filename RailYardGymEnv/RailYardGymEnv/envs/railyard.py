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
        self.marshalling_track1 = MarshallingTrack(3,5,PRODUCTS[0])
        self.marshalling_track2 = MarshallingTrack(4,5,PRODUCTS[1])
        self.rack1 = Rack(5,2,PRODUCTS[0],2)
        self.rack2 = Rack(6,2,PRODUCTS[1],2)    
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
            self.cars.append(RailCar(i,"m" + str(i+1),EMPTY, PRODUCTS[0]))
            self.inbound.push(self.cars[i])
        for j in range(2,4):
            self.cars.append(RailCar(j,"d" + str(j+1),EMPTY, PRODUCTS[1]))
            self.inbound.push(self.cars[j])

        #Create locomotive
        self.loco = Locomotive()
        
        #Build load schedule
        self.loading_schedule = LoadingSchedule()
        self.loading_schedule.add_to_schedule(1, self.cars[1], self.cars[1].product)
        self.loading_schedule.add_to_schedule(1, self.cars[2], self.cars[2].product)
        self.loading_schedule.add_to_schedule(2, self.cars[3], self.cars[3].product)


