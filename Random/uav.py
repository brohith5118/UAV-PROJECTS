from config import ENERGY_PER_METER, UAV_SPEED

class UAV:
    def __init__(self,uav_id,x,y,uav_type,max_energy,max_hover,max_compute):
        self.uav_id = uav_id

        self.x = x
        self.y = y

        self.uav_type = uav_type

        self.max_energy = max_energy
        self.max_hover = max_hover
        self.max_compute = max_compute

        #Runtime statistics
        self.curr_x = x
        self.curr_y = y

        self.curr_energy = max_energy
        self.curr_hover = max_hover
        self.curr_compute = max_compute

        self.assigned_tasks = []
        self.active = True

    def assign(self,task):
        self.assigned_tasks.append(task)

    def compute_resource(self,task):
        self.curr_energy -= task.energy_cost
        self.curr_hover -= task.hover_time
        self.curr_compute -= task.compute_cost

    def compute_resource_with_travel(self,task):
        travel_dist = ((self.curr_x - task.x)**2 + (self.curr_y - task.y)**2)**0.5
        travel_energy_cost = travel_dist * ENERGY_PER_METER
        travel_hover_time = travel_dist / UAV_SPEED

        self.curr_energy -= task.energy_cost + travel_energy_cost
        self.curr_hover -= task.hover_time + travel_hover_time
        self.curr_compute -= task.compute_cost

    def move_to(self,task):
        self.curr_x = task.x
        self.curr_y = task.y

    def reset_position(self):
        self.curr_x = self.x
        self.curr_y = self.y

    def reset_resource(self):
        self.curr_energy = self.max_energy
        self.curr_hover = self.max_hover
        self.curr_compute = self.max_compute