class Task:
    def __init__(self,task_id,x,y,priority,task_type,energy_cost,hover_time,compute_cost,deadline):

        self.task_id = task_id

        self.x = x
        self.y = y

        self.priority = priority

        self.task_type = task_type

        self.energy_cost = energy_cost
        self.hover_time = hover_time
        self.compute_cost = compute_cost

        self.deadline = deadline

        self.assigned_uav = None
        self.completed = None
        self.start_time = None
        self.finish_time = None