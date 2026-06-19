def find_centroid(uav):
    if len(uav.assigned_tasks) == 0:
        return uav.x, uav.y

    x = sum(task.x for task in uav.assigned_tasks)/len(uav.assigned_tasks)
    y = sum(task.y for task in uav.assigned_tasks)/len(uav.assigned_tasks)

    return x,y

def centroid_distance(uav,new_task):
    cen_x,cen_y = find_centroid(uav)
    return ((cen_x - new_task.x)**2 + (cen_y - new_task.y)**2)**(1/2)

def calculate_resource_penalty(uav, new_task):

    compute_used = sum(
        t.compute_load
        for t in uav.assigned_tasks
    )

    hover_used = sum(
        t.hover_time
        for t in uav.assigned_tasks
    )

    energy_used = sum(
        t.energy_cost
        for t in uav.assigned_tasks
    )

    compute_ratio = (
        compute_used + new_task.compute_load
    ) / max(1, uav.max_compute)

    hover_ratio = (
        hover_used + new_task.hover_time
    ) / max(1, uav.max_hover_time)

    energy_ratio = (
        energy_used + new_task.energy_cost
    ) / max(1, uav.max_energy)

    return (
        5 * compute_ratio +
        hover_ratio +
        energy_ratio
    )

def can_accept(uav, task):

    compute_used = sum(
        t.compute_load
        for t in uav.assigned_tasks
    )

    hover_used = sum(
        t.hover_time
        for t in uav.assigned_tasks
    )

    energy_used = sum(
        t.energy_cost
        for t in uav.assigned_tasks
    )

    if compute_used + task.compute_load > uav.max_compute:
        return False

    if hover_used + task.hover_time > uav.max_hover_time:
        return False

    if energy_used + task.energy_cost > uav.max_energy:
        return False

    return True

def assign_new_task(tasks, uavs, new_task):

    best_score = float('inf')
    best_uav = None

    for uav in uavs:

        if not can_accept(uav, new_task):
            continue

        x, y = find_centroid(uav)

        spatial_cost = centroid_distance(uav, new_task)

        resource_cost = calculate_resource_penalty(
            uav,
            new_task
        )

        score = spatial_cost + 1000 * resource_cost

        if score < best_score:
            best_score = score
            best_uav = uav

    if best_uav:
        best_uav.assigned_tasks.append(new_task)
        new_task.assigned_uav = best_uav