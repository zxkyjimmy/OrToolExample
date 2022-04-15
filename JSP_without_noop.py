"""Minimal jobshop example."""
import collections
from ortools.sat.python import cp_model


def main():
  """Minimal jobshop problem."""
  # Data.
  jobs_data = [  # task = (machine_id, processing_time).
    [(2,  1), (0,  3), (1,  6), (3,  7), (5,  3), (4,  6)],
    [(1,  8), (2,  5), (4, 10), (5, 10), (0, 10), (3,  4)],
    [(2,  5), (3,  4), (5,  8), (0,  9), (1,  1), (4,  7)],
    [(1,  5), (0,  5), (2,  5), (3,  3), (4,  8), (5,  9)],
    [(2,  9), (1,  3), (4,  5), (5,  4), (0,  3), (3,  1)],
    [(1,  3), (3,  3), (5,  9), (0, 10), (4,  4), (2,  1)],
  ]

  machines_count = 1 + max(task[0] for job in jobs_data for task in job)
  all_machines = range(machines_count)
  # Compute horizon dynamically as the sum of all durations.
  horizon = sum(task[1] for job in jobs_data for task in job)

  # Create the model.
  model = cp_model.CpModel()

  # Named tuple to store information about created variables.
  task_type = collections.namedtuple('task_type', 'start end interval')
  # Named tuple to manipulate solution information.
  assigned_task_type = collections.namedtuple('assigned_task_type',
                                              'start job index duration')

  # Create job intervals and add to the corresponding machine lists.
  all_tasks = {}
  all_waits = {}
  all_idles = {}
  machine_to_intervals = collections.defaultdict(list)

  machine_to_tasks = collections.defaultdict(list)
  machine_to_waits = collections.defaultdict(list)

  for job_id, job in enumerate(jobs_data):
    pre_end = 0
    for task_id, task in enumerate(job):
      machine = task[0]
      duration = task[1]
      suffix = f'_{job_id}_{task_id}'
      start_var = model.NewIntVar(0, horizon, 'start' + suffix)
      end_var = model.NewIntVar(0, horizon, 'end' + suffix)
      interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                          'interval' + suffix)
      all_tasks[job_id, task_id] = task_type(
          start=start_var, end=end_var, interval=interval_var)
      machine_to_intervals[machine].append(interval_var)
      machine_to_tasks[machine].append(all_tasks[job_id, task_id])

      # Job wait
      wait_time = model.NewIntVar(0, horizon, 'wait time' + suffix)
      wait = model.NewIntervalVar(pre_end, wait_time, start_var,
                                  'wait' + suffix)
      machine_to_waits[machine].append(wait)
      all_waits[job_id, task_id] = task_type(
          start=pre_end, end=start_var, interval=wait)
      pre_end = end_var

  # Create no-op constraints.
  for machine in all_machines:
    num = len(machine_to_tasks[machine])
    arcs = []
    for j1 in range(num + 1):
      for j2 in range(num + 1):
        if j1 == j2:
          continue
        lit = model.NewBoolVar(f'{j2} follows {j1} on {machine}')
        arcs.append([j1, j2, lit])
        if j2 == 0:
          continue
        j1_end = machine_to_tasks[machine][j1 - 1].end if j1 > 0 else 0
        j2_start = machine_to_tasks[machine][j2 - 1].start
        idle_time = model.NewIntVar(0, horizon, f'idle_{j1}_{j2} on {machine}')
        start = model.NewIntVar(0, horizon, f'start_{j1}_{j2} on {machine}')
        end = model.NewIntVar(0, horizon, f'end_{j1}_{j2} on {machine}')
        idle = model.NewIntervalVar(start, idle_time, end,
                                    f'idle_{j1}_{j2} on {machine}')
        all_idles[machine, j1, j2] = task_type(
            start=start, end=end, interval=idle)
        tmp_start = model.NewIntVar(0, horizon, f'tmp_start_{j1}_{j2} on {machine}')
        tmp_end = model.NewIntVar(0, horizon, f'tmp_end_{j1}_{j2} on {machine}')
        same = model.NewBoolVar(f'{j1}.end == {j2}.start on {machine}')
        model.Add(j1_end == j2_start).OnlyEnforceIf(same)
        model.Add(tmp_start == 0).OnlyEnforceIf(same)
        model.Add(tmp_end == 0).OnlyEnforceIf(same)
        model.Add(j1_end != j2_start).OnlyEnforceIf(same.Not())
        model.Add(tmp_start == j1_end).OnlyEnforceIf(same.Not())
        model.Add(tmp_end == j2_start).OnlyEnforceIf(same.Not())
        model.Add(start == tmp_start).OnlyEnforceIf(lit)
        model.Add(end == tmp_end).OnlyEnforceIf(lit)
        model.Add(start == 0).OnlyEnforceIf(lit.Not())
        model.Add(end == 0).OnlyEnforceIf(lit.Not())
        for wait in machine_to_waits[machine]:
          model.AddNoOverlap([idle, wait])
    model.AddCircuit(arcs)

  # Create and add disjunctive constraints.
  for machine in all_machines:
    model.AddNoOverlap(machine_to_intervals[machine])

  # Precedences inside a job.
  for job_id, job in enumerate(jobs_data):
    for task_id in range(len(job) - 1):
      model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id,
                                                                  task_id].end)

  # Makespan objective.
  makespan_var = model.NewIntVar(0, horizon, 'makespan')
  model.AddMaxEquality(makespan_var, [
      all_tasks[job_id, len(job) - 1].end
      for job_id, job in enumerate(jobs_data)
  ])

  obj_var = makespan_var
  model.Minimize(obj_var)

  # Create the solver and solve.
  solver = cp_model.CpSolver()
  status = solver.Solve(model)

  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print('Solution:')
    # Create on elist of assigned tasks per machine.
    assigned_jobs = collections.defaultdict(list)
    for job_id, job in enumerate(jobs_data):
      for task_id, task in enumerate(job):
        machine = task[0]
        assigned_jobs[machine].append(
            assigned_task_type(
                start=solver.Value(all_tasks[job_id, task_id].start),
                job=job_id,
                index=task_id,
                duration=task[1]))

    # Create per machine output lines.
    output = ''
    for machine in all_machines:
      # Sort by starting time.
      assigned_jobs[machine].sort()
      sol_line_tasks = 'Machine ' + str(machine) + ': '
      sol_line = '           '
      idle_line = ' idle '
      wait_line = ' wait '

      j1 = 0
      for assigned_task in assigned_jobs[machine]:
        name = f'job_{assigned_task.job}_task_{assigned_task.index}'

        # Add spaces to output to align columns.
        sol_line_tasks += f'{name:15s}'

        start = assigned_task.start
        duration = assigned_task.duration
        sol_tmp = f'[{start},{start + duration}]'
        # Add spaces to output to align columns.
        sol_line += f'{sol_tmp:15s}'

        # idle
        j2 = assigned_task.job + 1
        item = all_idles[machine, j1, j2]
        start = item.start
        if type(start) != int:
          start = solver.Value(start)
        end = item.end
        if type(end) != int:
          end = solver.Value(end)
        idle_tmp = f'[{start},{end}]'
        idle_line += f'{idle_tmp:15s}'
        j1 = j2

        # wait
        item = all_waits[assigned_task.job, assigned_task.index]
        start = item.start
        if type(start) != int:
          start = solver.Value(start)
        end = item.end
        if type(end) != int:
          end = solver.Value(end)
        wait_tmp = f'[{start},{end}]'
        wait_line += f'{wait_tmp:15s}'

      sol_line += '\n'
      idle_line += '\n'
      wait_line += '\n'
      sol_line_tasks += '\n'
      output += sol_line_tasks
      output += sol_line
      output += idle_line
      output += wait_line

    # Finally print the solution found.
    print(f'Optimal Schedule Objective: {solver.ObjectiveValue()}')
    print(output)

  else:
    print('No solution found.')

  # Statistics.
  print('\nStatistics')
  print(f'  - conflicts: {solver.NumConflicts()}')
  print(f'  - branches : {solver.NumBranches()}')
  print(f'  - wall time: {solver.WallTime():f} s')


if __name__ == '__main__':
  main()
