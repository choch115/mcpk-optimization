from optimizer import Optimizer

o = Optimizer()

o.add_ground_tick()
o.add_jump_tick()
for i in range(6):
    o.add_air_tick()

o.set_initial_velocity(0.364932652177083)

m = 1
m2 = 5
n = 8

o.set_objective_minimize('x', 0, n)
o.add_constraint('x', 0, m, 3/16, None)
o.add_constraint('x', 0, m2, 3/16, None)
o.add_constraint('z', m, m2, 16/16+0.6, None)

o.optimize()

o.set_x_position_anchor(0, 1.3)
o.set_z_position_anchor(0, 1.7)
o.print_result()