import numpy as np
from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint,Bounds

class Optimizer:
    def __init__(self):
        self._imux,self.imux = [],None
        self._imuz,self.imux = [],None
        self._mmu,self.mmu = [None],None
        self.inertia_ticks_x = set()
        self.inertia_ticks_z = set()
        self.F = None
        self.objective = None
        self.constraints = []
        self.angle_bounds = []
        self.bounds = None

        self.result = None

        self.x_pos_anchor_tick = 0
        self.x_pos_anchor = 0
        self.z_pos_anchor_tick = 0
        self.z_pos_anchor = 0
    
    def set_initial_velocity(self, v):
        self._mmu[0] = v

    def add_air_tick(self):
        if len(self._imux) == len(self._mmu):
            self._mmu.append(0.026)
        self._imux.append(0.91)
        self._imuz.append(0.91)
    
    def add_jump_tick(self):
        if len(self._imux) == len(self._mmu):
            self._mmu.append(0.3274)
        self._imux.append(0.546)
        self._imuz.append(0.546)
    
    def add_strafejump_tick(self):
        if len(self._imux) == len(self._mmu):
            self._mmu.append(0.3060548)
        self._imux.append(0.546)
        self._imuz.append(0.546)
    
    def add_ground_tick(self):
        if len(self._imux) == len(self._mmu):
            self._mmu.append(0.1274)
        self._imux.append(0.546)
        self._imuz.append(0.546)

    def _X(self, F, t):
        return np.sum([self.mmu[i]*np.sin(F[i])*np.sum([np.prod([self.imux[k] for k in range(i,j)]) for j in range(i,t)]) for i in range(t)])

    def _Z(self, F, t):
        return np.sum([self.mmu[i]*np.cos(F[i])*np.sum([np.prod([self.imuz[k] for k in range(i,j)]) for j in range(i,t)]) for i in range(t)])
    
    def set_objective_minimize(self, axis, t1, t2):
        if axis == 'x':
            self.objective = lambda F: self._X(F, t2)-self._X(F, t1)
        elif axis == 'z':
            self.objective = lambda F: self._Z(F, t2)-self._Z(F, t1)
        else:
            raise ValueError('axis should be either \'x\' or \'z\'')
    
    def set_objective_maximize(self, axis, t1, t2):
        if axis == 'x':
            self.objective = lambda F: -(self._X(F, t2)-self._X(F, t1))
        elif axis == 'z':
            self.objective = lambda F: -(self._Z(F, t2)-self._Z(F, t1))
        else:
            raise ValueError('axis should be either \'x\' or \'z\'')
    
    def add_constraint(self, axis, t1, t2, lower, upper):
        new_lower,new_upper = lower,upper
        if new_lower == None:
            new_lower = -np.inf
        if new_upper == None:
            new_upper = np.inf

        if axis == 'x':
            self.constraints.append(NonlinearConstraint(lambda F: self._X(F,t2)-self._X(F,t1), new_lower, new_upper))
        elif axis == 'z':
            self.constraints.append(NonlinearConstraint(lambda F: self._Z(F,t2)-self._Z(F,t1), new_lower, new_upper))
        else:
            raise ValueError('axis should be either \'x\' or \'z\'')
    
    def add_inertia_tick(self, axis, t):
        if axis == 'x':
            self.inertia_ticks_x.add(t)
        elif axis == 'z':
            self.inertia_ticks_z.add(t)
        else:
            raise ValueError('axis should be either \'x\' or \'z\'')
        
    def add_angle_constraint(self, t, lower, upper):
        new_lower,new_upper = lower,upper
        if new_lower == None:
            new_lower = -np.inf
        if new_upper == None:
            new_upper = np.inf
        new_lower *= np.pi/180
        new_upper *= np.pi/180
        
        # self.angle_bounds.append((t,new_lower,new_upper))
        self.constraints.append(NonlinearConstraint(lambda F: F[t], new_lower, new_upper))
    
    def set_x_position_anchor(self, t, value):
        self.x_pos_anchor = value
        self.x_pos_anchor_tick = t
    
    def set_z_position_anchor(self, t, value):
        self.z_pos_anchor = value
        self.z_pos_anchor_tick = t

    def optimize(self):

        for k in self.inertia_ticks_x:
            self._imux[k-1] = 0
            self.constraints.append(NonlinearConstraint(lambda F: abs(self._X(F,k)-self._X(F,k-1)), -np.inf, 0.005/0.91))
        for k in self.inertia_ticks_z:
            self._imuz[k-1] = 0
            self.constraints.append(NonlinearConstraint(lambda F: abs(self._Z(F,k)-self._Z(F,k-1)), -np.inf, 0.005/0.91))
        

        n = len(self._mmu)

        # lower_bnds = [-np.inf]*n
        # upper_bnds = [np.inf]*n
        # for k in self.angle_bounds:
        #     lower_bnds[k[0]] = k[1]
        #     upper_bnds[k[0]] = k[2]
        # bnds = Bounds(lower_bnds,upper_bnds)

        self.imux = np.array(self._imux)
        self.imuz = np.array(self._imuz)
        self.mmu = np.array(self._mmu)

        self.result = minimize(
            self.objective,
            2*np.pi*np.random.random(n),
            constraints=self.constraints,
            # bounds=((0,2*np.pi),)*n,
            options={"maxiter":2000},
            tol=1e-10
        )

    def print_result(self):
        fopt = self.result.x

        print(self.result.status, "|", self.result.message)
        foptd = (((self.result.x+np.pi)%(2*np.pi)-np.pi)*180/np.pi).tolist()
        print(self.result.fun)
        print(foptd)

        points = np.array([[self._X(fopt, t) - self._X(fopt, self.x_pos_anchor_tick) + self.x_pos_anchor, self._Z(fopt, t) - self._Z(fopt, self.z_pos_anchor_tick) + self.z_pos_anchor] for t in range(fopt.size+1)])
        print(points)
