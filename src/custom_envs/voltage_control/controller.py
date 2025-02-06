# env controller (droop control)
from custom_envs.controllers import Controller
from custom_envs.voltage_control.utils import VoltageControlObservationManager

import torch

class DroopControl:
    def __init__(self, q_u_thresholds, p_u_thresholds=None):
        """
        Implementation of the Sequential Droop Control from the paper: https://www.sciencedirect.com/science/article/abs/pii/S037877962030729X
        Parameters:
            q_u_thresholds: the key thresholds for volt-var droop control. q[1] to q[2] is deadband.
                            q[0] is where Q_max is set, q[3] is where -Q_max ist set.
            p_u_threshold: The key voltage thresholds to curtail p. If linear interpolation between max and min p between thresholds.
                           If None the p-curtailment is turned off.
        """
        assert len(q_u_thresholds) == 4, "You have to provide four thresholds for the Q(u) droop control"
        if p_u_thresholds is not None:
            assert len(p_u_thresholds) == 2, "You have to provide two thresholds for the P(u) droop control"

        self.q_u_thresholds = q_u_thresholds
        self.p_u_thresholds = p_u_thresholds

    def get_setpoint(self, v_pu: torch.Tensor):
        p = torch.ones_like(v_pu)
        q = torch.zeros_like(v_pu)  # initialize q array with same shape as v_pu

        if self.p_u_thresholds is not None:
            # linear interpolation between (p_u_thresholds[0], 1) and (p_u_thresholds[1], -1)
            mask_p = (v_pu > self.p_u_thresholds[0]) & (v_pu <= self.p_u_thresholds[1])
            p = torch.where(mask_p, 1 - 2 * (v_pu - self.p_u_thresholds[0])/(self.p_u_thresholds[1] - self.p_u_thresholds[0]), p)

            # minimum p for >= p_u_threshold
            mask_p2 = v_pu > self.p_u_thresholds[1]
            p = torch.where(mask_p2, -1, p)
        
        # maximum q for v_pu <= threshold[0]
        q = torch.where(v_pu <= self.q_u_thresholds[0], 1, q)
        
        # linear interpolation between threshold[0] and threshold[1]
        mask1 = (self.q_u_thresholds[0] < v_pu) & (v_pu <= self.q_u_thresholds[1])
        q = torch.where(mask1, 1 - (v_pu - self.q_u_thresholds[0])/(self.q_u_thresholds[1] - self.q_u_thresholds[0]), q)
        
        # zero q in deadband
        mask2 = (self.q_u_thresholds[1] < v_pu) & (v_pu <= self.q_u_thresholds[2])
        q = torch.where(mask2, 0, q)
        
        # linear interpolation between threshold[2] and threshold[3]
        mask3 = (self.q_u_thresholds[2] < v_pu) & (v_pu <= self.q_u_thresholds[3])
        q = torch.where(mask3,  -1 * (v_pu - self.q_u_thresholds[2]) / (self.q_u_thresholds[3] - self.q_u_thresholds[2]), q)
        
        # minimum q for v_pu >= threshold[3]
        q = torch.where(v_pu > self.q_u_thresholds[3], -1, q)
        
        return p, q


class CentralizedDroopController(Controller):
    # assumes that every observable bus is also controllable

    def __init__(self, obs_man: VoltageControlObservationManager, q_u_thresholds=(0.93, 0.95, 1.05, 1.07), p_u_thresholds=None):
        super().__init__(obs_man=obs_man)
        self.droop_controller = DroopControl(q_u_thresholds=q_u_thresholds, p_u_thresholds=p_u_thresholds)

    def get_action(self, state, greedy=True):
        if len(state.shape) == 1:
            p, q = self.droop_controller.get_setpoint(state[::4])
            #action = torch.stack((p, q), dim=1).to(p.device).flatten()
            action = torch.empty(size=(p.shape[0]*2,), dtype=p.dtype, device=p.device)
            # interleave p and q
            action[0::2] = p
            action[1::2] = q
        else:  # batch of states
            p, q = self.droop_controller.get_setpoint(state[:, ::4])
            action = torch.empty(size=(p.shape[0], p.shape[1] * 2), dtype=p.dtype, device=p.device)
            # interleave p and q
            action[:, 0::2] = p
            action[:, 1::2] = q

        return action
    
class DummyController(Controller):
    # always outputs p=p_val q=q_val
    def __init__(self, obs_man: VoltageControlObservationManager, p_val=1, q_val=0):
        super().__init__(obs_man=obs_man)

        assert -1 <= p_val <= 1, 'pval needs to be in [-1,1]'
        assert -1 <= q_val <= 1, 'qval needs to be in [-1,1]'
        self.p_val = p_val
        self.q_val = q_val

    def get_action(self, state, greedy=True):
        if len(state.shape) == 1:
            p = torch.ones_like(state[::4]) * self.p_val
            q = torch.ones_like(p) * self.q_val
            action = torch.empty(size=(p.shape[0]*2,), dtype=p.dtype, device=p.device)
            
            action[0::2] = p
            action[1::2] = q

        else:
            p = torch.ones_like(state[:, ::4]) * self.p_val
            q = torch.ones_like(p) * self.q_val
            action = torch.empty(size=(p.shape[0], p.shape[1] * 2), dtype=p.dtype, device=p.device)

            action[:, 0::2] = p
            action[:, 1::2] = q

        return action
    
        


