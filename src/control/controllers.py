"""Angular-domain controllers sharing one truth-free contract and one actuator."""
from __future__ import annotations

from time import perf_counter
import math
import numpy as np

from src.control.pid import PIDAxis
from src.core.contracts import (ActuatorLimits, Command, ControlCommand, ControllerInput,
                                GimbalState, PredictedState, TrackingState)


def _angles_and_rates(request: ControllerInput):
    current = request.current_state_estimate
    source = request.predicted_state if request.predicted_state.valid else current
    if not current.valid:
        return None
    cx, cy = request.target_pixel
    fx = request.focal_length_px
    ex = math.atan((source.x - cx) / fx)
    ey = math.atan((cy - source.y) / fx)
    vx = source.vx / fx
    vy = -source.vy / fx
    return ex, ey, vx, vy


def _bounded(name: str, version: str, request: ControllerInput, pan: float, tilt: float,
             diagnostics=None, fallback=False) -> ControlCommand:
    limit = request.actuator_limits.max_rate_rad_s
    if not np.all(np.isfinite([pan, tilt])):
        return ControlCommand(controller_name=name, controller_version=version, fallback_used=True,
                              diagnostics={"failure_reason": "INVALID_NUMERICS"})
    clipped = np.clip([pan, tilt], -limit, limit)
    return ControlCommand(float(clipped[0]), float(clipped[1]), controller_name=name,
                          controller_version=version, saturated=bool(np.any(clipped != [pan, tilt])),
                          fallback_used=fallback, diagnostics=diagnostics or {})


class PIDController:
    version = "pid-angular-2.0.0"
    name = "pid"

    def __init__(self, config, streams=None):
        p = config.pid
        self.config = config
        self.pan = PIDAxis(p.kp, p.ki, p.kd, p.max_rate_rad_s, p.integral_limit,
                           derivative_filter_tau_s=p.derivative_filter_tau_s)
        self.tilt = PIDAxis(p.kp, p.ki, p.kd, p.max_rate_rad_s, p.integral_limit,
                            derivative_filter_tau_s=p.derivative_filter_tau_s)
        self.derivative_source = p.derivative_source

    def reset(self):
        self.pan.reset(); self.tilt.reset()

    def _legacy_request(self, state, gimbal, centre, dt):
        predicted = state if isinstance(state, PredictedState) else PredictedState(
            valid=state.valid, horizon_s=0.0, x=state.x, y=state.y, vx=state.vx, vy=state.vy,
            covariance=state.covariance)
        current = TrackingState(x=state.x, y=state.y, vx=state.vx, vy=state.vy,
                                timestamp=getattr(state, "timestamp", 0.0), valid=state.valid,
                                covariance=state.covariance)
        return ControllerInput(current, predicted, centre, gimbal,
                               ActuatorLimits(self.config.pid.max_rate_rad_s,
                                              self.config.disturbance.gimbal.max_acceleration_rad_s2,
                                              *self.config.disturbance.gimbal.position_limit_rad),
                               self.config.camera.focal_length_px, dt,
                               getattr(state, "timestamp", 0.0), predicted.covariance, current.covariance)

    def compute(self, request: ControllerInput, gimbal=None, centre=None, dt=None) -> ControlCommand:
        if not isinstance(request, ControllerInput):
            request = self._legacy_request(request, gimbal, centre, dt)
        values = _angles_and_rates(request)
        if values is None:
            return ControlCommand(controller_name=self.name, controller_version=self.version,
                                  fallback_used=True, diagnostics={"failure_reason": "INVALID_ESTIMATE"})
        ex, ey, vx, vy = values
        dx = vx if self.derivative_source == "estimated_rate" else None
        dy = vy if self.derivative_source == "estimated_rate" else None
        pan = self.pan.step(ex, request.dt, dx)
        tilt = self.tilt.step(ey, request.dt, dy)
        return _bounded(self.name, self.version, request, pan, tilt, {
            "angular_error_rad": [ex, ey], "angular_rate_error_rad_s": [vx, vy],
            "derivative_source": self.derivative_source, "anti_windup": "conditional_integration",
            "derivative_filter_tau_s": self.config.pid.derivative_filter_tau_s,
            "pid_terms_pan": list(self.pan.terms), "pid_terms_tilt": list(self.tilt.terms)})

    def metadata(self):
        p = self.config.pid
        return {"name": self.name, "version": self.version, "kp": p.kp, "ki": p.ki, "kd": p.kd,
                "anti_windup": "conditional_integration", "derivative_source": p.derivative_source,
                "derivative_filter_tau_s": p.derivative_filter_tau_s}


class FeedForwardPIDController(PIDController):
    version = "ff-pid-angular-2.0.0"
    name = "ff_pid"

    def __init__(self, config, streams=None):
        super().__init__(config, streams)
        self.settings = config.ff_pid
        self.last_feedback = ControlCommand(controller_name="pid")
        self.last_feedforward = ControlCommand(controller_name="feedforward")
        self.last_uncertainty_weight = 0.0

    def compute(self, request: ControllerInput, gimbal=None, centre=None, dt=None) -> ControlCommand:
        legacy = not isinstance(request, ControllerInput)
        if legacy:
            legacy_state = request
            if not legacy_state.valid:
                return Command()
            feedback_legacy = Command(self.pan.step(legacy_state.x-centre[0], dt),
                                      self.tilt.step(centre[1]-legacy_state.y, dt))
            covariance = legacy_state.covariance
            trace = float(np.trace(np.asarray(covariance)[:2, :2])) if covariance is not None else 0.0
            weight = float(np.clip(1.0/(1.0+trace/self.settings.uncertainty_scale_px2),
                                   self.settings.minimum_weight, 1.0))
            ff_pan = self.settings.gain*weight*legacy_state.vx/self.config.camera.focal_length_px
            ff_tilt = -self.settings.gain*weight*legacy_state.vy/self.config.camera.focal_length_px
            limit = self.config.pid.max_rate_rad_s
            return Command(float(np.clip(feedback_legacy.pan+ff_pan,-limit,limit)),
                           float(np.clip(feedback_legacy.tilt+ff_tilt,-limit,limit)))
        feedback = super().compute(request)
        predicted = request.predicted_state
        if not predicted.valid:
            result = ControlCommand(feedback.pan, feedback.tilt, controller_name=self.name,
                                    controller_version=self.version, fallback_used=True,
                                    diagnostics={**feedback.diagnostics, "failure_reason": "INVALID_PREDICTION"})
            return result
        covariance = request.prediction_uncertainty
        trace = float(np.trace(np.asarray(covariance)[:2, :2])) if covariance is not None else 0.0
        weight = float(np.clip(1.0 / (1.0 + trace / self.settings.uncertainty_scale_px2),
                               self.settings.minimum_weight, 1.0))
        ff_pan = self.settings.gain * weight * predicted.vx / request.focal_length_px
        ff_tilt = -self.settings.gain * weight * predicted.vy / request.focal_length_px
        self.last_feedback = feedback
        self.last_feedforward = ControlCommand(ff_pan, ff_tilt, controller_name="feedforward")
        self.last_uncertainty_weight = weight
        result = _bounded(self.name, self.version, request, feedback.pan + ff_pan, feedback.tilt + ff_tilt,
                          {**feedback.diagnostics, "feedforward_rad_s": [ff_pan, ff_tilt],
                           "uncertainty_weight": weight, "uncertainty_mapping": "1/(1+trace(Sigma)/scale)"})
        return result

    def metadata(self):
        value = super().metadata()
        value.update(name=self.name, version=self.version, K_ff=self.settings.gain,
                     uncertainty_scale_px2=self.settings.uncertainty_scale_px2)
        return value


class GainScheduledPIDController(PIDController):
    version = "gain-scheduled-pid-1.0.0"
    name = "gain_scheduled_pid"

    def __init__(self, config, streams=None):
        super().__init__(config, streams)
        self.settings = config.gain_scheduled_pid
        self.base = np.array([config.pid.kp, config.pid.ki, config.pid.kd], dtype=float)
        self.schedule_value = 0.0

    def _scale(self, dynamic: float, covariance) -> tuple[np.ndarray, str]:
        low, high = self.settings.low_rate_rad_s, self.settings.high_rate_rad_s
        t = float(np.clip((dynamic - low) / max(high - low, 1e-12), 0, 1))
        target = np.asarray(self.settings.low_gain_scale) if t <= .5 else np.asarray(self.settings.medium_gain_scale) + (2*t-1)*(np.asarray(self.settings.high_gain_scale)-np.asarray(self.settings.medium_gain_scale))
        if t <= .5: target = np.asarray(self.settings.low_gain_scale) + 2*t*(np.asarray(self.settings.medium_gain_scale)-np.asarray(self.settings.low_gain_scale))
        if covariance is not None:
            uncertainty = float(np.trace(np.asarray(covariance)[:2, :2]))
            target *= 1.0 / (1.0 + uncertainty / self.settings.uncertainty_scale_px2)
        self.schedule_value += self.settings.smoothing * (t - self.schedule_value)
        region = "LOW_DYNAMIC" if self.schedule_value < .33 else "MEDIUM_DYNAMIC" if self.schedule_value < .67 else "HIGH_DYNAMIC"
        return target, region

    def compute(self, request: ControllerInput) -> ControlCommand:
        values = _angles_and_rates(request)
        if values is None:
            fallback = super().compute(request)
            return ControlCommand(fallback.pan, fallback.tilt, controller_name=self.name,
                                  controller_version=self.version, fallback_used=True,
                                  diagnostics={"failure_reason": "INVALID_SCHEDULE_INPUT"})
        ex, ey, vx, vy = values
        scale, region = self._scale(math.hypot(vx, vy), request.estimator_covariance)
        for axis in (self.pan, self.tilt): axis.kp, axis.ki, axis.kd = self.base * scale
        output = super().compute(request)
        return ControlCommand(output.pan, output.tilt, controller_name=self.name, controller_version=self.version,
                              saturated=output.saturated, diagnostics={**output.diagnostics,
                              "gain_region": region, "gain_scale": scale.tolist(),
                              "schedule_value": self.schedule_value})

    def metadata(self):
        return {"name": self.name, "version": self.version, "schedule": "LOS-rate smooth interpolation",
                "low_rate_rad_s": self.settings.low_rate_rad_s, "high_rate_rad_s": self.settings.high_rate_rad_s}


def solve_discrete_riccati(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, iterations=200):
    p = q.copy()
    for _ in range(iterations):
        middle = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        next_p = a.T @ p @ a - a.T @ p @ b @ middle + q
        if np.max(np.abs(next_p - p)) < 1e-11: p = next_p; break
        p = next_p
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return p, k


class LQRController:
    version = "lqr-gimbal-1.0.0"
    name = "lqr"

    def __init__(self, config, streams=None):
        self.config = config
        dt, tau = 1/config.fps, config.pid.actuator_response_time_s
        alpha = 1-math.exp(-dt/max(tau, 1e-9))
        self.a = np.array([[1.0, -dt], [0.0, 1-alpha]])
        self.b = np.array([[0.0], [alpha]])
        self.q = np.diag([config.lqr.tracking_error_weight, config.lqr.angular_rate_weight])
        self.r = np.array([[config.lqr.control_effort_weight]])
        self.p, self.k = solve_discrete_riccati(self.a, self.b, self.q, self.r, config.lqr.riccati_iterations)

    def reset(self): pass

    def compute(self, request: ControllerInput) -> ControlCommand:
        values = _angles_and_rates(request)
        if values is None:
            return ControlCommand(controller_name=self.name, controller_version=self.version, fallback_used=True,
                                  diagnostics={"failure_reason": "INVALID_STATE", "fallback_to": "pid"})
        ex, ey, target_pan_rate, target_tilt_rate = values
        pan_state = np.array([ex, request.current_gimbal_state.pan_rate-target_pan_rate])
        tilt_state = np.array([ey, request.current_gimbal_state.tilt_rate-target_tilt_rate])
        pan = target_pan_rate - float((self.k @ pan_state).item())
        tilt = target_tilt_rate - float((self.k @ tilt_state).item())
        return _bounded(self.name, self.version, request, pan, tilt, {"state_pan":pan_state.tolist(),
                        "state_tilt":tilt_state.tolist(), "gain":self.k.ravel().tolist(), "reference_feedforward":True})

    def metadata(self):
        return {"name":self.name,"version":self.version,"state":"[LOS_error_rad, gimbal_rate-reference_rate]",
                "A":self.a.tolist(),"B":self.b.tolist(),"Q":self.q.tolist(),"R":self.r.tolist(),"K":self.k.tolist()}


class MPCController(LQRController):
    version = "mpc-projected-qp-1.0.0"
    name = "mpc"

    def __init__(self, config, streams=None):
        super().__init__(config, streams)
        self.settings = config.mpc
        self.fallback = FeedForwardPIDController(config, streams) if config.mpc.fallback_controller == "ff_pid" else PIDController(config, streams)
        self.previous = np.zeros(2)

    def reset(self):
        self.previous[:] = 0; self.fallback.reset()

    def _axis(self, state, reference, request, previous_command):
        n, m = self.settings.prediction_horizon_steps, self.settings.control_horizon_steps
        sx=np.zeros((2*n,2)); su=np.zeros((2*n,m))
        for i in range(n):
            ai=np.linalg.matrix_power(self.a,i+1); sx[2*i:2*i+2]=ai
            for j in range(min(i+1,m)): su[2*i:2*i+2,j]=(np.linalg.matrix_power(self.a,i-j)@self.b).ravel()
        q=np.diag([self.settings.tracking_error_weight,self.settings.angular_rate_weight])
        qbar=np.kron(np.eye(n),q); rbar=np.eye(m)*self.settings.control_effort_weight
        d=np.eye(m)-np.eye(m,k=-1); d[0]=0
        h=su.T@qbar@su+rbar+self.settings.command_change_weight*(d.T@d)+np.eye(m)*1e-9
        f=su.T@qbar@sx@state
        u=np.linalg.solve(h,-f)+reference
        limit=request.actuator_limits.max_rate_rad_s
        u=np.clip(u,-limit,limit)
        acceleration=request.actuator_limits.max_acceleration_rad_s2
        if acceleration is not None:
            for i in range(m):
                prior=previous_command if i==0 else u[i-1]
                u[i]=np.clip(u[i],prior-acceleration*request.dt,prior+acceleration*request.dt)
        return float(u[0])

    def compute(self, request: ControllerInput) -> ControlCommand:
        started=perf_counter(); values=_angles_and_rates(request)
        try:
            if values is None: raise ValueError("INVALID_STATE")
            ex,ey,rx,ry=values
            pan=self._axis(np.array([ex,request.current_gimbal_state.pan_rate-rx]),rx,request,
                           self.previous[0])
            tilt=self._axis(np.array([ey,request.current_gimbal_state.tilt_rate-ry]),ry,request,
                            self.previous[1])
            self.previous=np.array([pan,tilt])
            elapsed=(perf_counter()-started)*1000
            if elapsed>self.settings.solver_timeout_ms: raise TimeoutError(f"MPC_TIMEOUT_{elapsed:.3f}ms")
            return _bounded(self.name,self.version,request,pan,tilt,{"solver":"dense convex QP",
                            "horizon_steps":self.settings.prediction_horizon_steps,"solver_latency_ms":elapsed})
        except Exception as exc:
            fallback=self.fallback.compute(request)
            return ControlCommand(fallback.pan,fallback.tilt,controller_name=self.name,controller_version=self.version,
                                  saturated=fallback.saturated,fallback_used=True,
                                  diagnostics={"failure_reason":str(exc),"fallback_to":self.settings.fallback_controller})

    def metadata(self):
        return {"name":self.name,"version":self.version,"status":"EXPERIMENTAL","solver":"NumPy dense convex QP",
                "prediction_horizon_steps":self.settings.prediction_horizon_steps,
                "control_horizon_steps":self.settings.control_horizon_steps,
                "cost_weights":{"tracking":self.settings.tracking_error_weight,"rate":self.settings.angular_rate_weight,
                                "control":self.settings.control_effort_weight,"delta_u":self.settings.command_change_weight},
                "timeout_ms":self.settings.solver_timeout_ms,"fallback":self.settings.fallback_controller}
