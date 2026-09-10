"""Adapters preserve current algorithms and provide scenario motion/reacquisition."""
import math
import numpy as np
from src.core.contracts import PhysicalState, Measurement, TrackingState, Command
from src.core.errors import PhysicalErrorSystem
from src.physics.cw import AnalyticalClohessyWiltshire, ClohessyWiltshire
from src.tracking.kalman import ImageKalmanFilter
from src.control.pid import PIDAxis
from src.simulation.stationary_demo import detect_beacon


class CWMotion:
    version = "cw-rk4-1.0.0"
    def __init__(self, config, streams):
        o, rng = config.orbit, streams['motion']
        position = np.array(o.initial_hill_position_m)+rng.normal(0,config.disturbance.navigation.initial_position_sigma_m,3)
        velocity = np.array(o.initial_hill_velocity_m_s)+rng.normal(0,config.disturbance.navigation.initial_velocity_sigma_m_s,3)
        self.model = ClohessyWiltshire(math.sqrt(o.earth_mu_m3_s2/(o.earth_radius_m+o.altitude_m)**3),position,velocity)

    def update(self, timestamp, dt):
        p,v = self.model.step(dt)
        return PhysicalState(p,v,"hill")


class AnalyticalCWMotion(CWMotion):
    version = "cw-analytical-1.0.0"
    def __init__(self,config,streams):
        o,rng=config.orbit,streams['motion']
        position=np.array(o.initial_hill_position_m)+rng.normal(0,config.disturbance.navigation.initial_position_sigma_m,3)
        velocity=np.array(o.initial_hill_velocity_m_s)+rng.normal(0,config.disturbance.navigation.initial_velocity_sigma_m_s,3)
        mean_motion=math.sqrt(o.earth_mu_m3_s2/(o.earth_radius_m+o.altitude_m)**3)
        self.model=AnalyticalClohessyWiltshire(mean_motion,position,velocity)


class KinematicMotion:
    version = "relative-kinematic-1.0.0"
    def __init__(self, config, streams):
        k, rng = config.kinematic, streams['motion']
        self.p = np.array(k.initial_position_m)+rng.normal(0,k.initial_position_sigma_m,3)
        self.v = np.array(k.velocity_m_s)+rng.normal(0,k.initial_velocity_sigma_m_s,3)
        self.rate, self.amplitude = k.angular_rate_rad_s,k.lateral_amplitude_m
        self.last_t = 0.0

    def update(self, timestamp, dt):
        # Nominal scenario kinematics, explicitly separate from orbital dynamics.
        self.p += self.v*dt
        if self.rate:
            self.p[1] += self.amplitude*(math.sin(self.rate*timestamp)-math.sin(self.rate*self.last_t))
        self.last_t = timestamp
        velocity = self.v.copy()
        velocity[1] += self.amplitude*self.rate*math.cos(self.rate*timestamp)
        return PhysicalState(self.p.copy(),velocity,"world")


class CentroidTracker:
    version = "opencv-centroid-1.0.0"
    def __init__(self, config, streams): self.threshold = config.camera.threshold
    def measure(self, frame, timestamp): return Measurement(detect_beacon(frame,self.threshold),timestamp)


class KalmanEstimator:
    version = "linear-kf-cv-1.0.0"
    def __init__(self, config, streams):
        self.filter = ImageKalmanFilter(config.kalman.process_noise,config.kalman.measurement_noise)
    def update(self, measurement, dt):
        state = self.filter.step(measurement.pixel,dt)
        if state is None: return TrackingState(timestamp=measurement.timestamp)
        return TrackingState(*map(float,state),timestamp=measurement.timestamp,valid=True,
                             covariance=self.filter.covariance.copy(),confidence=measurement.confidence)


class PassThroughEstimator:
    version = "passthrough-1.0.0"
    def __init__(self, config, streams): pass
    def update(self, measurement, dt):
        if not measurement.valid: return TrackingState(timestamp=measurement.timestamp)
        return TrackingState(*measurement.pixel,timestamp=measurement.timestamp,valid=True,confidence=measurement.confidence)


class NoPrediction:
    version = "none-1.0.0"
    def __init__(self, config, streams): pass
    def predict(self, state, horizon): return state


class PIDController:
    version = "pid-1.0.0"
    def __init__(self, config, streams):
        p = config.pid
        self.pan = PIDAxis(p.kp,p.ki,p.kd,p.max_rate_rad_s,p.integral_limit)
        self.tilt = PIDAxis(p.kp,p.ki,p.kd,p.max_rate_rad_s,p.integral_limit)
    def compute(self, state, gimbal, centre, dt):
        if not state.valid: return Command()
        return Command(self.pan.step(state.x-centre[0],dt),self.tilt.step(centre[1]-state.y,dt))


class NoReacquisition:
    version = "none-1.0.0"
    def __init__(self, config, streams): pass
    def search(self, last, uncertainty, time_since_loss): return None


class BasicReacquisition:
    """Deterministic expanding rate scan; intentionally coarse, not AI."""
    version = "basic-scan-1.0.0"
    def __init__(self, config, streams): self.limit = config.pid.max_rate_rad_s
    def search(self, last, uncertainty, time_since_loss):
        amplitude = min(self.limit,.025+.012*time_since_loss)
        return Command(amplitude*math.sin(2*time_since_loss),amplitude*.6*math.cos(1.3*time_since_loss))


DisturbanceStage = PhysicalErrorSystem
