"""Deterministic Gaussian far-field optical-link consequence model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def _power_dbm(power_w: float) -> float:
    return 10*math.log10(max(power_w,1e-300)/1e-3)


def _loss_db(factor: float) -> float:
    return -10*math.log10(max(factor,1e-300))


def _wrap(angle: float) -> float:
    return (angle+math.pi)%(2*math.pi)-math.pi


@dataclass(frozen=True)
class OpticalLinkResult:
    enabled: bool
    model_version: str
    range_m: float | None = None
    range_mode: str | None = None
    wavelength_nm: float | None = None
    pointing_error_x_rad: float | None = None
    pointing_error_y_rad: float | None = None
    pointing_error_rad: float | None = None
    lateral_offset_m: float | None = None
    beam_radius_m: float | None = None
    geometric_capture_factor: float | None = None
    atmospheric_transmission: float | None = None
    optical_efficiency_factor: float | None = None
    pointing_factor: float | None = None
    geometric_loss_db: float | None = None
    atmospheric_loss_db: float | None = None
    optical_loss_db: float | None = None
    pointing_loss_db: float | None = None
    ideal_received_power_w: float | None = None
    received_power_w: float | None = None
    received_power_dbm: float | None = None
    receiver_sensitivity_dbm: float | None = None
    link_margin_db: float | None = None
    link_available: bool = False
    link_state: str = "NOT_CONFIGURED"
    communication_snr_db: float | None = None
    communication_snr_status: str = "NOT_CONFIGURED"
    failure_cause: str | None = None

    def as_dict(self): return asdict(self)


class GaussianOpticalLink:
    """Far-field model where w is the 1/e² intensity radius."""
    version="gaussian-far-field-1.0.0"

    def __init__(self,config): self.config=config

    def evaluate(self,los_pan_rad: float,los_tilt_rad: float,optical_pan_rad: float,
                 optical_tilt_rad: float,scenario_range_m: float,
                 scenario_atmospheric_transmission: float=1.0) -> OpticalLinkResult:
        c=self.config
        if not c.enabled: return OpticalLinkResult(False,c.model_version)
        range_m=scenario_range_m if c.range_mode=="AUTO_FROM_SCENARIO" else c.manual_range_m
        atmosphere=(scenario_atmospheric_transmission if c.atmosphere_mode=="FROM_SCENARIO"
                    else c.manual_atmospheric_transmission)
        atmosphere=float(min(1,max(1e-12,atmosphere)))
        ex=_wrap(los_pan_rad-optical_pan_rad); ey=_wrap(los_tilt_rad-optical_tilt_rad)
        theta=math.hypot(ex,ey); divergence=c.beam_divergence_urad*1e-6
        beam_radius=max(range_m*divergence,1e-12); lateral=range_m*theta
        aperture_radius=c.receiver_aperture_m/2
        geometric=1-math.exp(-2*(aperture_radius/beam_radius)**2)
        pointing=math.exp(max(-700,-2*(lateral/beam_radius)**2))
        optical=c.transmitter_efficiency*c.receiver_efficiency
        ideal=c.transmit_power_w*geometric*optical*atmosphere
        received=ideal*pointing; dbm=_power_dbm(received); margin=dbm-c.receiver_sensitivity_dbm
        available=received>=1e-3*10**(c.receiver_sensitivity_dbm/10)
        if margin>=c.good_margin_db: state="GOOD_MARGIN"
        elif available: state="LINK_AVAILABLE"
        elif margin>=c.marginal_margin_db: state="MARGINAL"
        else: state="NO_LINK"
        losses={"POINTING_LOSS":_loss_db(pointing),"ATMOSPHERIC_ATTENUATION":_loss_db(atmosphere),
                "RANGE_LOSS":_loss_db(geometric)}
        important=[name for name,value in losses.items() if value>=3]
        if available: failure=None
        elif len(important)>1: failure="COMBINED"
        elif important: failure=important[0]
        else: failure="RECEIVER_THRESHOLD"
        snr=None if c.communication_noise_power_w is None else 10*math.log10(max(received,1e-300)/c.communication_noise_power_w)
        return OpticalLinkResult(True,c.model_version,range_m,c.range_mode,c.wavelength_nm,
                                 ex,ey,theta,lateral,beam_radius,geometric,atmosphere,optical,pointing,
                                 _loss_db(geometric),_loss_db(atmosphere),_loss_db(optical),_loss_db(pointing),
                                 ideal,received,dbm,c.receiver_sensitivity_dbm,margin,available,state,snr,
                                 "APPROXIMATE" if snr is not None else "NOT_CONFIGURED",failure)

    def metadata(self):
        c=self.config
        return {"model_version":c.model_version,"beam_radius_convention":"1/e^2 intensity radius",
                "range_mode":c.range_mode,"wavelength_nm":c.wavelength_nm,
                "transmit_power_w":c.transmit_power_w,"beam_divergence_urad":c.beam_divergence_urad,
                "receiver_aperture_m":c.receiver_aperture_m,"receiver_sensitivity_dbm":c.receiver_sensitivity_dbm,
                "ber":"NOT_CONFIGURED"}
