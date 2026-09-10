"""Prompt 8 truth-free acquisition and reacquisition search strategies."""
from __future__ import annotations

import math
import numpy as np

from src.core.contracts import SearchCommand, SearchInput


class SearchBase:
    version="search-1.0.0"; name="hold"
    def __init__(self,config,streams=None): self.config=config; self.settings=config.acquisition
    def reset(self): pass
    def metadata(self): return {"name":self.name,"version":self.version,"truth_free":True}
    def point_command(self,state,request,gain=2.0):
        if not state.valid: return None
        cx,cy=request.image_centre; f=request.focal_length_px
        return gain*math.atan((state.x-cx)/f),gain*math.atan((cy-state.y)/f)
    def command(self,request,pan,tilt,phase,diagnostics=None,fallback=False):
        limit=min(request.actuator_limits.max_rate_rad_s,self.settings.search_command_limit_rad_s)
        clipped=np.clip(np.nan_to_num([pan,tilt]),-limit,limit)
        return SearchCommand(float(clipped[0]),float(clipped[1]),self.name,phase,fallback,
                             {**(diagnostics or {}),"command_limit_rad_s":limit})


class HoldSearch(SearchBase):
    name="hold"
    def search(self,request): return self.command(request,0,0,"HOLD")


class LastKnownSearch(SearchBase):
    name="last_known"
    def search(self,request):
        value=self.point_command(request.last_known_state,request)
        if value is None: return self.command(request,0,0,"HOLD",{"failure_reason":"NO_LAST_KNOWN"},True)
        return self.command(request,*value,"LAST_KNOWN",{"last_known_age_s":request.elapsed_search_s})


class RasterSearch(SearchBase):
    name="raster"
    def search(self,request):
        period=self.settings.raster_period_s; phase=(request.elapsed_search_s%period)/period
        row=int((request.elapsed_search_s//period)%self.settings.raster_rows)
        direction=1 if row%2==0 else -1
        pan=direction*self.settings.scan_rate_rad_s
        tilt=(2*(row/(self.settings.raster_rows-1)) - 1)*self.settings.scan_rate_rad_s*.35
        return self.command(request,pan,tilt,"RASTER",{"row":row,"phase":phase,"direction":direction})


class SpiralSearch(SearchBase):
    name="spiral"
    def search(self,request):
        angle=self.settings.spiral_angular_rate_rad_s*request.elapsed_search_s
        radial=min(self.settings.scan_rate_rad_s,
                   self.settings.spiral_radial_rate_rad_s*request.elapsed_search_s)
        return self.command(request,radial*math.cos(angle),radial*math.sin(angle),"SPIRAL",
                            {"spiral_angle_rad":angle,"radial_rate_rad_s":radial})


class PredictedPointSearch(SearchBase):
    name="predicted_point"
    def search(self,request):
        value=self.point_command(request.predicted_state,request)
        stale=request.elapsed_search_s>self.settings.max_prediction_age_s
        if value is None or stale:
            fallback=LastKnownSearch(self.config).search(request)
            return SearchCommand(fallback.pan,fallback.tilt,self.name,"LAST_KNOWN_FALLBACK",True,
                                 {**fallback.diagnostics,"failure_reason":"INVALID_OR_STALE_PREDICTION"})
        return self.command(request,*value,"PREDICTED_POINT",{"prediction_age_s":request.elapsed_search_s})


class CovarianceSearch(SearchBase):
    name="covariance_search"
    def search(self,request):
        covariance=request.uncertainty
        if covariance is None:
            fallback=SpiralSearch(self.config).search(request)
            return SearchCommand(fallback.pan,fallback.tilt,self.name,"SPIRAL_FALLBACK",True,
                                 {**fallback.diagnostics,"failure_reason":"NO_COVARIANCE"})
        matrix=np.asarray(covariance,dtype=float)[:2,:2]
        values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
        values=np.maximum(values,1e-9); major=vectors[:,int(np.argmax(values))]
        angle=self.settings.spiral_angular_rate_rad_s*request.elapsed_search_s
        anisotropy=float(math.sqrt(max(values)/min(values)))
        unit=vectors@np.array([math.cos(angle),math.sin(angle)])
        rate=self.settings.scan_rate_rad_s*np.array([unit[0],unit[1]/max(1,anisotropy*.5)])
        return self.command(request,rate[0],rate[1],"COVARIANCE_SEARCH",
                            {"eigenvalues_px2":values.tolist(),"major_axis":major.tolist(),"anisotropy":anisotropy})


class PredictiveCovarianceSearch(CovarianceSearch):
    name="predictive_covariance"
    def search(self,request):
        scan=super().search(request); point=self.point_command(request.predicted_state,request)
        if point is None:
            return SearchCommand(scan.pan,scan.tilt,self.name,scan.phase,True,
                                 {**scan.diagnostics,"failure_reason":"INVALID_PREDICTION"})
        return self.command(request,point[0]+.45*scan.pan,point[1]+.45*scan.tilt,
                            "PREDICTIVE_COVARIANCE",scan.diagnostics,scan.fallback_used)


class HybridSearch(SearchBase):
    name="hybrid"; version="hybrid-search-1.0.0"
    def search(self,request):
        t=request.elapsed_search_s
        if t<self.settings.hold_duration_s: strategy=HoldSearch(self.config)
        elif t<self.settings.hold_duration_s+self.settings.last_known_duration_s: strategy=LastKnownSearch(self.config)
        elif request.predicted_state.valid and t<self.settings.max_prediction_age_s+self.settings.last_known_duration_s:
            strategy=PredictiveCovarianceSearch(self.config)
        elif request.uncertainty is not None: strategy=CovarianceSearch(self.config)
        else: strategy=RasterSearch(self.config)
        result=strategy.search(request)
        return SearchCommand(result.pan,result.tilt,self.name,f"HYBRID_{result.phase}",result.fallback_used,
                             {**result.diagnostics,"sub_strategy":strategy.name})


class NoSearch(SearchBase):
    name="none"
    def search(self,request): return None


BasicReacquisition=HybridSearch
