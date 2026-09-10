"""Pinhole camera; only rendered pixels cross the vision boundary."""
import math
import cv2
import numpy as np


class VirtualCamera:
    def __init__(self, config, streams):
        self.config = config
        rng = streams['environment']
        self.stars = tuple(zip(rng.integers(0,config.width_px,45),rng.integers(0,config.height_px,45),rng.integers(25,95,45)))

    @staticmethod
    def basis(pan,tilt):
        forward=np.array([math.sin(pan)*math.cos(tilt),math.sin(tilt),math.cos(pan)*math.cos(tilt)])
        right=np.array([math.cos(pan),0,-math.sin(pan)])
        return right,np.cross(forward,right),forward

    def project(self,target,gimbal):
        right,up,forward=self.basis(gimbal.pan,gimbal.tilt)
        x,y,z=float(target@right),float(target@up),float(target@forward)
        c=self.config
        if z<=.05: return float('nan'),float('nan'),z
        return c.width_px/2+c.focal_length_px*x/z,c.height_px/2-c.focal_length_px*y/z,z

    def render(self,scene):
        c=self.config
        frame=np.zeros((c.height_px,c.width_px,3),dtype=np.uint8)
        for x,y,b in self.stars: frame[y,x]=b
        if scene.beacon_pixel is not None and not scene.beacon_dropout:
            u,v=scene.beacon_pixel
            if -30<=u<c.width_px+30 and -30<=v<c.height_px+30:
                axes=(max(1.0,scene.beacon_radius_px*scene.ellipse_ratio),max(1.0,scene.beacon_radius_px))
                intensity=round(min(max(scene.beacon_intensity,0),255))
                # OpenCV fixed-point coordinates preserve the projected subpixel
                # centre instead of silently rounding simulator truth to integers.
                shift=4; scale=1<<shift
                cv2.ellipse(frame,(round(u*scale),round(v*scale)),
                            (round(axes[0]*scale),round(axes[1]*scale)),0,0,360,
                            (intensity,)*3,-1,cv2.LINE_AA,shift=shift)
        for u,v,intensity,radius in scene.distractors:
            cv2.circle(frame,(round(u),round(v)),round(radius),(round(intensity),)*3,-1,cv2.LINE_AA)
        return frame

    def annotate(self,frame,measurement,estimate):
        if measurement.valid: cv2.drawMarker(frame,tuple(np.rint(measurement.pixel).astype(int)),(0,255,90),cv2.MARKER_CROSS,22,2)
        if estimate.valid:
            centre=tuple(np.rint(estimate.pixel).astype(int))
            cv2.circle(frame,centre,12,(255,210,40),2,cv2.LINE_AA)
            ellipse=estimate.uncertainty_ellipse
            if ellipse is not None:
                axes=(max(1,round(ellipse['major_axis_px']/2)),max(1,round(ellipse['minor_axis_px']/2)))
                cv2.ellipse(frame,centre,axes,ellipse['orientation_deg'],0,360,(255,120,35),1,cv2.LINE_AA)
        c=self.config
        cv2.drawMarker(frame,(c.width_px//2,c.height_px//2),(50,80,255),cv2.MARKER_CROSS,28,1)
        cv2.putText(frame,"green: OpenCV  cyan: estimate  red: optical axis",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(220,230,240),1,cv2.LINE_AA)
