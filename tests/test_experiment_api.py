"""Real HTTP/WebSocket smoke checks; isolated server, no browser dependency."""
import json
import socket
import subprocess
import sys
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
import pytest
from websockets.sync.client import connect
from src.core.engine import ROOT


@pytest.fixture(scope='module')
def server():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    process = subprocess.Popen([sys.executable,'-m','uvicorn','backend.app:app','--host','127.0.0.1','--port',str(port)],
                               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f'http://127.0.0.1:{port}'
    try:
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail('API process exited during startup')
            try:
                with urlopen(base+'/api/health',timeout=1) as response:
                    assert response.status == 200
                break
            except URLError:
                time.sleep(.1)
        else:
            pytest.fail('API startup timed out')
        yield base
    finally:
        process.terminate()
        process.wait(timeout=10)


def request(base, path, data=None):
    req = Request(base+path, data=None if data is None else json.dumps(data).encode(),
                  headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=5) as response:
        return json.load(response)


def test_reset_validation_and_live_telemetry(server):
    initial = request(server,'/api/experiment')
    assert {'kalman','none','kf_cv','ekf_angular','ukf_angular','akf_r'} <= set(initial['algorithms']['estimator'])
    config = initial['config']
    config['estimator'] = 'none'
    config['seed'] = 42
    assert request(server,'/api/experiment/reset',config)['config']['seed'] == 42
    with connect(server.replace('http:','ws:')+'/ws/telemetry',open_timeout=5) as socket:
        data = json.loads(socket.recv(timeout=5))
        assert data['estimator'] == 'none'
        assert data['predicted_pixel'] is None
        assert data['measurement_valid']
    config['estimator'] = 'ukf'
    with pytest.raises(HTTPError) as error:
        request(server,'/api/experiment/reset',config)
    assert error.value.code == 422
    assert 'Unavailable' in error.value.read().decode()
    assert request(server,'/api/experiment')['config']['estimator'] == 'none'
    assert request(server,'/api/experiment/metrics')['summary']['frame_count'] > 0


def test_mjpeg_contains_real_jpeg(server):
    with urlopen(server+'/api/camera.mjpeg',timeout=5) as response:
        assert 'multipart/x-mixed-replace' in response.headers['Content-Type']
        content = response.read(1024)
        assert b'Content-Type: image/jpeg' in content
        assert b'\xff\xd8' in content
    with urlopen(server+'/api/camera-raw.mjpeg',timeout=5) as response:
        content=response.read(1024)
        assert b'Content-Type: image/jpeg' in content and b'\xff\xd8' in content


def test_final_evidence_controls_export_and_report(server):
    registry=request(server,'/api/evidence/registry')
    assert {'fixed','uniform','normal'} <= set(registry['sampling_modes'])
    assert 'disturbance.vibration.stochastic_sigma_rad' in registry['variables']
    config=request(server,'/api/experiment')['config']
    applied=request(server,'/api/evidence/preset',{'config':config,'level':'LOW'})
    assert applied['config']['disturbance']['preset_level']=='LOW'
    changed=request(server,'/api/evidence/change',{'path':'camera.noise_sigma','value':13})
    assert changed['event']['after']==13 and changed['config']['camera']['noise_sigma']==13
    live=request(server,'/api/evidence/live')
    assert {'waterfall','running','resolved_config','optical_link'} <= set(live)
    csv_payload={'runs':[{'run_id':'demo','batch_id':'batch','metrics':{'rmse_tracking_error_px':1.0}}]}
    csv_request=Request(server+'/api/evidence/export',data=json.dumps({'format':'csv_runs','payload':csv_payload}).encode(),headers={'Content-Type':'application/json'})
    with urlopen(csv_request,timeout=10) as response:
        assert response.headers['Content-Type'].startswith('text/csv') and b'run_id' in response.read(2048)
    with urlopen(server+'/api/evidence/report',timeout=10) as response:
        assert response.headers['Content-Type'].startswith('text/html') and b'PAT Evidence Report' in response.read()


def test_headless_batch_api_persists_individual_runs(server):
    config=request(server,'/api/experiment')['config']
    config['camera'].update(width_px=64,height_px=48,focal_length_px=50)
    config.update(fps=10,time_scale=2,duration_s=.2)
    result=request(server,'/api/experiments/monte-carlo',{'config':config,'runs':2,'base_seed':42,'duration_s':.2})
    assert result['aggregate']['runs']==2 and len(result['results'])==2
    history=request(server,'/api/experiment')
    assert history['catalogue']['pipelines']['RECOMMENDED_HYBRID']['status']=='evidence gate failed'


def test_vision_and_estimator_lab_api_use_shared_data(server):
    vision=request(server,'/api/vision/compare-frame',{'algorithms':['binary_centroid','weighted_centroid'],'benchmark_mode':'FULL_FRAME'})
    assert len(vision['results'])==2
    assert len({item['input_checksum'] for item in vision['results']})==1
    config=request(server,'/api/experiment')['config']
    config.update(fps=10,duration_s=.2);config['camera'].update(width_px=80,height_px=60,focal_length_px=65)
    replay=request(server,'/api/estimation/replay',{'config':config,'duration_s':.2,'estimators':['kf_cv','ekf_angular','ukf_angular','akf_r']})
    assert replay['shared_measurement_samples']==2
    assert set(replay['estimators'])=={'kf_cv','ekf_angular','ukf_angular','akf_r'}
