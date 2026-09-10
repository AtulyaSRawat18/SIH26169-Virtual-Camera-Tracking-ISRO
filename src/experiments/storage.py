"""Append-only SQLite store for versioned batches and individual runs."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3
import uuid
from src.core.config import SIMULATION_SCHEMA_VERSION,PHYSICS_MODEL_VERSION

DEFAULT_DB=Path(__file__).resolve().parents[2]/'data'/'experiments.sqlite3'


class ExperimentStore:
    def __init__(self,path=DEFAULT_DB):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self._create()
    def connect(self):
        con=sqlite3.connect(self.path); con.row_factory=sqlite3.Row; return con
    def _create(self):
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS batches(
              batch_id TEXT PRIMARY KEY, experiment_id TEXT, created_at TEXT, mode TEXT,
              base_seed INTEGER, requested_runs INTEGER, configuration_json TEXT,
              simulation_schema_version TEXT, physics_model_version TEXT, scenario_version TEXT);
            CREATE TABLE IF NOT EXISTS runs(
              run_id TEXT PRIMARY KEY, batch_id TEXT, pair_index INTEGER, variant TEXT, created_at TEXT,
              simulation_schema_version TEXT, physics_model_version TEXT, scenario_version TEXT,
              scenario TEXT, scenario_preset TEXT, algorithm_pipeline_json TEXT, algorithm_versions_json TEXT,
              base_seed INTEGER, run_seed INTEGER, subsystem_seeds_json TEXT, duration_s REAL,
              git_commit_hash TEXT,
              sampled_inputs_json TEXT, resolved_config_json TEXT, metrics_json TEXT,
              failure_cause TEXT, completion_status TEXT, telemetry_json TEXT,
              vision_metadata_json TEXT, estimator_metadata_json TEXT, measurement_quality_json TEXT,
              correction_metadata_json TEXT, predictor_metadata_json TEXT, controller_metadata_json TEXT,
              optical_metadata_json TEXT, sampling_metadata_json TEXT,
              FOREIGN KEY(batch_id) REFERENCES batches(batch_id));
            CREATE INDEX IF NOT EXISTS idx_runs_batch ON runs(batch_id);
            CREATE INDEX IF NOT EXISTS idx_runs_compat ON runs(simulation_schema_version,physics_model_version,scenario_version);
            CREATE TABLE IF NOT EXISTS lab_results(
              result_id TEXT PRIMARY KEY, created_at TEXT, kind TEXT,
              simulation_schema_version TEXT, physics_model_version TEXT, scenario_version TEXT,
              configuration_json TEXT, result_json TEXT);
            CREATE INDEX IF NOT EXISTS idx_lab_results_kind ON lab_results(kind,created_at);
            ''')
            columns={row[1] for row in db.execute('PRAGMA table_info(runs)')}
            migrations={'git_commit_hash':'TEXT','vision_metadata_json':'TEXT','estimator_metadata_json':'TEXT',
                        'measurement_quality_json':'TEXT','correction_metadata_json':'TEXT',
                        'predictor_metadata_json':'TEXT','controller_metadata_json':'TEXT',
                        'optical_metadata_json':'TEXT','sampling_metadata_json':'TEXT'}
            for name,kind in migrations.items():
                if name not in columns: db.execute(f'ALTER TABLE runs ADD COLUMN {name} {kind}')
    def create_batch(self,config,mode,base_seed,runs,experiment_id=None):
        batch_id=str(uuid.uuid4()); created=datetime.now(timezone.utc).isoformat()
        experiment_id=experiment_id or str(uuid.uuid4())
        with self.connect() as db:
            db.execute('INSERT INTO batches VALUES(?,?,?,?,?,?,?,?,?,?)',(batch_id,experiment_id,created,mode,base_seed,runs,
                       json.dumps(config.model_dump(mode='json'),sort_keys=True),config.simulation_schema_version,
                       config.physics_model_version,config.scenario_version))
        return batch_id
    def append_run(self,record):
        row=dict(record); row.setdefault('run_id',str(uuid.uuid4())); row.setdefault('created_at',datetime.now(timezone.utc).isoformat())
        json_fields=('algorithm_pipeline','algorithm_versions','subsystem_seeds','sampled_inputs','resolved_config','metrics','telemetry',
                     'vision_metadata','estimator_metadata','measurement_quality','correction_metadata',
                     'predictor_metadata','controller_metadata','optical_metadata','sampling_metadata')
        for name in json_fields: row[name+'_json']=None if row.get(name) is None else json.dumps(row[name],sort_keys=True)
        values=(row['run_id'],row['batch_id'],row.get('pair_index'),row.get('variant'),row['created_at'],row['simulation_schema_version'],
                row['physics_model_version'],row['scenario_version'],row['scenario'],row['scenario_preset'],row['algorithm_pipeline_json'],
                row['algorithm_versions_json'],row['base_seed'],row['run_seed'],row['subsystem_seeds_json'],row['duration_s'],row.get('git_commit_hash'),
                row['sampled_inputs_json'],row['resolved_config_json'],row['metrics_json'],row.get('failure_cause'),
                row.get('completion_status','COMPLETE'),row['telemetry_json'],row['vision_metadata_json'],
                row['estimator_metadata_json'],row['measurement_quality_json'],row['correction_metadata_json'],
                row['predictor_metadata_json'],row['controller_metadata_json'],row['optical_metadata_json'],row['sampling_metadata_json'])
        columns=('run_id','batch_id','pair_index','variant','created_at','simulation_schema_version','physics_model_version',
                 'scenario_version','scenario','scenario_preset','algorithm_pipeline_json','algorithm_versions_json','base_seed',
                 'run_seed','subsystem_seeds_json','duration_s','git_commit_hash','sampled_inputs_json','resolved_config_json',
                 'metrics_json','failure_cause','completion_status','telemetry_json','vision_metadata_json',
                 'estimator_metadata_json','measurement_quality_json','correction_metadata_json',
                 'predictor_metadata_json','controller_metadata_json','optical_metadata_json','sampling_metadata_json')
        with self.connect() as db:
            db.execute(f"INSERT INTO runs ({','.join(columns)}) VALUES ({','.join('?'*len(values))})",values)
        return row['run_id']
    def query_runs(self,filters=None,include_incompatible=False):
        filters=filters or {}; clauses=[]; args=[]
        columns={'run_id','scenario','scenario_preset','variant','batch_id','failure_cause','completion_status','physics_model_version','simulation_schema_version','scenario_version'}
        for key,value in filters.items():
            if key in columns: clauses.append(f'{key}=?'); args.append(value)
        if not include_incompatible:
            clauses.extend(['simulation_schema_version=?','physics_model_version=?']); args.extend([SIMULATION_SCHEMA_VERSION,PHYSICS_MODEL_VERSION])
        sql='SELECT * FROM runs'+((' WHERE '+' AND '.join(clauses)) if clauses else '')+' ORDER BY created_at DESC'
        with self.connect() as db: rows=db.execute(sql,args).fetchall()
        result=[]
        for source in rows:
            row=dict(source)
            for name in ('algorithm_pipeline','algorithm_versions','subsystem_seeds','sampled_inputs','resolved_config','metrics','telemetry',
                         'vision_metadata','estimator_metadata','measurement_quality','correction_metadata',
                         'predictor_metadata','controller_metadata','optical_metadata','sampling_metadata'):
                row[name]=json.loads(row.pop(name+'_json')) if row[name+'_json'] is not None else None
            for key,value in filters.items():
                if key in columns: continue
                if key in ('estimator','predictor','controller') and row['algorithm_pipeline'].get(key)!=value: break
            else: result.append(row)
        return result
    def list_batches(self):
        with self.connect() as db: return [dict(x) for x in db.execute('SELECT * FROM batches ORDER BY created_at DESC')]
    def append_lab_result(self,kind,config,result):
        result_id=str(uuid.uuid4()); created=datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute('INSERT INTO lab_results VALUES(?,?,?,?,?,?,?,?)',
                       (result_id,created,kind,config.simulation_schema_version,config.physics_model_version,
                        config.scenario_version,json.dumps(config.model_dump(mode='json'),sort_keys=True),
                        json.dumps(result,sort_keys=True)))
        return result_id
    def query_lab_results(self,kind=None,include_incompatible=False):
        clauses=[]; args=[]
        if kind is not None: clauses.append('kind=?'); args.append(kind)
        if not include_incompatible:
            clauses.extend(['simulation_schema_version=?','physics_model_version=?'])
            args.extend([SIMULATION_SCHEMA_VERSION,PHYSICS_MODEL_VERSION])
        sql='SELECT * FROM lab_results'+((' WHERE '+' AND '.join(clauses)) if clauses else '')+' ORDER BY created_at DESC'
        with self.connect() as db: rows=db.execute(sql,args).fetchall()
        result=[]
        for source in rows:
            row=dict(source); row['configuration']=json.loads(row.pop('configuration_json')); row['result']=json.loads(row.pop('result_json'))
            result.append(row)
        return result
