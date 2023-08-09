
class SimpleJob:
    def __init__(self, job_id, job_name, job_type, job_status, job_params, experiment_id, dataset,
                 job_logs):
        self.job_id=job_id
        self.name=job_name
        self.job_type=job_type
        self.status=job_status
        self.parameters=job_params
        self.experiment_id=experiment_id
        self.dataset=dataset
        self.job_logs=job_logs
        pass
    
    @staticmethod
    def compute_job_to_simple_job(job):
        params = str(job['job_kwargs']['kwargs']['params'])
        if job['job_kwargs']['kwargs']['job_type'].split()[0] != 'train_model':
            params = f"{params}\nTraining Parameters: {job['job_kwargs']['kwargs']['train_params']}"
        return SimpleJob(
            job['uid'], 
            job['description'], 
            job['job_kwargs']['kwargs']['job_type'],
            job['status']['state'], 
            params, 
            job['job_kwargs']['kwargs']['experiment_id'],
            job['job_kwargs']['kwargs']['dataset'],
            job['logs']
            )
