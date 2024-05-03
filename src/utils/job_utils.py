import json
import os
import urllib
from dotenv import load_dotenv
import requests

load_dotenv(".env")

COMPUTE_URL = str(os.environ["MLEX_COMPUTE_URL"])


class TableJob:
    def __init__(
        self,
        job_id,
        job_name,
        job_type,
        job_status,
        job_params,
        experiment_id,
        dataset,
        job_logs,
    ):
        self.job_id = job_id
        self.name = job_name
        self.job_type = job_type
        self.status = job_status
        self.parameters = job_params
        self.experiment_id = experiment_id
        self.dataset = dataset
        self.job_logs = job_logs
        pass

    @staticmethod
    def compute_job_to_table_job(compute_job):
        params = str(compute_job["job_kwargs"]["kwargs"]["params"])
        if compute_job["job_kwargs"]["kwargs"]["job_type"].split()[0] != "train_model":
            params = f"{params}\nTraining Parameters: {compute_job['job_kwargs']['kwargs']['train_params']}"
        return TableJob(
            compute_job["uid"],
            compute_job["description"],
            compute_job["job_kwargs"]["kwargs"]["job_type"],
            compute_job["status"]["state"],
            params,
            compute_job["job_kwargs"]["kwargs"]["experiment_id"],
            compute_job["job_kwargs"]["kwargs"]["dataset"],
            compute_job["logs"],
        )

    @staticmethod
    def get_job(user, mlex_app, job_type=None, deploy_location=None):
        """
        Queries the job from the computing database
        Args:
            user:               username
            mlex_app:           mlexchange application
            job_type:           type of job
            deploy_location:    deploy location
        Returns:
            list of jobs that match the query
        """
        url = f"{COMPUTE_URL}/jobs?"
        if user:
            url += "&user=" + user
        if mlex_app:
            url += "&mlex_app=" + mlex_app
        if job_type:
            url += "&job_type=" + job_type
        if deploy_location:
            url += "&deploy_location=" + deploy_location
        response = urllib.request.urlopen(url)
        data = json.loads(response.read())
        return data

    @staticmethod
    def terminate_job(job_uid):
        requests.patch(f"{COMPUTE_URL}/jobs/{job_uid}/terminate")
        pass

    @staticmethod
    def delete_job(job_uid):
        requests.delete(f"{COMPUTE_URL}/jobs/{job_uid}/delete")
        pass


class MlexJob:
    def __init__(
        self,
        service_type,
        description,
        working_directory,
        job_kwargs,
        mlex_app="data_clinic",
        status={"state": "queue"},
        logs="",
        uid="",
        **kwargs,
    ):
        self.uid = uid
        self.mlex_app = mlex_app
        self.description = description
        self.service_type = service_type
        self.working_directory = working_directory
        self.job_kwargs = job_kwargs
        self.status = status
        self.logs = logs

    def submit(self, user, num_cpus, num_gpus):
        """
        Sends job to computing service
        Args:
            user:       user UID
            num_cpus:   Number of CPUs
            num_gpus:   Number of GPUs
        Returns:
            Workflow status
        """
        workflow = {
            "user_uid": user,
            "job_list": [self.__dict__],
            "host_list": [
                "mlsandbox.als.lbl.gov",
                "local.als.lbl.gov",
                "vaughan.als.lbl.gov",
            ],
            "dependencies": {"0": []},
            "requirements": {
                "num_processors": num_cpus,
                "num_gpus": num_gpus,
                "num_nodes": 1,
            },
        }
        url = f"{COMPUTE_URL}/workflows"
        return requests.post(url, json=workflow).status_code


def get_host(host_nickname):
    hosts = requests.get(f"{COMPUTE_URL}/hosts?&nickname={host_nickname}").json()
    max_processors = hosts[0]["backend_constraints"]["num_processors"]
    max_gpus = hosts[0]["backend_constraints"]["num_gpus"]
    return max_processors, max_gpus


def str_to_dict(text):
    text = text.replace("True", "true")
    text = text.replace("False", "false")
    text = text.replace("None", "null")
    return json.loads(text.replace("'", '"'))
