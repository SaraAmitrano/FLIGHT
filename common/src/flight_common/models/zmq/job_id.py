from pydantic import UUID4

JobID = UUID4


# def get_job_id_str(job_id: JobID) -> str:
#     return str(job_id)


# def parse_job_id(job_id_str: str) -> JobID:
#     return JobID.model_validate(job_id_str)

__all__ = ["JobID"]
