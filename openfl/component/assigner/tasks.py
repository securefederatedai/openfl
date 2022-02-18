from dataclasses import dataclass


@dataclass
class Task:
    name: str
    function_name: str  # validate or train
    is_local: bool = False
    # hyper_parameters: dict  # We can expend it in the future


train_task = Task(
    name='train',
    function_name='train'
)
localy_tuned_model_validate = Task(
    # in fact we have a typo in the name, it should be 'locally_'
    # and probably that name is redundant
    name='localy_tuned_model_validate',
    function_name='validate',
    is_local=True
)
aggregated_model_validate = Task(
    name='aggregated_model_validate',
    function_name='validate'
)

train_and_validate_group = [
    train_task,
    localy_tuned_model_validate,
    aggregated_model_validate
]
validate_group = [aggregated_model_validate]


def train_and_validate_assigner(collaborator_name: str, round_number: int, **kwargs):
    return train_and_validate_group


def validate_assigner(collaborator_name: str, round_number: int, **kwargs):
    return validate_group
