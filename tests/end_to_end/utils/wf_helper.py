from metaflow import Flow
import logging

log = logging.getLogger(__name__)

def validate_flow(flow_obj, expected_flow_steps):
    """
    Validate:
    1. If the given training round were completed
    2. If all the steps were executed
    3. If each collaborator step was executed
    4. If end was executed once
    """

    cli_flow_obj = Flow("TestFlowInternalLoop")  # Flow object from CLI
    cli_flow_steps = list(cli_flow_obj.latest_run)  # Steps from CLI
    cli_step_names = [step.id for step in cli_flow_steps]

    # 1. If the given training round were completed
    assert flow_obj.training_rounds == flow_obj.train_count, "Number of training completed is not equal to training rounds"

    for step in cli_flow_steps:
        task_count = 0
        func = getattr(flow_obj, step.id)
        for task in list(step):
            task_count = task_count + 1

        # Each aggregator step should be executed for training rounds times
        if (
            (func.aggregator_step is True)
            and (task_count != flow_obj.training_rounds)
            and (step.id != "end")
        ):
            assert False, f"More than one execution detected for Aggregator Step: {step}"

        # Each collaborator step is executed for (training rounds)*(number of collaborator) times
        if (func.collaborator_step is True) and (
            task_count != len(flow_obj.collaborators) * flow_obj.training_rounds
        ):
            assert False, f"Incorrect number of execution detected for Collaborator Step: {step}. Expected: {flow_obj.training_rounds*len(flow_obj.collaborators)} Actual: {task_count}"

    steps_present_in_cli = [
        step for step in expected_flow_steps if step in cli_step_names
    ]
    missing_steps_in_cli = [
        step for step in expected_flow_steps if step not in cli_step_names
    ]
    extra_steps_in_cli = [
        step for step in cli_step_names if step not in expected_flow_steps
    ]
    return steps_present_in_cli, missing_steps_in_cli, extra_steps_in_cli
