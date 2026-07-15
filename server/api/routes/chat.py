from fastapi import APIRouter

from server.dependencies import orchestration_service
from server.schemas import ChatRequest, ChatResponse
from server.services.dataset_registry import dataset_registry

from state.pipeline_state import PipelineState

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


@router.post("/")
async def chat(request: ChatRequest):

    dataset = None

    if request.dataset_id:
        dataset = dataset_registry.get_dataset(
            request.dataset_id
        )

    state = PipelineState(
        user_prompt=request.prompt
    )

    if dataset:
        state.dataset.dataset_path = dataset["dataset_path"]
        state.dataset.dataset_name = dataset["dataset_name"]
        state.dataset.source_type = dataset["source_type"]

        state.logs.append("Uploaded dataset detected.")

    result = orchestration_service.run(state)

    return ChatResponse(
        user_prompt=result.user_prompt,
        assistant_message=result.assistant_message,
        run_id=result.run_id,
        status=result.status,
        execution_time=result.execution_time,
        completed_steps=result.completed_steps,
        logs=result.logs,
        dataset_name=result.dataset.dataset_name,
        rows=result.dataset.num_rows,
        columns=result.dataset.num_columns,
        problem_type=result.validation.problem_type,
        target_column=result.validation.target_column,
    )