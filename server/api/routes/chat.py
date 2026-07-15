from fastapi import APIRouter

from server.dependencies import orchestration_service
from server.schemas import ChatRequest, ChatResponse

from state.pipeline_state import PipelineState

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/")
async def chat(request: ChatRequest):

    state = PipelineState(
        user_prompt=request.prompt
    )

    result = orchestration_service.run(state)

    return ChatResponse(
    user_prompt = result.user_prompt,
    #ai_response = result.ai_response,
    status=result.status,
    completed_steps=result.completed_steps,
    logs=result.logs,
    dataset_name=result.dataset.dataset_name,
    rows=result.dataset.num_rows,
    columns=result.dataset.num_columns,
    problem_type=result.validation.problem_type,
    target_column=result.validation.target_column,
)