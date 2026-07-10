from graph.workflow import workflow

from state.pipeline_state import PipelineState


def main():

    state = PipelineState(
        user_prompt="Build a Titanic survival prediction model using data/titanic.csv"
    )

    result = workflow.invoke(state)

    print(result)


if __name__ == "__main__":
    main()