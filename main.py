from agents.data_ingestion_agent import DataIngestionAgent

agent = DataIngestionAgent()

result = agent.ingest("data/titanic.csv")

if result["status"] == "success":
    print("Dataset Loaded Successfully\n")

    print(result["metadata"])

    print("\nFirst 5 Rows\n")
    print(result["data"].head())

else:
    print(result["message"])