from loaders import csv_loader, excel_loader, json_loader, url_loader, zip_loader
from utils import metadata_generator
class DataIngestionAgent:

    def __init__(self):
        self.loaders = {
            "csv": csv_loader(),
            "excel": excel_loader(),
            "json": json_loader(),
            "url": url_loader(),
            "zip": zip_loader()
        }

    def ingest(self, source_type, source):
        if source_type not in self.loaders:
            raise Exception("Unsupported source")

        df = self.loaders[source_type].load(source)

        metadata = metadata_generator.generate(df, source)

        return {
            "data": df,
            "metadata": metadata
        }