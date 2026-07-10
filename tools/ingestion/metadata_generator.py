class MetadataGenerator:

    @staticmethod
    def generate(df, source):
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "column_names": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_mb": round(df.memory_usage(deep=True).sum()/1024/1024,2),
            "source": source
        }