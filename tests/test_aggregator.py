from src.preprocessing.aggregator import AISAggregator
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

print('Testing aggregator...')
agg = AISAggregator(output_path='outputs/processed/sample_agg.parquet')
df_agg = agg.run('outputs/processed/sample_clean.parquet')
print(f'Aggregated rows: {len(df_agg):,}')
print(f'Columns: {list(df_agg.columns)}')
print(df_agg.head(3))
