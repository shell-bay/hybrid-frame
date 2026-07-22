#!/usr/bin/env python3
"""
HybridFrame Demo - Quick showcase of key features.

Run: python demo.py
"""
from hybrid_frame import HybridFrame
import pandas as pd
import numpy as np
import time

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def main():
    # Create sample data
    print_header("1. Creating Sample Data")
    np.random.seed(42)
    n = 100000
    df = pd.DataFrame({
        'id': range(n),
        'name': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'], n),
        'age': np.random.randint(18, 65, n),
        'salary': np.random.uniform(30000, 120000, n),
        'dept': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], n)
    })
    print(f"Created DataFrame with {len(df):,} rows")
    print(df.head())

    # Create HybridFrame
    print_header("2. Creating HybridFrame")
    hf = HybridFrame.from_pandas(df)
    print(f"HybridFrame shape: {hf.shape}")
    print(f"Engine: {hf._engine}")

    # Filter operation
    print_header("3. Filter Operation (DuckDB)")
    hf.filter("age > 30")
    print(f"After filter - Shape: {hf.shape}")
    print(f"Engine: {hf._engine}")

    # Sort operation
    print_header("4. Sort Operation (DuckDB)")
    hf.sort_values('salary', ascending=False)
    print(f"After sort - Shape: {hf.shape}")

    # Head operation
    print_header("5. Head Operation (Pandas)")
    result = hf.head(10)
    print(f"After head - Shape: {result.shape}")
    print(f"Engine: {result._engine}")
    print(result.to_pandas())

    # Aggregation
    print_header("6. GroupBy Aggregation")
    agg_result = (
        HybridFrame.from_pandas(df)
        .groupby_agg(['dept'], {'salary': ['sum', 'mean', 'max']})
    )
    print(agg_result.to_pandas())

    # SQL Query
    print_header("7. SQL Query")
    sql_result = (
        HybridFrame.from_pandas(df)
        .sql("""
            SELECT dept, 
                   COUNT(*) as count,
                   ROUND(AVG(salary), 2) as avg_salary
            FROM self
            WHERE age > 25
            GROUP BY dept
            ORDER BY avg_salary DESC
        """)
    )
    print(sql_result.to_pandas())

    # Performance comparison
    print_header("8. Performance Comparison")
    def benchmark_pandas():
        return df[df['age'] > 30].sort_values('salary', ascending=False).head(10)

    def benchmark_hybrid():
        return (
            HybridFrame.from_pandas(df)
            .filter('age > 30')
            .sort_values('salary', ascending=False)
            .head(10)
            .to_pandas()
        )

    # Warm up
    _ = benchmark_pandas()
    _ = benchmark_hybrid()

    # Benchmark
    n_runs = 5
    start = time.time()
    for _ in range(n_runs):
        _ = benchmark_pandas()
    pandas_time = (time.time() - start) / n_runs

    start = time.time()
    for _ in range(n_runs):
        _ = benchmark_hybrid()
    hybrid_time = (time.time() - start) / n_runs

    print(f"Pandas: {pandas_time*1000:.2f}ms")
    print(f"HybridFrame: {hybrid_time*1000:.2f}ms")
    print(f"Speedup: {pandas_time/hybrid_time:.2f}x")

    print_header("Demo Complete!")
    print("HybridFrame: The fastest way to work with data in Python")
    print("GitHub: https://github.com/shell-bay/hybrid-frame")
    print("PyPI: https://pypi.org/project/hybrid-frame/")

if __name__ == "__main__":
    main()
