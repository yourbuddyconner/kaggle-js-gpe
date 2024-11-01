use polars::prelude::*;
use std::path::Path;
use anyhow::{Result, Context};
use glob::glob;
use tracing::debug;

/// DataLoader handles efficient loading and preprocessing of market data
pub struct DataLoader {
    lazy_frame: LazyFrame,
    window_size: usize,
}

impl DataLoader {
    /// Creates a new DataLoader instance from a parquet file pattern
    pub fn new<P: AsRef<Path>>(path_pattern: P, window_size: usize) -> Result<Self> {
        println!("Loading parquet files matching pattern: {}", path_pattern.as_ref().display());
        
        // Get all files matching the pattern
        let paths: Vec<_> = glob(path_pattern.as_ref().to_str().unwrap())
            .context("Failed to read glob pattern")?
            .filter_map(Result::ok)
            .collect();
            
        if paths.is_empty() {
            anyhow::bail!("No files found matching pattern: {}", path_pattern.as_ref().display());
        }

        // Create a LazyFrame from the first file to get schema
        let first_file = &paths[0];
        let schema = ParquetReader::new(std::fs::File::open(first_file)?)
            .schema()
            .context("Failed to read parquet schema")?;
        println!("Schema from first file:\n{:#?}", schema);

        // Create a LazyFrame that scans all matching files
        let lazy_frame = LazyFrame::scan_parquet_files(Arc::from(paths), Default::default())
            .context("Failed to scan parquet files")?;
            
        Ok(Self {
            lazy_frame,
            window_size,
        })
    }

    /// Loads a batch of data with specified columns
    pub fn load_batch(&self) -> Result<DataFrame> {
        debug!("Starting load_batch");
        
        // Generate feature column names
        let feature_columns: Vec<String> = (0..79)
            .map(|i| format!("feature_{:02}", i))
            .collect();
        debug!(?feature_columns, "Feature columns generated");

        // Create expressions for metadata columns with proper casting
        let mut columns: Vec<Expr> = vec![
            col("date_id").cast(DataType::Float32),
            col("time_id").cast(DataType::Float32),
            col("symbol_id").cast(DataType::Float32),
            col("weight")
        ];
        debug!(?columns, "Initial columns selected");
        
        // Add feature columns with casting for known integer columns
        columns.extend(feature_columns.iter().map(|s| {
            let expr = match s.as_str() {
                "feature_09" | "feature_10" => col(s).cast(DataType::Float32),
                "feature_11" => col(s).cast(DataType::Float32),
                _ => col(s)
            };
            debug!("Added column expression for {}: {:?}", s, expr);
            expr
        }));

        debug!(?columns, "All columns selected");

        // Use the existing lazy_frame
        debug!("Executing lazy query");
        let df = self.lazy_frame
            .clone()
            .select(columns)
            .collect()?;
        
        debug!("Query executed. DataFrame schema: {:?}", df.schema());
        debug!("DataFrame shape: {:?}", df.shape());

        Ok(df)
    }

    /// Process features with rolling statistics grouped by symbol
    pub fn process_features(&self) -> Result<DataFrame> {
        self.lazy_frame.clone()
            .group_by([col("symbol_id")])
            .agg([
                col("feature_*").rolling_mean(RollingOptionsFixedWindow {
                    window_size: self.window_size,
                    ..Default::default()
                }),
                col("feature_*").rolling_std(RollingOptionsFixedWindow {
                    window_size: self.window_size,
                    ..Default::default()
                }),
            ])
            .collect()
            .context("Failed to process features")
    }

    /// Clean the data by removing nulls and filtering by weight
    pub fn clean_data(&self) -> Result<DataFrame> {
        self.lazy_frame.clone()
            .drop_nulls(None)
            .filter(col("weight").gt(0.0))
            .collect()
            .context("Failed to clean data")
    }

    /// Build a complete processing pipeline as specified
    pub fn build_processing_pipeline(&self) -> Result<LazyFrame> {
        Ok(self.lazy_frame.clone()
            .drop_nulls(None)
            .filter(col("weight").gt(0.0))
            .with_columns(
                self.compute_time_features()
                    .into_iter()
                    .chain(self.compute_cross_features())
                    .collect::<Vec<_>>()
            ))
    }

    /// Compute time-based features
    fn compute_time_features(&self) -> Vec<Expr> {
        vec![
            col("feature_*")
                .rolling_mean(RollingOptionsFixedWindow {
                    window_size: 24,
                    ..Default::default()
                })
                .alias("feature_mean_24"),
            col("feature_*")
                .rolling_std(RollingOptionsFixedWindow {
                    window_size: 24,
                    ..Default::default()
                })
                .alias("feature_std_24"),
            col("feature_*")
                .rolling_mean(RollingOptionsFixedWindow {
                    window_size: 48,
                    ..Default::default()
                })
                .alias("feature_mean_48"),
            col("feature_*")
                .rolling_std(RollingOptionsFixedWindow {
                    window_size: 48,
                    ..Default::default()
                })
                .alias("feature_std_48"),
        ]
    }

    /// Compute cross-sectional features
    fn compute_cross_features(&self) -> Vec<Expr> {
        vec![
            col("feature_*")
                .mean()
                .over([col("symbol_id")])
                .alias("feature_symbol_mean"),
            col("feature_*")
                .std(1)  // Fix: Add ddof parameter
                .over([col("symbol_id")])
                .alias("feature_symbol_std"),
        ]
    }

    /// Get a subset of data for a specific date range
    pub fn get_date_range(&self, start_date: i32, end_date: i32) -> Result<DataFrame> {
        self.lazy_frame.clone()
            .filter(
                col("date_id")
                    .gt_eq(lit(start_date))
                    .and(col("date_id").lt_eq(lit(end_date)))
            )
            .collect()
            .context("Failed to get date range")
    }

    /// Get a subset of data for specific symbols
    pub fn get_symbols(&self, symbols: Vec<i32>) -> Result<DataFrame> {
        self.lazy_frame.clone()
            .filter(col("symbol_id").is_in(lit(Series::new("symbol_id".into(), symbols))))
            .collect()
            .context("Failed to get symbols")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader_creation() {
        let loader = DataLoader::new("test_data.parquet", 24);
        assert!(loader.is_ok());
    }

    #[test]
    fn test_clean_data() {
        let loader = DataLoader::new("test_data.parquet", 24).unwrap();
        let result = loader.clean_data();
        assert!(result.is_ok());
    }

    // Add more tests as needed...
}
