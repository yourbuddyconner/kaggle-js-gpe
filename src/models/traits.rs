use polars::prelude::*;
use std::error::Error;
use gbdt::decision_tree::{DataVec, Data};
use anyhow::Result;

pub trait Model {
    fn train(&mut self, features: &DataFrame, targets: &Series) -> Result<(), Box<dyn Error>>;
    fn predict(&self, features: &DataFrame) -> Result<Series, Box<dyn Error>>;
    fn save(&self, path: &str) -> Result<(), Box<dyn Error>>;
    fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>>;
}

pub trait ModelFactory {
    type ModelType: Model;
    
    fn create(config: &crate::config::ModelParams) -> Result<Self::ModelType, Box<dyn Error>>;
}

pub trait IntoDataVec {
    fn into_data_vec(self) -> Result<DataVec, Box<dyn std::error::Error>>;
}

impl IntoDataVec for DataFrame {
    fn into_data_vec(self) -> Result<DataVec, Box<dyn std::error::Error>> {
        // Convert all columns to f32 for consistency
        let df = self.lazy()
            .select([col("*").cast(DataType::Float32)])
            .collect()?;
        
        // Convert to Vec<Vec<f32>>
        let arrays: Vec<Vec<f32>> = df.iter()
            .map(|series| series.f32()
                .unwrap()
                .into_iter()
                .map(|opt_val| opt_val.unwrap_or(0.0))
                .collect())
            .collect();
        
        // Transpose the data to row-major format as expected by GBDT
        let n_rows = arrays[0].len();
        let n_cols = arrays.len();
        let mut data_vec = DataVec::with_capacity(n_rows);
        
        // Convert each row to Data type as required by GBDT
        for row_idx in 0..n_rows {
            let row_data: Vec<f32> = (0..n_cols)
                .map(|col_idx| arrays[col_idx][row_idx])
                .collect();
            let data = Data::new_training_data(row_data, 0.0, 1.0, None);
            data_vec.push(data);
        }
        
        Ok(data_vec)
    }
} 