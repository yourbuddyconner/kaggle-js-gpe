use super::traits::{Model, ModelFactory};
use crate::config::ModelParams;
use polars::prelude::*;
use xgboost::{Booster, DMatrix, parameters};
use std::error::Error;

pub struct XGBoostModel {
    booster: Option<Booster>,
    params: parameters::BoosterParameters,
}

impl Model for XGBoostModel {
    fn train(&mut self, features: &DataFrame, targets: &Series) -> Result<(), Box<dyn Error>> {
        // Convert DataFrame to DMatrix
        let x = features_to_dmatrix(features)?;
        let y = series_to_vec(targets)?;
        
        let train_matrix = DMatrix::from_dense(&x, features.height())?;
        train_matrix.set_labels(&y)?;

        // Train the model
        self.booster = Some(Booster::train(
            &train_matrix,
            &self.params,
            100, // num_rounds
            &[], // evaluation sets
            None, // obj_func
            None, // eval_func
        )?);

        Ok(())
    }

    fn predict(&self, features: &DataFrame) -> Result<Series, Box<dyn Error>> {
        let booster = self.booster.as_ref()
            .ok_or("Model not trained")?;
            
        let x = features_to_dmatrix(features)?;
        let test_matrix = DMatrix::from_dense(&x, features.height())?;
        
        let predictions = booster.predict(&test_matrix)?;
        Ok(Series::new("predictions", predictions))
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        if let Some(booster) = &self.booster {
            booster.save(path)?;
        }
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.booster = Some(Booster::load(path)?);
        Ok(())
    }
}

impl ModelFactory for XGBoostModel {
    type ModelType = Self;
    
    fn create(config: &ModelParams) -> Result<Self::ModelType, Box<dyn Error>> {
        let mut params = parameters::BoosterParameters::default();
        params.learning_rate = Some(config.learning_rate);
        params.max_depth = Some(6);
        params.objective = Some(String::from("reg:squarederror"));
        
        Ok(Self {
            booster: None,
            params,
        })
    }
}

// Helper functions
fn features_to_dmatrix(df: &DataFrame) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut features = Vec::with_capacity(df.height() * df.width());
    
    for col in df.get_columns() {
        let values: Vec<f32> = col.f64()?
            .into_iter()
            .map(|opt_val| opt_val.unwrap_or(0.0) as f32)
            .collect();
        features.extend(values);
    }
    
    Ok(features)
}

fn series_to_vec(series: &Series) -> Result<Vec<f32>, Box<dyn Error>> {
    series.f64()?
        .into_iter()
        .map(|opt_val| Ok(opt_val.unwrap_or(0.0) as f32))
        .collect()
} 