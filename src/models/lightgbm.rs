use super::traits::{Model, ModelFactory};
use crate::config::ModelParams;
use polars::prelude::*;
use lightgbm::{Dataset, Booster, BoosterParameters};
use std::error::Error;

pub struct LightGBMModel {
    booster: Option<Booster>,
    params: BoosterParameters,
}

impl Model for LightGBMModel {
    fn train(&mut self, features: &DataFrame, targets: &Series) -> Result<(), Box<dyn Error>> {
        let (data, labels) = prepare_data(features, targets)?;
        
        let dataset = Dataset::from_mat(data, labels)?;
        
        self.booster = Some(Booster::train(
            dataset,
            &self.params,
        )?);
        
        Ok(())
    }

    fn predict(&self, features: &DataFrame) -> Result<Series, Box<dyn Error>> {
        let booster = self.booster.as_ref()
            .ok_or("Model not trained")?;
            
        let data = features_to_matrix(features)?;
        let predictions = booster.predict(data)?;
        
        Ok(Series::new("predictions", predictions))
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        if let Some(booster) = &self.booster {
            booster.save_file(path)?;
        }
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.booster = Some(Booster::from_file(path)?);
        Ok(())
    }
}

impl ModelFactory for LightGBMModel {
    type ModelType = Self;
    
    fn create(config: &ModelParams) -> Result<Self::ModelType, Box<dyn Error>> {
        let mut params = BoosterParameters::default();
        params.learning_rate = config.learning_rate;
        params.num_leaves = 31;
        params.objective = String::from("regression");
        
        Ok(Self {
            booster: None,
            params,
        })
    }
}

// Helper functions
fn prepare_data(features: &DataFrame, targets: &Series) 
    -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> 
{
    let data = features.to_ndarray::<Float64Type>()?;
    let features_vec: Vec<Vec<f64>> = data.rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();
    
    let labels: Vec<f64> = targets.f64()?
        .into_iter()
        .map(|opt_val| opt_val.unwrap_or(0.0))
        .collect();
    
    Ok((features_vec, labels))
}

fn features_to_matrix(features: &DataFrame) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let data = features.to_ndarray::<Float64Type>()?;
    Ok(data.rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect())
} 