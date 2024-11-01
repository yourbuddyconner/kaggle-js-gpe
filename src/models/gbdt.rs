use super::traits::{Model, ModelFactory};
use crate::config::ModelParams;
use polars::prelude::*;
use gbdt::{config::Config as GBDTConfig, gradient_boost::GBDT, decision_tree::DataVec};
use std::error::Error;

pub struct GBDTModel {
    model: Option<GBDT>,
    config: GBDTConfig,
}

impl Model for GBDTModel {
    fn train(&mut self, features: &DataFrame, targets: &Series) -> Result<(), Box<dyn Error>> {
        // Convert data to GBDT format
        let mut train_data = prepare_data(features, targets)?;
        
        // Create and train the model
        let mut gbdt = GBDT::new(&self.config);
        gbdt.fit(&mut train_data);
        
        self.model = Some(gbdt);
        Ok(())
    }

    fn predict(&self, features: &DataFrame) -> Result<Series, Box<dyn Error>> {
        let model = self.model.as_ref()
            .ok_or("Model not trained")?;
            
        let test_data = features_to_datavec(features)?;
        let predictions = model.predict(&test_data);
        
        Ok(Series::new(PlSmallStr::from("predictions"), predictions))
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        if let Some(model) = &self.model {
            model.save_model(path)?;
        }
        Ok(())
    }

    fn load(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        self.model = Some(GBDT::load_model(path)?);
        Ok(())
    }
}

impl ModelFactory for GBDTModel {
    type ModelType = Self;
    
    fn create(params: &ModelParams) -> Result<Self::ModelType, Box<dyn Error>> {
        let mut config = GBDTConfig::new();
        
        // Configure GBDT parameters
        config.set_iterations(100);
        config.set_max_depth(6);
        config.set_shrinkage(params.learning_rate as f32);
        config.set_loss("SquaredError"); // for regression
        config.set_debug(false);
        config.set_data_sample_ratio(1.0);
        config.set_feature_sample_ratio(1.0);
        config.set_training_optimization_level(2);
        
        Ok(Self {
            model: None,
            config,
        })
    }
}

// Helper functions
fn prepare_data(features: &DataFrame, targets: &Series) -> Result<DataVec, Box<dyn Error>> {
    let mut data = features_to_datavec(features)?;
    
    // Add targets (previously labels)
    let targets: Vec<f32> = targets.f64()?
        .into_iter()
        .map(|opt_val| opt_val.unwrap_or(0.0) as f32)
        .collect();
    
    for (i, target) in targets.into_iter().enumerate() {
        data[i].target = target;  // Changed from 'label' to 'target'
    }
    
    Ok(data)
}

fn features_to_datavec(features: &DataFrame) -> Result<DataVec, Box<dyn Error>> {
    use gbdt::decision_tree::Data;
    
    let mut data_vec = Vec::with_capacity(features.height());
    
    for row_idx in 0..features.height() {
        let mut row_values = Vec::with_capacity(features.width());
        
        for col in features.get_columns() {
            // Get the optional f64 value and convert to f32, defaulting to 0.0 if None
            let value = col.f64()?
                .get(row_idx);
            if let Some(value) = value {
                row_values.push(value as f32);
            } else {
                row_values.push(0.0);
            }
        }
        
        data_vec.push(Data {
            label: 0.0,
            weight: 1.0,
            feature: row_values,  // Changed from 'features' to 'feature'
            target: 0.0,         // Changed from 'label' to 'target'
            residual: 0.0,       // Added missing field
            initial_guess: 0.0,  // Added missing field
        });
    }
    
    Ok(data_vec)
} 