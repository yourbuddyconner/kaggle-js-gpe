use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::error::Error;
use std::fmt;
use ndarray::s;

// Import the geometric entropy estimation types
use geometric_entropy::{
    GeometricPartitionEntropy, 
    MeasureFunction,
};


#[derive(Debug, Clone)]
pub struct GPEConfig {
    pub epsilon: f64,
    pub k: usize,
    pub measure_fn: MeasureFunction,
}

impl Default for GPEConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            k: 10, // Number of partitions
            measure_fn: MeasureFunction::RatioMeasure,
        }
    }
}

pub struct GPEFeatureExtractor {
    config: GPEConfig,
    window_size: usize,
    gpe: GeometricPartitionEntropy,
}

// Create a wrapper type for GeometricPartitionError
#[derive(Debug)]
pub struct GPError(geometric_entropy::GeometricPartitionError);

impl From<geometric_entropy::GeometricPartitionError> for GPError {
    fn from(err: geometric_entropy::GeometricPartitionError) -> Self {
        GPError(err)
    }
}

impl fmt::Display for GPError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Geometric partition error: {:?}", self.0)
    }
}

impl std::error::Error for GPError {}

impl GPEFeatureExtractor {
    pub fn new(window_size: usize) -> Result<Self, GPError> {
        let config = GPEConfig::default();
        let gpe = GeometricPartitionEntropy::with_measure_function(
            config.epsilon,
            config.k,
            config.measure_fn,
        ).map_err(GPError)?;

        Ok(Self {
            config,
            window_size,
            gpe,
        })
    }

    pub fn compute_gpe_features(&self, series: &Series) -> Result<Vec<f64>, Box<dyn Error>> {
        let values: Vec<f64> = series.f64()?.into_iter()
            .filter_map(|x| x)
            .collect();

        if values.len() < self.window_size {
            return Ok(vec![0.0, 0.0, 0.0]); // Return zeros for insufficient data
        }

        // Create sliding windows and compute GPE for each
        let mut features = Vec::new();
        for window in values.windows(self.window_size) {
            // Convert window to 2D array for GPE computation
            let window_array = Array2::from_shape_vec(
                (window.len(), 1),
                window.to_vec(),
            )?;

            // Compute GPE entropy
            let entropy = self.gpe.compute_entropy(&window_array)
                .map_err(|e| Box::new(GPError(e)) as Box<dyn Error>)?;
            features.push(entropy);
        }

        // Compute summary statistics of the GPE values
        let mean_gpe = features.iter().sum::<f64>() / features.len() as f64;
        let var_gpe = features.iter()
            .map(|x| (x - mean_gpe).powi(2))
            .sum::<f64>() / features.len() as f64;

        Ok(vec![
            mean_gpe,                // Average complexity
            var_gpe.sqrt(),         // Stability of complexity
            features.last().copied().unwrap_or(0.0), // Most recent complexity
        ])
    }

    pub fn compute_multivariate_gpe(&self, features: &[Series]) -> Result<Vec<f64>, Box<dyn Error>> {
        // Convert series to 2D array
        let n_samples = features[0].len();
        let n_features = features.len();
        
        let mut data = Array2::zeros((n_samples, n_features));
        
        for (i, series) in features.iter().enumerate() {
            let values: Vec<f64> = series.f64()?.into_iter()
                .filter_map(|x| x)
                .collect();
            data.column_mut(i).assign(&Array1::from_vec(values));
        }

        // Compute multivariate GPE
        let entropy = self.gpe.compute_entropy(&data)
            .map_err(|e| Box::new(GPError(e)) as Box<dyn Error>)?;
        
        // For pairs of features, compute mutual information
        let mut mutual_info = Vec::new();
        for i in 0..n_features {
            for j in (i+1)..n_features {
                let mi = self.gpe.compute_mutual_information(
                    &data.slice(s![.., i..i+1]).to_owned(),
                    &data.slice(s![.., j..j+1]).to_owned(),
                )
                .map_err(|e| Box::new(GPError(e)) as Box<dyn Error>)?;
                mutual_info.push(mi);
            }
        }

        // Return joint entropy and average mutual information
        let avg_mi = mutual_info.iter().sum::<f64>() / mutual_info.len() as f64;
        Ok(vec![entropy, avg_mi])
    }
}

pub struct MarketRegimeDetector {
    gpe_extractor: GPEFeatureExtractor,
    threshold: f64,
}

impl MarketRegimeDetector {
    pub fn new(threshold: f64) -> Result<Self, GPError> {
        Ok(Self {
            gpe_extractor: GPEFeatureExtractor::new(100)?,
            threshold,
        })
    }

    pub fn detect_regime_change(
        &self,
        historical: &DataFrame,
        current: &DataFrame,
    ) -> Result<bool, Box<dyn Error>> {
        // Extract feature columns
        let hist_features: Vec<Series> = historical
            .select(["feature_*"])?
            .get_columns()
            .iter()
            .map(|col| (*col).clone().as_series().unwrap().clone())
            .collect();
        
        let curr_features: Vec<Series> = current
            .select(["feature_*"])?
            .get_columns()
            .iter()
            .map(|col| (*col).clone().as_series().unwrap().clone())
            .collect();

        // Compute multivariate GPE for both periods
        let hist_gpe = self.gpe_extractor.compute_multivariate_gpe(&hist_features)?;
        let curr_gpe = self.gpe_extractor.compute_multivariate_gpe(&curr_features)?;

        // Compare complexity measures
        let entropy_diff = (curr_gpe[0] - hist_gpe[0]).abs();
        let mi_diff = (curr_gpe[1] - hist_gpe[1]).abs();

        // Detect regime change if either measure changes significantly
        Ok(entropy_diff > self.threshold || mi_diff > self.threshold)
    }
} 