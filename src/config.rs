use serde::Deserialize;
use std::fs;
use std::path::Path;
use crate::BoxError;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub data_path: String,
    pub model_params: ModelParams,
    pub feature_params: FeatureParams,
}

impl AsRef<Path> for Config {
    fn as_ref(&self) -> &Path {
        self.data_path.as_ref()
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelParams {
    pub learning_rate: f64,
    pub batch_size: usize,
}

#[derive(Debug, Deserialize)]
pub struct FeatureParams {
    pub window_size: usize,
    pub use_technical_indicators: bool,
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, BoxError> {
        let contents = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }
} 