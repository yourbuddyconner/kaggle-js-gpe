use std::error::Error;
use std::fmt;
use polars::error::PolarsError;

#[derive(Debug)]
pub enum MarketPredictionError {
    DataLoading(PolarsError),
    FeatureEngineering(String),
    GPEComputation(String),
    ModelPrediction(String),
    InvalidInput(String),
}

impl fmt::Display for MarketPredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataLoading(e) => write!(f, "Data loading error: {}", e),
            Self::FeatureEngineering(msg) => write!(f, "Feature engineering error: {}", msg),
            Self::GPEComputation(msg) => write!(f, "GPE computation error: {}", msg),
            Self::ModelPrediction(msg) => write!(f, "Model prediction error: {}", msg),
            Self::InvalidInput(msg) => write!(f, "Invalid input error: {}", msg),
        }
    }
}

impl Error for MarketPredictionError {}

impl From<PolarsError> for MarketPredictionError {
    fn from(err: PolarsError) -> Self {
        Self::DataLoading(err)
    }
} 