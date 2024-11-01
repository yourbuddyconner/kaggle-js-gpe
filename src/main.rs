use jane_street_gpe::{config::Config, DataLoader, FeatureEngineer};
use jane_street_gpe::models::traits::IntoDataVec;
use gbdt::config as gbdt_config;
use gbdt::decision_tree::{DataVec, PredVec};
use gbdt::gradient_boost::GBDT;
use gbdt::input::{InputFormat, load};
use tracing::{debug, info, instrument};

#[instrument]
fn main() -> Result<(), jane_street_gpe::BoxError> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    info!("Starting Jane Street GPE application");
    
    // Load Jane Street config
    let config_path = "config.toml";
    debug!("Loading config from path: {}", config_path);
    let config = Config::load(config_path)?;
    debug!(?config, "Config loaded successfully");

    let train_path = "data/train.parquet/**/part-*.parquet";
    let test_path = "data/test.parquet/**/part-*.parquet";
    debug!("Train path: {}, Test path: {}", train_path, test_path);
    
    // Create DataLoader with the config path and handle the Result
    debug!("Initializing DataLoaders");
    let test_data_loader = DataLoader::new(test_path, config.feature_params.window_size)?;
    let train_data_loader = DataLoader::new(train_path, config.feature_params.window_size)?;
    debug!("DataLoaders initialized successfully");
    
    // Load training data
    debug!("Loading training and test features");
    let train_features = train_data_loader.load_batch()?;
    debug!(train_features_shape = ?train_features.shape(), "Training features loaded");
    
    let test_features = test_data_loader.load_batch()?;
    debug!(test_features_shape = ?test_features.shape(), "Test features loaded");
    
    // Create FeatureEngineer with ownership of data_loader
    debug!("Initializing FeatureEngineer");
    let feature_engineer = FeatureEngineer::new(test_data_loader)?;
    let features = feature_engineer.engineer_features()?;
    debug!(engineered_features_shape = ?features.shape(), "Features engineered successfully");

    // Set up GBDT configuration
    debug!("Configuring GBDT model");
    let mut gbdt_cfg = gbdt_config::Config::new();
    gbdt_cfg.set_feature_size(83);
    gbdt_cfg.set_max_depth(3);
    gbdt_cfg.set_iterations(50);
    gbdt_cfg.set_shrinkage(0.1);
    gbdt_cfg.set_loss("LogLikelyhood");
    gbdt_cfg.set_debug(true);
    gbdt_cfg.set_data_sample_ratio(1.0);
    gbdt_cfg.set_feature_sample_ratio(1.0);
    gbdt_cfg.set_training_optimization_level(2);
    // Convert features to GBDT format
    debug!("Setting up input format");
    let mut input_format = InputFormat::csv_format();
    input_format.set_feature_size(83);
    input_format.set_label_index(83);
    debug!(?input_format, "Input format configured");

    // Train GBDT model
    debug!("Initializing GBDT model");
    let mut gbdt = GBDT::new(&gbdt_cfg);
    
    debug!("Converting features to DataVec format");
    let mut data_vec = features.into_data_vec()?;
    debug!("Starting model training");
    
    gbdt.fit(&mut data_vec);
    debug!("Model training completed");

    // Save the trained model
    debug!("Saving model to gbdt.model");
    gbdt.save_model("gbdt.model")?;
    debug!("Model saved successfully");

    // Make predictions
    debug!("Converting test features to DataVec format");
    let test_data_vec = test_features.into_data_vec()?;
    debug!("Making predictions on test data");
    let predictions: PredVec = gbdt.predict(&test_data_vec);
    debug!(predictions_len = predictions.len(), "Predictions complete");
    
    info!("Training complete! Model saved as gbdt.model");
    
    Ok(())
} 