use crate::data_loader::DataLoader;
use crate::gpe::GPEFeatureExtractor;
use polars::prelude::*;
use augurs::{ets::AutoETS, mstl::MSTLModel, prelude::*, seasons::{Detector, PeriodogramDetector}};
use augurs::ets::trend::AutoETSTrendModel;
use std::error::Error;

pub struct TimeSeriesProcessor {
    season_detector: PeriodogramDetector,
    mstl_model: MSTLModel<AutoETSTrendModel>,
    window_size: usize,
}

impl TimeSeriesProcessor {
    pub fn new(window_size: usize) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            season_detector: PeriodogramDetector::builder()
                .min_period(2_u32)
                .max_period(window_size.try_into().unwrap())
                .build(),
            mstl_model: MSTLModel::new(
                vec![window_size],
                AutoETS::non_seasonal().into_trend_model()
            ),
            window_size,
        })
    }

    pub fn detect_seasonality(&self, series: &Series) -> Result<Vec<usize>, Box<dyn Error>> {
        let values: Vec<f64> = series.f64()?
            .into_iter()
            .filter_map(|x| x)
            .collect();

        // Convert detected periods from u32 to usize
        Ok(self.season_detector.detect(&values)
            .into_iter()
            .map(|p| p as usize)
            .collect())
    }
    
    pub fn decompose_series(&self, series: &Series) -> Result<DataFrame, Box<dyn Error>> {
        let values: Vec<f64> = series.f64()?
            .into_iter()
            .filter_map(|x| x)
            .collect();

        // Detect seasonal periods
        let periods = self.detect_seasonality(series)?;
        
        // Create new MSTL model with detected periods
        let mstl = MSTLModel::new(
            periods,
            AutoETS::non_seasonal().into_trend_model()
        );

        // Fit and get components
        let fitted = mstl.fit(&values)?;
        let forecast = fitted.predict_in_sample(None)?;
        
        // Convert forecast components to DataFrame
        let trend = Series::new("trend".into(), forecast.point);
        
        // Convert seasonal component (nested Vec<f32>) to Vec<f64>
        let seasonal_component: Vec<f64> = fitted.fit()
            .seasonal()
            .iter()
            .flat_map(|v| v.iter())
            .map(|&x| x as f64)
            .collect();

        // Convert remainder (Vec<f32>) to Vec<f64>
        let remainder: Vec<f64> = fitted.fit()
            .remainder()
            .iter()
            .map(|&x| x as f64)
            .collect();

        let seasonal = Series::new("seasonal".into(), seasonal_component);
        let remainder = Series::new("remainder".into(), remainder);

        DataFrame::new(vec![
            Column::Series(trend),
            Column::Series(seasonal), 
            Column::Series(remainder)
        ])
            .map_err(|e| Box::new(e) as Box<dyn Error>)
    }
}

pub struct FeatureEngineer {
    data_loader: DataLoader,
    ts_processor: TimeSeriesProcessor,
    gpe_extractor: GPEFeatureExtractor,
}

impl FeatureEngineer {
    pub fn new(data_loader: DataLoader) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            data_loader,
            ts_processor: TimeSeriesProcessor::new(24)?,
            gpe_extractor: GPEFeatureExtractor::new(24)?,
        })
    }

    pub fn engineer_features(&self) -> Result<DataFrame, Box<dyn Error>> {
        let df = self.data_loader.load_batch()?;
        
        // Process each feature column with time series analysis
        let mut enhanced_features = Vec::new();
        
        for col_name in df.get_column_names() {
            if col_name.starts_with("feature_") {
                let series = df.column(col_name)?.as_series().unwrap();
                
                // Compute decomposition
                let decomp = self.ts_processor.decompose_series(&series)?;
                
                // Add decomposition components with feature name prefix
                enhanced_features.push(Series::new(
                    format!("{}_trend", col_name).into(),
                    decomp.column("trend")?.as_series().unwrap().clone()
                ));
                enhanced_features.push(Series::new(
                    format!("{}_seasonal", col_name).into(),
                    decomp.column("seasonal")?.as_series().unwrap().clone()
                ));
                enhanced_features.push(Series::new(
                    format!("{}_remainder", col_name).into(),
                    decomp.column("remainder")?.as_series().unwrap().clone()
                ));
                
                // Compute GPE features
                let gpe_features = self.gpe_extractor.compute_gpe_features(&series.clone().into_series())?;
                enhanced_features.push(Series::new(
                    format!("{}_gpe_entropy", col_name).into(),
                    vec![gpe_features[0]; series.len()]
                ));
                enhanced_features.push(Series::new(
                    format!("{}_gpe_stability", col_name).into(),
                    vec![gpe_features[1]; series.len()]
                ));
            }
        }

        // Create DataFrame with all enhanced features
        let mut enhanced_df = DataFrame::new(
            enhanced_features.into_iter().map(Column::Series).collect()
        )?;
        
        // Add original features and metadata
        enhanced_df.extend(&df)?;

        // Compute aggregated features
        Ok(enhanced_df.lazy()
            .group_by([col("symbol_id")])
            .agg([
                self.compute_time_features(),
                self.compute_cross_features(),
                self.compute_gpe_features(),
            ].concat())
            .collect()?)
    }

    fn compute_time_features(&self) -> Vec<Expr> {
        vec![
            col("feature_*").rolling_mean(
                RollingOptionsFixedWindow {
                    window_size: 24,
                    min_periods: 1,
                    weights: None,
                    center: false,
                    fn_params: None,
                }
            ),
            col("feature_*").rolling_std(
                RollingOptionsFixedWindow {
                    window_size: 24,
                    min_periods: 1,
                    weights: None,
                    center: false,
                    fn_params: None,
                }
            ),
            col("feature_*").shift(lit(1)),
            (col("feature_*") - col("feature_*").shift(lit(1))) / 
                col("feature_*").shift(lit(1)) * lit(100.0),
        ]
    }

    fn compute_cross_features(&self) -> Vec<Expr> {
        vec![
            col("feature_*").mean(),
            col("feature_*").std(1),
            col("feature_*").min(),
            col("feature_*").max(),
        ]
    }

    fn compute_gpe_features(&self) -> Vec<Expr> {
        vec![
            col("*_gpe_entropy").mean(),
            col("*_gpe_stability").mean(),
            col("*_gpe_entropy").std(1),
            col("*_gpe_stability").std(1),
        ]
    }
} 